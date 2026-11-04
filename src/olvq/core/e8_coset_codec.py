# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import hashlib
import inspect
import json
import struct
import time
import zlib
import torch
from torch.nn import functional as F
from olvq.core.e8_fast_codec import _check_model, _padded, _wavefronts, _parameters, _conditions, _symbol_hash, _state_hash, _cdf_bits, capture_e8_examples
from olvq.core.conditional_lattice import _closest_e8_point
from olvq.core.e8_coset_entropy import CosetSymbols, points_to_coset, coset_to_points, restore_last, SAFE_RESIDUAL_LIMIT
MAGIC = b'E8C3'
HEADER = struct.Struct('>4s8sIIIII')

def cpu_parameters(prepared):
    """One GPU-to-CPU copy for all forty parameters of a causal front."""
    keys = tuple(prepared)
    sizes = [value.shape[1:] for value in prepared.values()]
    widths = [value[0].numel() for value in prepared.values()]
    count = len(next(iter(prepared.values())))
    packed = torch.cat([value.reshape(count, -1) for value in prepared.values()], 1).cpu()
    return {key: value.reshape(count, *shape) for (key, shape, value) in zip(keys, sizes, packed.split(widths, 1))}

class E8CosetCodec:

    def __init__(self, model, entropy_model, model_id, *, threads=1, max_pixels=64 * 1024 * 1024):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError('threads must be a positive integer')
        torch.set_num_threads(threads)
        self.model = model.eval()
        self.device = next(model.parameters()).device
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('The original frozen reconstruction requires CUDA')
        _check_model(model)
        self.entropy_model = entropy_model.to(self.device).eval()
        for module in (model, entropy_model):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.max_pixels = max_pixels
        cell = model.cells[0]
        if cell.lattice_shape.mode != 'rotation':
            raise ValueError('Only the frozen scale-independent rotation is supported')
        with torch.no_grad():
            (self.rotations, self.gains) = cell._geometry(torch.ones((cell.lattice_groups, 8), device=self.device))
        if not torch.equal(entropy_model.rotations, self.rotations) or not torch.equal(entropy_model.gains, self.gains):
            raise ValueError('Entropy head geometry differs from the frozen cell')
        self.basis_inv_cpu = cell.quantizer.basis_inv.detach().double().cpu()
        self.full_model_id = hashlib.sha256(('e8-coset-cuda-v3/' + str(model_id) + '/' + _state_hash(entropy_model) + '/' + json.dumps(entropy_model.configuration(), sort_keys=True)).encode()).hexdigest()
        self.model_id = bytes.fromhex(self.full_model_id)[:8]
        eb = model.net.entropy_bottleneck
        args = {'force': True}
        if 'update_quantiles' in inspect.signature(eb.update).parameters:
            args['update_quantiles'] = False
        eb.update(**args)
        self.tables = entropy_model.build_coding_tables()

    def capture(self, x, *, verify_reference=True):
        return capture_e8_examples(self.model, x.to(self.device), verify_reference=verify_reference)

    def _reconstruct(self, points, means, condition):
        (sigma, grouping, layout, groups) = condition
        points = points.to(device=self.device, dtype=sigma.dtype)
        residual = torch.bmm(points.unsqueeze(1), self.rotations[groups]).squeeze(1)
        residual = residual * self.gains[groups, None]
        return self.model.cells[0]._unflatten(residual, layout, grouping) + means

    def _quantize(self, y, means, condition):
        (sigma, grouping, _, groups) = condition
        cell = self.model.cells[0]
        (residual, _) = cell._flatten(y - means, grouping)
        canonical = torch.linalg.solve(self.rotations[groups].transpose(-1, -2), residual.unsqueeze(-1)).squeeze(-1)
        canonical = canonical / self.gains[groups, None]
        points = _closest_e8_point(canonical)
        return (self._reconstruct(points, means, condition), points)

    def _prepare(self, condition):
        (sigma, _, _, groups) = condition
        return cpu_parameters(self.entropy_model.prepare(sigma.log(), groups, validate=False))

    def _decode_batch(self, decoder, indexes):
        (local, cdfs, lengths, offsets) = self.tables.subset(indexes)
        return torch.tensor(decoder.decode_stream(local, cdfs, lengths, offsets), dtype=torch.int64)

    def _decode_front(self, decoder, prepared):
        count = len(prepared['coset_logits'])
        result = CosetSymbols(torch.zeros(count, dtype=torch.int64), torch.zeros((count, 7), dtype=torch.int64), torch.zeros(count, dtype=torch.int64))
        active = self._decode_batch(decoder, self.tables.binary_indexes(prepared['nonzero_logits'])).bool() if self.entropy_model.zero_flag else torch.ones(count, dtype=torch.bool)
        if not bool(active.any()):
            return result
        prepared = {key: value[active] for (key, value) in prepared.items()}
        coset = self._decode_batch(decoder, self.tables.binary_indexes(prepared['coset_logits']))
        (means, scales) = self.entropy_model.select_coordinates(prepared, coset)
        (indexes, integer) = self.tables.coordinate_indexes(means, scales)
        first = self._decode_batch(decoder, indexes.reshape(-1)).reshape(-1, 7) + integer
        (means, scales) = self.entropy_model.select_last(prepared, coset, first)
        (indexes, integer) = self.tables.coordinate_indexes(means, scales)
        last = self._decode_batch(decoder, indexes) + integer
        if self.entropy_model.zero_flag:
            last = restore_last(coset, first, last)
        (result.coset[active], result.first[active], result.last[active]) = (coset, first, last)
        return result

    def _indexes(self, points):
        return torch.round(points.double().cpu() @ self.basis_inv_cpu).to(torch.int64)

    @torch.no_grad()
    def compress(self, x, *, collect_stats=True):
        from compressai.ans import BufferedRansEncoder
        if x.device != self.device:
            raise ValueError('Image and frozen model must share the CUDA device')
        _check_model(self.model)
        torch.cuda.synchronize(self.device)
        tick = time.perf_counter()
        (padded, padding, _) = _padded(x)
        (height, width) = map(int, x.shape[-2:])
        if padded.shape[-2] * padded.shape[-1] > self.max_pixels:
            raise ValueError('Image exceeds pixel limit')
        (net, cell) = (self.model.net, self.model.cells[0])
        y = net.g_a(padded)
        z = net.h_a(y)
        eb = net.entropy_bottleneck
        (z_reference, z_likelihood) = eb(z)
        z_streams = eb.compress(z)
        if len(z_streams) != 1:
            raise RuntimeError('Expected a single hyperlatent stream')
        z_stream = z_streams[0]
        z_hat = eb.decompress([z_stream], z.shape[-2:])
        if not torch.equal(z_reference, z_hat):
            raise RuntimeError('Hyperlatent coding changes frozen reconstruction')
        if tuple(z.shape[-2:]) != (padded.shape[-2] // 64, padded.shape[-1] // 64):
            raise RuntimeError('Unexpected Cheng hyperlatent spatial factor')
        (params, y_hat) = (net.h_s(z_hat), torch.zeros_like(y))
        (all_points, rans_symbols, rans_indexes) = ([], [], [])
        ideal_y_bits = integer_y_bits = statistics_seconds = 0.0
        pad = net.context_prediction.kernel_size[0] // 2
        fronts = 0
        for (rows, cols) in _wavefronts(*y.shape[-2:], pad, y.device):
            fronts += 1
            (scales, means) = _parameters(net, y_hat, params, rows, cols)
            condition = _conditions(cell, scales)
            (q, points) = self._quantize(y[:, :, rows, cols].unsqueeze(-1), means, condition)
            prepared = self._prepare(condition)
            points_cpu = points.cpu()
            symbols = points_to_coset(points_cpu)
            for (values, indexes) in self.entropy_model._coding_batches(symbols, prepared):
                if bool((values.abs() > SAFE_RESIDUAL_LIMIT).any()):
                    raise ValueError('Symbol exceeds safe bypass range')
                rans_symbols.extend(values.tolist())
                rans_indexes.extend(indexes.tolist())
                if collect_stats:
                    stat_tick = time.perf_counter()
                    integer_y_bits += self.tables.bit_cost(values, indexes)
                    statistics_seconds += time.perf_counter() - stat_tick
            if collect_stats:
                stat_tick = time.perf_counter()
                (original_q, original_idx, _) = cell._olvq_quantize_with_means(y[:, :, rows, cols].unsqueeze(-1), means, scales)
                if not torch.equal(q, original_q) or not torch.equal(original_idx.reshape(-1, 8).to(torch.int64).cpu(), self._indexes(points_cpu)):
                    raise RuntimeError('Cached geometry changes original quantization')
                ideal_y_bits += float(self.entropy_model.nll_bits(CosetSymbols(*(value.to(self.device) for value in symbols)), condition[0].log(), condition[3]).double().sum())
                torch.cuda.synchronize(self.device)
                statistics_seconds += time.perf_counter() - stat_tick
            all_points.append(points_cpu)
            y_hat[:, :, rows, cols] = q.squeeze(-1)
        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes(rans_symbols, rans_indexes, self.tables.cdfs, self.tables.lengths, self.tables.offsets)
        y_stream = encoder.flush()
        payload = z_stream + y_stream
        bitstream = HEADER.pack(MAGIC, self.model_id, height, width, len(z_stream), len(y_stream), zlib.crc32(payload)) + payload
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - tick
        stats = dict(encode_seconds=elapsed, statistics_seconds=statistics_seconds, actual_bits=8 * len(bitstream), header_bits=HEADER.size * 8, y_stream_bits=len(y_stream) * 8, z_stream_bits=len(z_stream) * 8, symbol_sha256=_symbol_hash(self._indexes(torch.cat(all_points))), y_hat=y_hat, z_hat=z_hat, padding=padding, wavefronts=fronts, model_id=self.full_model_id, entropy_device=str(self.device), reconstruction_device=str(self.device))
        if collect_stats:
            z_symbols = eb.quantize(z, 'symbols', eb._get_medians().detach()).reshape(-1).tolist()
            channels = torch.arange(z.shape[1]).view(1, -1, 1, 1).expand_as(z)
            z_cdf_bits = _cdf_bits(z_symbols, channels.reshape(-1).tolist(), eb._quantized_cdf.tolist(), eb._cdf_length.tolist(), eb._offset.tolist())
            ideal_z = float(-z_likelihood.double().log2().sum())
            stats.update(ideal_y_bits=ideal_y_bits, integer_y_bits=integer_y_bits, ideal_z_bits=ideal_z, integer_z_bits=z_cdf_bits, estimated_bits=integer_y_bits + z_cdf_bits, ideal_bits=ideal_y_bits + ideal_z)
        return (bitstream, stats)

    def _parse(self, bitstream):
        if not isinstance(bitstream, bytes) or len(bitstream) < HEADER.size:
            raise ValueError('Truncated E8 header')
        (magic, model_id, h, w, nz, ny, crc) = HEADER.unpack_from(bitstream)
        if magic != MAGIC or model_id != self.model_id:
            raise ValueError('Bitstream format or shared model mismatch')
        if h < 1 or w < 1 or h * w > self.max_pixels:
            raise ValueError('Invalid image dimensions')
        if nz < 4 or ny < 8 or HEADER.size + nz + ny != len(bitstream):
            raise ValueError('Invalid entropy stream lengths')
        payload = bitstream[HEADER.size:]
        if zlib.crc32(payload) != crc:
            raise ValueError('Entropy payload checksum mismatch')
        return (h, w, payload[:nz], payload[nz:])

    @torch.no_grad()
    def decompress(self, bitstream):
        from compressai.ans import RansDecoder
        from compressai.ops import compute_padding
        _check_model(self.model)
        torch.cuda.synchronize(self.device)
        tick = time.perf_counter()
        (height, width, z_stream, y_stream) = self._parse(bitstream)
        (padding, unpadding) = compute_padding(height, width, min_div=128)
        (ph, pw) = (height + padding[2] + padding[3], width + padding[0] + padding[1])
        if ph * pw > self.max_pixels:
            raise ValueError('Padded image exceeds pixel limit')
        (net, cell) = (self.model.net, self.model.cells[0])
        z_hat = net.entropy_bottleneck.decompress([z_stream], (ph // 64, pw // 64))
        params = net.h_s(z_hat)
        y_hat = torch.zeros((1, cell.channels, ph // 16, pw // 16), dtype=params.dtype, device=self.device)
        decoder = RansDecoder()
        decoder.set_stream(y_stream)
        all_points = []
        pad = net.context_prediction.kernel_size[0] // 2
        for (rows, cols) in _wavefronts(*y_hat.shape[-2:], pad, y_hat.device):
            (scales, means) = _parameters(net, y_hat, params, rows, cols)
            condition = _conditions(cell, scales)
            symbols = self._decode_front(decoder, self._prepare(condition))
            points = coset_to_points(symbols, dtype=params.dtype)
            y_hat[:, :, rows, cols] = self._reconstruct(points, means, condition).squeeze(-1)
            all_points.append(points)
        x_hat = F.pad(net.g_s(y_hat), unpadding).clamp(0, 1)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - tick
        return (x_hat, dict(decode_seconds=elapsed, symbol_sha256=_symbol_hash(self._indexes(torch.cat(all_points))), y_hat=y_hat, z_hat=z_hat, original_shape=(height, width), padding=tuple(padding), model_id=self.full_model_id, reconstruction_device=str(self.device), entropy_device=str(self.device)))
