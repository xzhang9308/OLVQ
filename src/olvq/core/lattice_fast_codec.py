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
from olvq.core.e8_fast_codec import _padded, _wavefronts, _parameters, _conditions as _e8_conditions, _symbol_hash, _state_hash, _cdf_bits, reconstruct_symbols as _e8_reconstruct
from olvq.core.e8_coset_codec import cpu_parameters, E8CosetCodec
from olvq.core.e8_coset_entropy import CosetSymbols, CosetEntropyModel, indexes_to_coset, coset_to_indexes, SAFE_RESIDUAL_LIMIT
from olvq.core.nonorthogonal_lattice8 import nearest_lattice_point
MAGIC = b'LFC2'
HEADER = struct.Struct('>4s8sIIIII')
ROUTES = ('e8', 'orthogonal', 'nonorthogonal')

FAMILIES = ('bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn', 'tcm')
AR_FAMILIES = ('mbt2018', 'cheng2020-anchor', 'cheng2020-attn')

def _check_model(model):
    if model.training or model.net.training:
        raise ValueError('The frozen lattice model must be in evaluation mode')
    if getattr(model, 'family', None) not in FAMILIES or getattr(model, 'route', None) not in ROUTES:
        raise ValueError('Unsupported backbone or lattice route')
    expected = model.net.num_slices if model.family == 'tcm' else 1
    if len(model.cells) != expected:
        raise ValueError('Cell count differs from the backbone layout')
    if model.route == 'e8' and any(c.companding_law != 'none' or c.lattice_shape.mode != 'rotation' for c in model.cells):
        raise ValueError('E8 requires frozen rotation without companding')

def _padding(model, h, w):
    # Preserve the existing Cheng Attention codec convention. Newly supported
    # families use the numerical evaluator's existing right/bottom padding.
    if model.family == 'cheng2020-attn':
        from compressai.ops import compute_padding
        padding, unpadding = compute_padding(h, w, min_div=128)
        return tuple(padding), tuple(unpadding)
    unit = 128 if model.family == 'tcm' else 64
    minimum = 256 if model.family == 'tcm' else 64
    ph, pw = max(minimum, (h+unit-1)//unit*unit), max(minimum, (w+unit-1)//unit*unit)
    padding = (0, pw-w, 0, ph-h)
    return padding, tuple(-v for v in padding)

def _pad_image(model, x):
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3 or min(x.shape[-2:]) < 1 or not bool(torch.isfinite(x).all()):
        raise ValueError('Expected one finite nonempty RGB image [1,3,H,W]')
    padding, unpadding = _padding(model, *x.shape[-2:])
    return F.pad(x, padding), padding, unpadding

def _analysis(model, x):
    y = model.net.g_a(x)
    z = model.net.h_a(y.abs() if model.family == 'bmshj2018-hyperprior' else y)
    return y, z

def _visit_latents(model, z_hat, visit, y=None):
    """Decoder-known traversal; visit(i, source_or_none, scales, means) returns q.

    Preserve tensor shapes and operation order of the frozen model, including
    TCM's slice support and post-quantization LRP. No source data is used by decode.
    """
    net = model.net
    h, w = z_hat.shape[-2] * 4, z_hat.shape[-1] * 4
    if model.family == 'tcm':
        ls, lm = net.h_scale_s(z_hat), net.h_mean_s(z_hat)
        hats = []
        sources = y.chunk(net.num_slices, 1) if y is not None else [None] * net.num_slices
        for i, source in enumerate(sources):
            support = hats if net.max_support_slices < 0 else hats[:net.max_support_slices]
            ms = net.atten_mean[i](torch.cat([lm] + support, 1))
            ss = net.atten_scale[i](torch.cat([ls] + support, 1))
            means = net.cc_mean_transforms[i](ms)[:, :, :h, :w]
            scales = net.cc_scale_transforms[i](ss)[:, :, :h, :w]
            q = visit(i, source, scales, means)
            q = q + .5 * torch.tanh(net.lrp_transforms[i](torch.cat([ms, q], 1)))
            hats.append(q)
        return torch.cat(hats, 1)
    params = net.h_s(z_hat)
    if model.family not in AR_FAMILIES:
        if model.family == 'bmshj2018-hyperprior':
            scales, means = params, torch.zeros_like(params)
        else:
            scales, means = params.chunk(2, 1)
        return visit(0, y, scales, means)
    y_hat = torch.zeros((1, model.cells[0].channels, h, w), dtype=params.dtype, device=params.device) if y is None else torch.zeros_like(y)
    for rows, cols in _wavefronts(h, w, net.context_prediction.kernel_size[0] // 2, params.device):
        scales, means = _parameters(net, y_hat, params, rows, cols)
        source = None if y is None else y[:, :, rows, cols].unsqueeze(-1)
        y_hat[:, :, rows, cols] = visit(0, source, scales, means).squeeze(-1)
    return y_hat

def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def _versions(module):
    return tuple(((name, id(value), value._version, str(value.device), str(value.dtype)) for (name, value) in list(module.named_parameters()) + list(module.named_buffers())))

class FrozenLatticeAdapter:
    """A lossless integer interface; decoder never sees the source latent."""

    def __init__(self, model, cell_index=0):
        _check_model(model)
        (self.model, self.cell, self.route) = (model, model.cells[cell_index], model.route)
        self.groups = self.cell.channels // 8
        self.device = next(model.parameters()).device
        with torch.no_grad():
            if self.route == 'e8':
                (self.rotations, self.gains) = self.cell._geometry(torch.ones((self.groups, 8), device=self.device))
                self.bases = self.cell.quantizer.basis[None] @ self.rotations * self.gains[:, None, None]
            elif self.route == 'orthogonal':
                (self.rotations, self.steps) = self.cell.geometry()
                self.bases = self.steps[:, :, None] * self.rotations
            else:
                self.bases = self.cell.basis()
                self.facets = self.cell._group_facets(self.bases)

    def conditions(self, scales):
        if self.route == 'e8':
            return _e8_conditions(self.cell, scales)
        (b, c, h, w) = scales.shape
        sigma = self.cell.gaussian_conditional.lower_bound_scale(scales)
        sigma = sigma.permute(0, 2, 3, 1).reshape(-1, c)
        order = torch.argsort(sigma, dim=-1, descending=True, stable=True)
        inverse = torch.argsort(order, dim=-1)
        sigma = sigma.gather(1, order).reshape(-1, 8)
        groups = torch.arange(len(sigma), device=scales.device) % self.groups
        return (sigma, (order, inverse), (b, c, h, w), groups)

    def _restore(self, values, means, condition):
        (_, (_, inverse), (b, c, h, w), _) = condition
        packed = torch.stack(values, 1).reshape(-1, c).gather(1, inverse)
        return packed.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous() + means

    def quantize(self, y, scales, means, condition=None):
        condition = self.conditions(scales) if condition is None else condition
        if self.route == 'e8':
            (q, symbols, _) = self.cell._olvq_quantize_with_means(y, means, scales)
            return (q, symbols.reshape(-1, 8).to(torch.int64))
        (_, (order, _), (_, c, _, _), _) = condition
        values = (y - means).permute(0, 2, 3, 1).reshape(-1, c)
        values = values.gather(1, order).reshape(-1, self.groups, 8)
        (reconstructions, symbols) = ([], [])
        for g in range(self.groups):
            if self.route == 'orthogonal':
                count = len(values)
                (q, indexes) = self.cell.quantize_flat(values[:, g], self.rotations[g].expand(count, 8, 8), self.steps[g].expand(count, 8), False)
            else:
                (q, indexes, _) = nearest_lattice_point(values[:, g], self.bases[g], self.facets[g])
            reconstructions.append(q.to(y.dtype))
            symbols.append(indexes.to(torch.int64))
        return (self._restore(reconstructions, means, condition), torch.stack(symbols, 1).reshape(-1, 8))

    def reconstruct(self, symbols, scales, means, condition=None):
        condition = self.conditions(scales) if condition is None else condition
        sigma = condition[0]
        if symbols.shape != sigma.shape or symbols.dtype != torch.int64:
            raise ValueError('Decoder symbols must be int64 with the grouped scale shape')
        if self.route == 'e8':
            return _e8_reconstruct(self.cell, symbols.to(self.device), scales, means)
        symbols = symbols.to(self.device).reshape(-1, self.groups, 8)
        values = []
        for g in range(self.groups):
            if self.route == 'orthogonal':
                n = len(symbols)
                q = torch.bmm((symbols[:, g].to(self.steps) * self.steps[g])[:, None], self.rotations[g].expand(n, 8, 8)).squeeze(1)
            else:
                q = symbols[:, g].to(self.bases) @ self.bases[g]
            values.append(q.to(means.dtype))
        return self._restore(values, means, condition)

    def to_wire(self, symbols):
        if self.route == 'e8':
            return indexes_to_coset(symbols, self.cell.quantizer.basis)
        return symbols

    def from_wire(self, wire):
        if self.route == 'e8':
            return coset_to_indexes(wire, basis_inv=self.cell.quantizer.basis_inv, basis=self.cell.quantizer.basis).to(torch.int64)
        return wire

def wire_symbols(model, symbols):
    """Map captured integer coordinates into the route's lossless wire form."""
    if model.route == 'e8':
        return indexes_to_coset(symbols, model.cells[0].quantizer.basis)
    return symbols

def _adapters(model):
    adapters = [FrozenLatticeAdapter(model, i) for i in range(len(model.cells))]
    offsets = []
    total = 0
    for adapter in adapters:
        offsets.append(total)
        total += adapter.groups
    return adapters, offsets

def _geometry(model):
    adapters, _ = _adapters(model)
    if model.route == 'e8':
        return dict(rotations=torch.cat([a.rotations for a in adapters]), gains=torch.cat([a.gains for a in adapters]))
    return dict(bases=torch.cat([a.bases for a in adapters]))

def create_entropy_head(model, **kwargs):
    """One head per checkpoint, with distinct geometry/group IDs for each slice."""
    adapters, _ = _adapters(model)
    groups = sum(a.groups for a in adapters)
    geometry = _geometry(model)
    if model.route == 'e8':
        head = CosetEntropyModel(groups, **kwargs).to(adapters[0].device)
        head.configure_geometry(geometry['rotations'], geometry['gains'])
    else:
        from olvq.core.lattice_fast_entropy import IntegerLatticeEntropyModel
        head = IntegerLatticeEntropyModel(groups, **kwargs).to(adapters[0].device)
        head.configure_geometry(geometry['bases'])
    return head

@torch.no_grad()
def frozen_reference_capture(model, x):
    """Run the original public forward path, disabling only probability integrals."""
    _check_model(model)
    padded, padding, unpadding = _pad_image(model, x)
    y, z = _analysis(model, padded)
    z_hat, zp = model.net.entropy_bottleneck(z)
    previous, rows = model._capture_enabled, model._captured
    captured = {}
    def before_synthesis(module, inputs):
        captured['y_hat'] = inputs[0].detach().clone()
    handle = model.net.g_s.register_forward_pre_hook(before_synthesis)
    model._capture_enabled, model._captured = True, []
    try:
        output = model(padded)
    finally:
        model._capture_enabled, model._captured = previous, rows
        handle.remove()
    return dict(x_hat=F.pad(output['x_hat'], unpadding).clamp(0, 1), y_hat=captured['y_hat'], z_hat=z_hat,
                padding=padding, z_ideal_bits=float(-zp.double().log2().sum()))

@torch.no_grad()
def capture_lattice_examples(model, x, *, verify_reference=True):
    adapters, offsets = _adapters(model)
    if x.device != adapters[0].device:
        raise ValueError('Image and frozen model must share a device')
    _sync(x.device)
    tick = time.perf_counter()
    padded, padding, unpadding = _pad_image(model, x)
    y, z = _analysis(model, padded)
    z_hat, z_likelihood = model.net.entropy_bottleneck(z)
    symbols, logs, groups, front_sizes = [], [], [], []
    def visit(i, source, scales, means):
        adapter = adapters[i]
        condition = adapter.conditions(scales)
        q, indexes = adapter.quantize(source, scales, means, condition)
        if verify_reference and not torch.equal(q, adapter.reconstruct(indexes, scales, means, condition)):
            raise RuntimeError('Integer reconstruction changes original quantization')
        symbols.append(indexes.cpu())
        logs.append(condition[0].log().cpu())
        groups.append((condition[3] + offsets[i]).cpu())
        front_sizes.append(len(indexes))
        return q
    y_hat = _visit_latents(model, z_hat, visit, y)
    x_hat = F.pad(model.net.g_s(y_hat), unpadding).clamp(0, 1)
    _sync(x.device)
    indexes = torch.cat(symbols)
    result = dict(symbols=indexes, log_scales=torch.cat(logs), group_ids=torch.cat(groups), x_hat=x_hat, y_hat=y_hat,
        z_hat=z_hat, padding=padding, original_shape=tuple(x.shape[-2:]), pixels=int(x.shape[-2]*x.shape[-1]),
        z_ideal_bits=float(-z_likelihood.double().log2().sum()), symbol_sha256=_symbol_hash(indexes), front_sizes=front_sizes,
        capture_seconds=time.perf_counter()-tick, reference_bitexact=None, reference_max_abs_error=None)
    if verify_reference:
        reference = frozen_reference_capture(model, x)
        exact = all(torch.equal(result[k], reference[k]) for k in ('x_hat', 'y_hat', 'z_hat'))
        result.update(reference_bitexact=exact, reference_max_abs_error=float((x_hat-reference['x_hat']).abs().max()))
        if not exact:
            raise RuntimeError('Fast adapter changes original frozen reconstruction')
    return result

class LatticeFastCodec:
    """Three-route estimator and real rANS codec, one state over all fronts.

    Fast probability estimates use GPU NLL when the model is on GPU. Integer
    CDFs and rANS use CPU. No probability integral or head fitting occurs here.
    Model loading and CDF construction precede codec timing. Optional correctness
    checks add overhead; use collect_stats=False/verify_reference=False for timing.
    """

    def __init__(self, model, entropy_model, model_id, *, threads=1, max_pixels=64 * 1024 * 1024):
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError('threads must be a positive integer')
        torch.set_num_threads(threads)
        self.model = model.eval()
        self.adapters, self.offsets = _adapters(model)
        self.adapter = self.adapters[0]
        (self.device, self.route) = (self.adapter.device, model.route)
        self.entropy_model = entropy_model.to(self.device).eval()
        if self.route == 'e8':
            equal = isinstance(entropy_model, CosetEntropyModel) and torch.equal(entropy_model.rotations, _geometry(model)['rotations']) and torch.equal(entropy_model.gains, _geometry(model)['gains'])
        else:
            from olvq.core.lattice_fast_entropy import IntegerLatticeEntropyModel
            equal = isinstance(entropy_model, IntegerLatticeEntropyModel) and torch.equal(entropy_model.bases, _geometry(model)['bases'].to(entropy_model.bases))
        if not equal:
            raise ValueError('Entropy head type/geometry differs from the frozen route')
        for module in (model, entropy_model):
            for p in module.parameters():
                p.requires_grad_(False)
        self.max_pixels = max_pixels
        eb = model.net.entropy_bottleneck
        args = {'force': True}
        if 'update_quantiles' in inspect.signature(eb.update).parameters:
            args['update_quantiles'] = False
        eb.update(**args)
        identity = dict(profile='lattice-fast-codec-v2', family=model.family, route=self.route, checkpoint=str(model_id), device_profile=self.device.type, model_sha256=_state_hash(model), entropy_sha256=_state_hash(entropy_model), entropy=entropy_model.configuration())
        self.full_model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.model_id = bytes.fromhex(self.full_model_id)[:8]
        self._versions = (_versions(model), _versions(entropy_model))
        self.tables = None

    def _check(self):
        _check_model(self.model)
        if self._versions != (_versions(self.model), _versions(self.entropy_model)):
            raise RuntimeError('Frozen model/head changed after codec construction')

    def _tables(self):
        if self.tables is None:
            self.tables = self.entropy_model.build_coding_tables()
        return self.tables

    def capture(self, x, *, verify_reference=True):
        self._check()
        return capture_lattice_examples(self.model, x, verify_reference=verify_reference)

    def _decode_batch(self, decoder, indexes):
        (local, cdfs, lengths, offsets) = self.tables.subset(indexes)
        return torch.tensor(decoder.decode_stream(local, cdfs, lengths, offsets), dtype=torch.int64)

    def _decode_front(self, decoder, prepared):
        if self.route == 'e8':
            return E8CosetCodec._decode_front(self, decoder, prepared)
        return self.entropy_model.decode_front(decoder, prepared)

    @torch.no_grad()
    def estimate(self, x, *, integer_cdf=False, verify_reference=False, chunk=16384):
        """Whole-image fast NLL (+ z), optionally exact integer-CDF+bypass cost.

        No bitstream is created. ``integer_cdf_bits`` excludes image header,
        finite-state flush and byte alignment. Continuous NLL is not the old
        Voronoi integral and must be labelled as this learned model's rate.
        """
        if not isinstance(chunk, int) or chunk < 1:
            raise ValueError('chunk must be positive')
        if integer_cdf:
            self._tables()
        _sync(self.device)
        tick = time.perf_counter()
        capture = self.capture(x, verify_reference=verify_reference)
        (y_bits, y_cdf) = (0.0, 0.0)
        sizes = capture['front_sizes'] if integer_cdf else [min(chunk, len(capture['symbols']) - s) for s in range(0, len(capture['symbols']), chunk)]
        start = 0
        for count in sizes:
            symbols = capture['symbols'][start:start + count].to(self.device)
            logs = capture['log_scales'][start:start + count].to(self.device)
            groups = capture['group_ids'][start:start + count].to(self.device)
            start += count
            wire = self.adapter.to_wire(symbols)
            y_bits += float(self.entropy_model.nll_bits(wire, logs, groups).double().sum())
            if integer_cdf:
                prepared = cpu_parameters(self.entropy_model.prepare(logs, groups, validate=False))
                cpu_wire = CosetSymbols(*(v.cpu() for v in wire)) if self.route == 'e8' else wire.cpu()
                for (residuals, indexes) in self.entropy_model._coding_batches(cpu_wire, prepared):
                    y_cdf += self.tables.bit_cost(residuals, indexes)
        result = dict(ideal_y_bits=y_bits, ideal_z_bits=capture['z_ideal_bits'], ideal_bits=y_bits + capture['z_ideal_bits'], rate_kind='learned-discrete-nll', x_hat=capture['x_hat'], symbol_sha256=capture['symbol_sha256'], reference_bitexact=capture['reference_bitexact'], pixels=capture['pixels'])
        if integer_cdf:
            (padded, _, _) = _pad_image(self.model, x)
            eb = self.model.net.entropy_bottleneck
            _, z = _analysis(self.model, padded)
            symbols = eb.quantize(z, 'symbols', eb._get_medians().detach()).reshape(-1).tolist()
            channels = torch.arange(z.shape[1]).view(1, -1, 1, 1).expand_as(z)
            z_cdf = _cdf_bits(symbols, channels.reshape(-1).tolist(), eb._quantized_cdf.tolist(), eb._cdf_length.tolist(), eb._offset.tolist())
            result.update(integer_y_bits=y_cdf, integer_z_bits=z_cdf, estimated_bits=y_cdf + z_cdf)
        _sync(self.device)
        result['estimate_seconds'] = time.perf_counter() - tick
        return result

    @torch.no_grad()
    def compress(self, x, *, collect_stats=True):
        from compressai.ans import BufferedRansEncoder
        self._check()
        self._tables()
        if x.device != self.device:
            raise ValueError('Image and frozen model must share a device')
        _sync(self.device)
        tick = time.perf_counter()
        (padded, padding, _) = _pad_image(self.model, x)
        (h, w) = map(int, x.shape[-2:])
        if padded.shape[-2] * padded.shape[-1] > self.max_pixels:
            raise ValueError('Image exceeds pixel limit')
        (net, eb) = (self.model.net, self.model.net.entropy_bottleneck)
        y, z = _analysis(self.model, padded)
        (z_reference, zp) = eb(z)
        z_strings = eb.compress(z)
        if len(z_strings) != 1:
            raise RuntimeError('Expected one hyperlatent stream')
        z_stream = z_strings[0]
        z_hat = eb.decompress([z_stream], z.shape[-2:])
        if not torch.equal(z_reference, z_hat):
            raise RuntimeError('Hyperlatent coding changes frozen reconstruction')
        if tuple(z.shape[-2:]) != (padded.shape[-2] // 64, padded.shape[-1] // 64):
            raise RuntimeError('Unexpected hyperlatent spatial factor')
        (rans_symbols, rans_indexes, all_symbols) = ([], [], [])
        y_bits = y_cdf = statistics_seconds = 0.0
        def visit(i, source, scales, means):
            nonlocal y_bits, y_cdf, statistics_seconds
            adapter = self.adapters[i]
            condition = adapter.conditions(scales)
            (q, symbols) = adapter.quantize(source, scales, means, condition)
            wire = adapter.to_wire(symbols)
            (logs, groups) = (condition[0].log(), condition[3] + self.offsets[i])
            prepared = cpu_parameters(self.entropy_model.prepare(logs, groups, validate=False))
            cpu_wire = CosetSymbols(*(v.cpu() for v in wire)) if self.route == 'e8' else wire.cpu()
            for (residuals, indexes) in self.entropy_model._coding_batches(cpu_wire, prepared):
                if bool((residuals.abs() > SAFE_RESIDUAL_LIMIT).any()):
                    raise ValueError('Symbol exceeds safe bypass range')
                rans_symbols.extend(residuals.tolist())
                rans_indexes.extend(indexes.tolist())
                if collect_stats:
                    st = time.perf_counter()
                    y_cdf += self.tables.bit_cost(residuals, indexes)
                    statistics_seconds += time.perf_counter() - st
            if collect_stats:
                st = time.perf_counter()
                if not torch.equal(q, adapter.reconstruct(symbols, scales, means, condition)):
                    raise RuntimeError('Integer decoding changes original quantization')
                y_bits += float(self.entropy_model.nll_bits(wire, logs, groups).double().sum())
                _sync(self.device)
                statistics_seconds += time.perf_counter() - st
            all_symbols.append(symbols.cpu())
            return q
        y_hat = _visit_latents(self.model, z_hat, visit, y)
        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes(rans_symbols, rans_indexes, self.tables.cdfs, self.tables.lengths, self.tables.offsets)
        y_stream = encoder.flush()
        payload = z_stream + y_stream
        blob = HEADER.pack(MAGIC, self.model_id, h, w, len(z_stream), len(y_stream), zlib.crc32(payload)) + payload
        _sync(self.device)
        result = dict(encode_seconds=time.perf_counter() - tick, statistics_seconds=statistics_seconds, actual_bits=len(blob) * 8, header_bits=HEADER.size * 8, y_stream_bits=len(y_stream) * 8, z_stream_bits=len(z_stream) * 8, symbol_sha256=_symbol_hash(torch.cat(all_symbols)), y_hat=y_hat, z_hat=z_hat, padding=padding, model_id=self.full_model_id, entropy_device=str(self.device), reconstruction_device=str(self.device))
        if collect_stats:
            zs = eb.quantize(z, 'symbols', eb._get_medians().detach()).reshape(-1).tolist()
            channels = torch.arange(z.shape[1]).view(1, -1, 1, 1).expand_as(z)
            z_cdf = _cdf_bits(zs, channels.reshape(-1).tolist(), eb._quantized_cdf.tolist(), eb._cdf_length.tolist(), eb._offset.tolist())
            z_bits = float(-zp.double().log2().sum())
            result.update(ideal_y_bits=y_bits, ideal_z_bits=z_bits, ideal_bits=y_bits + z_bits, integer_y_bits=y_cdf, integer_z_bits=z_cdf, estimated_bits=y_cdf + z_cdf)
        return (blob, result)

    def _parse(self, blob):
        if not isinstance(blob, bytes) or len(blob) < HEADER.size:
            raise ValueError('Truncated lattice stream header')
        (magic, identity, h, w, nz, ny, crc) = HEADER.unpack_from(blob)
        if magic != MAGIC or identity != self.model_id:
            raise ValueError('Bitstream format or shared model/device profile differs')
        if h < 1 or w < 1 or h * w > self.max_pixels:
            raise ValueError('Invalid image dimensions')
        if nz < 4 or ny < 8 or HEADER.size + nz + ny != len(blob):
            raise ValueError('Invalid entropy stream lengths')
        payload = blob[HEADER.size:]
        if zlib.crc32(payload) != crc:
            raise ValueError('Entropy payload checksum mismatch')
        return (h, w, payload[:nz], payload[nz:])

    @torch.no_grad()
    def decompress(self, blob):
        from compressai.ans import RansDecoder
        self._check()
        self._tables()
        _sync(self.device)
        tick = time.perf_counter()
        (h, w, z_stream, y_stream) = self._parse(blob)
        (padding, unpadding) = _padding(self.model, h, w)
        (ph, pw) = (h + padding[2] + padding[3], w + padding[0] + padding[1])
        if ph * pw > self.max_pixels:
            raise ValueError('Padded image exceeds pixel limit')
        net = self.model.net
        z_hat = net.entropy_bottleneck.decompress([z_stream], (ph // 64, pw // 64))
        decoder = RansDecoder()
        decoder.set_stream(y_stream)
        all_symbols = []
        def visit(i, source, scales, means):
            adapter = self.adapters[i]
            condition = adapter.conditions(scales)
            prepared = cpu_parameters(self.entropy_model.prepare(condition[0].log(), condition[3] + self.offsets[i], validate=False))
            wire = self._decode_front(decoder, prepared)
            symbols = adapter.from_wire(wire).to(self.device)
            all_symbols.append(symbols.cpu())
            return adapter.reconstruct(symbols, scales, means, condition)
        y_hat = _visit_latents(self.model, z_hat, visit)
        x_hat = F.pad(net.g_s(y_hat), unpadding).clamp(0, 1)
        _sync(self.device)
        return (x_hat, dict(decode_seconds=time.perf_counter() - tick, symbol_sha256=_symbol_hash(torch.cat(all_symbols)), y_hat=y_hat, z_hat=z_hat, padding=tuple(padding), original_shape=(h, w), model_id=self.full_model_id, entropy_device=str(self.device), reconstruction_device=str(self.device)))
