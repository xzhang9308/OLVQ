# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import hashlib
import inspect
import json
import math
import struct
import time
import zlib
from typing import Any
import torch
from torch.nn import functional as F
MAGIC = b'E8F1'
HEADER = struct.Struct('>4s8sIIIII')

def _check_model(model):
    if model.training or model.net.training:
        raise ValueError('The frozen E8 model must be in evaluation mode')
    if getattr(model, 'route', None) != 'e8' or len(model.cells) != 1:
        raise ValueError('This prototype supports the controlled single-cell E8 route')
    if getattr(model, 'family', None) != 'cheng2020-attn':
        raise ValueError('The bitstream profile is registered for Cheng Attention')
    if model.cells[0].companding_law != 'none':
        raise ValueError('The frozen profile has no companding')

def _padded(x):
    from compressai.ops import compute_padding
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
        raise ValueError('Expected one RGB image [1,3,H,W]')
    (h, w) = map(int, x.shape[-2:])
    if h < 1 or w < 1 or (not bool(torch.isfinite(x).all())):
        raise ValueError('Image dimensions and values must be finite and positive')
    (padding, unpadding) = compute_padding(h, w, min_div=128)
    return (F.pad(x, padding), tuple(padding), tuple(unpadding))

def _wavefronts(height, width, pad, device):
    for t in range((pad + 1) * (height - 1) + width):
        rows = torch.arange(height, device=device)
        cols = t - (pad + 1) * rows
        keep = (cols >= 0) & (cols < width)
        (rows, cols) = (rows[keep], cols[keep])
        if len(rows):
            yield (rows, cols)

def _parameters(net, y_hat, params, rows, cols):
    """The same operations, shapes and order as GeometryRateMixin._ar."""
    (batch, _, _, _) = y_hat.shape
    k = net.context_prediction.kernel_size[0]
    pad = k // 2
    weights = net.context_prediction.weight * net.context_prediction.mask
    patches = F.pad(y_hat, (pad, pad, pad, pad)).unfold(2, k, 1).unfold(3, k, 1)
    chosen = patches[:, :, rows, cols].permute(0, 2, 1, 3, 4).reshape(batch * len(rows), -1)
    ctx = F.linear(chosen, weights.reshape(weights.shape[0], -1), net.context_prediction.bias)
    ctx = ctx.reshape(batch, len(rows), -1).permute(0, 2, 1).unsqueeze(-1)
    hyper = params[:, :, rows, cols].unsqueeze(-1)
    return net.entropy_parameters(torch.cat([hyper, ctx], 1)).chunk(2, 1)

def _conditions(cell, scales):
    bounded = cell.gaussian_conditional.lower_bound_scale(scales)
    grouping = cell._adaptive_grouping_index(bounded)
    (sigma, layout) = cell._flatten(bounded, grouping)
    groups = torch.arange(len(sigma), device=scales.device, dtype=torch.long) % cell.lattice_groups
    return (sigma, grouping, layout, groups)

def reconstruct_symbols(cell, symbols, scales, means):
    """Invert the exact integer-basis map without re-running quantization."""
    (sigma, grouping, layout, _) = _conditions(cell, scales)
    if symbols.shape != sigma.shape:
        raise ValueError('Integer symbols and decoder-known grouped scales differ')
    (shape, gain) = cell._geometry(sigma)
    points = symbols.to(sigma.dtype) @ cell.quantizer.basis.to(sigma)
    residual = torch.bmm(points.unsqueeze(1), shape).squeeze(1) * gain[:, None]
    return cell._unflatten(residual, layout, grouping) + means

def _symbol_hash(symbols):
    return hashlib.sha256(symbols.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()).hexdigest()

@torch.no_grad()
def frozen_reference_capture(model, x):
    """Run the unmodified causal reconstruction path with integration disabled."""
    _check_model(model)
    (padded, padding, unpadding) = _padded(x)
    net = model.net
    y = net.g_a(padded)
    z = net.h_a(y)
    (z_hat, z_likelihood) = net.entropy_bottleneck(z)
    previous = model._capture_enabled
    previous_rows = model._captured
    (model._capture_enabled, model._captured) = (True, [])
    try:
        (y_hat, _) = model._ar(y, net.h_s(z_hat))
    finally:
        (model._capture_enabled, model._captured) = (previous, previous_rows)
    x_hat = F.pad(net.g_s(y_hat), unpadding).clamp(0, 1)
    return {'x_hat': x_hat, 'y_hat': y_hat, 'z_hat': z_hat, 'padding': padding, 'original_shape': tuple(x.shape[-2:]), 'z_ideal_bits': float(-z_likelihood.double().log2().sum())}

@torch.no_grad()
def capture_e8_examples(model, x, *, verify_reference=True):
    """Collect CPU training examples; the frozen model may run on CPU or GPU.

    No ANS or probability integration is performed.  Example ordering is
    wavefront, spatial position, sigma-ranked group, integer coordinate.
    """
    _check_model(model)
    tick = time.perf_counter()
    (padded, padding, unpadding) = _padded(x)
    (net, cell) = (model.net, model.cells[0])
    y = net.g_a(padded)
    (z_hat, _) = net.entropy_bottleneck(net.h_a(y))
    (params, y_hat) = (net.h_s(z_hat), torch.zeros_like(y))
    (all_symbols, all_logs, all_groups) = ([], [], [])
    (height, width) = y.shape[-2:]
    pad = net.context_prediction.kernel_size[0] // 2
    for (rows, cols) in _wavefronts(height, width, pad, y.device):
        (scales, means) = _parameters(net, y_hat, params, rows, cols)
        (q, idx, _) = cell._olvq_quantize_with_means(y[:, :, rows, cols].unsqueeze(-1), means, scales)
        symbols = idx.reshape(-1, 8).to(torch.int64)
        (sigma, _, _, groups) = _conditions(cell, scales)
        if verify_reference and (not torch.equal(q, reconstruct_symbols(cell, symbols, scales, means))):
            raise RuntimeError('Integer basis inversion changes the existing quantized latent')
        y_hat[:, :, rows, cols] = q.squeeze(-1)
        all_symbols.append(symbols.cpu())
        all_logs.append(sigma.log().cpu())
        all_groups.append(groups.cpu())
    x_hat = F.pad(net.g_s(y_hat), unpadding).clamp(0, 1)
    symbols = torch.cat(all_symbols)
    result = {'symbols': symbols, 'log_scales': torch.cat(all_logs), 'group_ids': torch.cat(all_groups), 'x_hat': x_hat, 'y_hat': y_hat, 'z_hat': z_hat, 'padding': padding, 'original_shape': tuple(x.shape[-2:]), 'symbol_sha256': _symbol_hash(symbols), 'capture_seconds': time.perf_counter() - tick, 'reference_max_abs_error': None, 'reference_bitexact': None}
    if verify_reference:
        reference = frozen_reference_capture(model, x)
        error = float((x_hat - reference['x_hat']).abs().max())
        bitexact = torch.equal(y_hat, reference['y_hat']) and torch.equal(z_hat, reference['z_hat']) and torch.equal(x_hat, reference['x_hat'])
        result.update(reference_max_abs_error=error, reference_bitexact=bitexact)
        if not bitexact:
            raise RuntimeError('The fast coordinate path changes frozen causal reconstruction')
    return result

def _state_hash(module):
    digest = hashlib.sha256()
    for (name, value) in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def _cdf_bits(symbols, indexes, cdfs, lengths, offsets):
    """CDF cross entropy plus the exact CompressAI four-bit bypass payload."""
    bits = 0.0
    for (symbol, index) in zip(symbols, indexes):
        value = int(symbol) - int(offsets[index])
        sentinel = int(lengths[index]) - 2
        raw = None
        if value < 0:
            (raw, value) = (-2 * value - 1, sentinel)
        elif value >= sentinel:
            (raw, value) = (2 * (value - sentinel), sentinel)
        freq = int(cdfs[index][value + 1]) - int(cdfs[index][value])
        bits += math.log2(int(cdfs[index][lengths[index] - 1])) - math.log2(freq)
        if raw is not None:
            chunks = (raw.bit_length() + 3) // 4
            bits += 4 * (chunks + chunks // 15 + 1)
    return bits
