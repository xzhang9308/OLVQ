# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
from dataclasses import asdict
from typing import NamedTuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from olvq.core.e8_discrete_entropy import CDFTableConfig, DiscreteCDFTables, LOG2, SAFE_RESIDUAL_LIMIT
DIMENSION = 8
EXACT_TRANSFORM_LIMIT = 1 << 46
LOGISTIC_STD_TO_SCALE = math.sqrt(3.0) / math.pi

def _positive_logistic_log_mass(symbols, means, scales):
    """The shared stable formula without rechecking already-positive scales.

    ``prepare`` constructs these scales by a positive clamp.  Avoiding a
    redundant ``bool((scales <= 0).any())`` prevents a CUDA synchronization
    inside every cached-data training loss.
    """
    centered = symbols.to(means.dtype) - means
    (upper, lower) = ((centered + 0.5) / scales, (centered - 0.5) / scales)
    return F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(-1.0 / scales))

class CosetSymbols(NamedTuple):
    coset: torch.Tensor
    first: torch.Tensor
    last: torch.Tensor

def _integers(value, name):
    value = torch.as_tensor(value)
    if value.dtype == torch.bool or value.is_complex():
        raise ValueError(f'{name} must contain integers')
    if torch.is_floating_point(value):
        if not bool(torch.isfinite(value).all()) or not bool((value == value.round()).all()):
            raise ValueError(f'{name} must contain finite integers')
    if bool((value.double().abs() > EXACT_TRANSFORM_LIMIT).any()):
        raise ValueError(f'{name} exceeds the exact transform range')
    return value.to(torch.int64)

def _symbols(coset, first=None, last=None):
    if first is None and last is None:
        if not isinstance(coset, (tuple, list)) or len(coset) != 3:
            raise ValueError('symbols must be a (coset, first, last) tuple')
        (coset, first, last) = coset
    (coset, first, last) = (_integers(coset, 'coset'), _integers(first, 'first'), _integers(last, 'last'))
    if coset.ndim != 1 or first.shape != (len(coset), 7) or last.shape != coset.shape:
        raise ValueError('coset/first/last must have shapes [N], [N,7], [N]')
    if first.device != coset.device or last.device != coset.device:
        raise ValueError('Symbol tensors must share a device')
    if bool(((coset < 0) | (coset > 1)).any()):
        raise ValueError('coset must be zero or one')
    return CosetSymbols(coset, first, last)

def points_to_coset(points):
    """Losslessly represent canonical E8 points; reject points outside E8."""
    points = torch.as_tensor(points)
    if points.ndim != 2 or points.shape[1] != DIMENSION or points.is_complex():
        raise ValueError('points must have shape [N,8]')
    points = points.double()
    if not bool(torch.isfinite(points).all()) or bool((points.abs() > EXACT_TRANSFORM_LIMIT).any()):
        raise ValueError('points exceed the finite exact transform range')
    doubled = points * 2
    if not bool((doubled == doubled.round()).all()):
        raise ValueError('E8 coordinates must be integers or half-integers')
    doubled = doubled.to(torch.int64)
    coset = doubled[:, 0].remainder(2)
    if not bool((doubled.remainder(2) == coset[:, None]).all()):
        raise ValueError('An E8 point must use a single integer/half-integer coset')
    integer = torch.div(doubled - coset[:, None], 2, rounding_mode='floor')
    if bool((integer.sum(-1).remainder(2) != 0).any()):
        raise ValueError('The integer part of an E8 point must have even sum')
    first = integer[:, :7]
    parity = first.sum(-1).remainder(2)
    last = torch.div(integer[:, 7] - parity, 2, rounding_mode='floor')
    return CosetSymbols(coset, first, last)

def coset_to_points(coset, first=None, last=None, *, dtype=torch.float64):
    """Invert :func:`points_to_coset` without rounding or a quantizer call."""
    (coset, first, last) = _symbols(coset, first, last)
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError('Canonical points require a floating point dtype')
    parity = first.sum(-1).remainder(2)
    integer = torch.cat((first, (2 * last + parity)[:, None]), -1)
    return (integer.double() + coset[:, None].double() * 0.5).to(dtype)

def _basis(value, device, name):
    value = torch.as_tensor(value, device=device, dtype=torch.float64)
    if value.shape != (8, 8) or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be a finite [8,8] matrix')
    return value

def indexes_to_coset(indexes, basis):
    """Convert pre-existing integer basis indexes to the canonical wire form."""
    indexes = _integers(indexes, 'indexes')
    if indexes.ndim != 2 or indexes.shape[1] != 8:
        raise ValueError('indexes must have shape [N,8]')
    return points_to_coset(indexes.double() @ _basis(basis, indexes.device, 'basis'))

def coset_to_indexes(coset, first=None, last=None, *, basis_inv, basis=None):
    """Invert the canonical form and verify that no basis rounding was needed.

    Supplying ``basis`` additionally verifies exact reconstruction in canonical
    space, which is recommended once when accepting an existing codec basis.
    """
    points = coset_to_points(coset, first, last)
    indexes = points @ _basis(basis_inv, points.device, 'basis_inv')
    if not bool((indexes == indexes.round()).all()):
        raise ValueError('basis_inv does not map the E8 points to integer indexes')
    indexes = _integers(indexes, 'indexes')
    if basis is not None:
        rebuilt = indexes.double() @ _basis(basis, points.device, 'basis')
        if not torch.equal(rebuilt, points):
            raise ValueError('Basis conversion does not preserve the canonical points')
    return indexes

def nonzero_mask(symbols):
    (coset, first, last) = _symbols(symbols)
    return (coset != 0) | first.ne(0).any(-1) | (last != 0)

def _special_last(coset, first):
    return (coset == 0) & first.eq(0).all(-1)

def wire_last(coset, first, last):
    """Remove the impossible zero from the nonzero-vector special branch.

    Only use this mapping after an all-zero-vector flag has been encoded.
    ``t < 0`` maps to itself and ``t > 0`` maps to ``t-1``.  Thus every integer
    wire value represents exactly one nonzero last coordinate in this branch.
    """
    (coset, first, last) = _symbols(coset, first, last)
    special = _special_last(coset, first)
    if bool((special & (last == 0)).any()):
        raise ValueError('The special nonzero branch cannot contain the zero vector')
    return last - (special & (last > 0)).to(last.dtype)

def restore_last(coset, first, wire):
    (coset, first, wire) = _symbols(coset, first, wire)
    return wire + (_special_last(coset, first) & (wire >= 0)).to(wire.dtype)

class CosetCodingTables:
    """One static CDF bank shared by both binary flags and integer symbols."""

    def __init__(self, config=None, binary_bins=256, binary_logit_limit=12.0):
        self.config = CDFTableConfig() if config is None else config
        if isinstance(self.config, dict):
            self.config = CDFTableConfig(**self.config)
        self.config.validate()
        if binary_bins < 2 or binary_logit_limit <= 0:
            raise ValueError('Invalid binary CDF grid')
        (self.binary_bins, self.binary_logit_limit) = (int(binary_bins), float(binary_logit_limit))
        self.logistic = DiscreteCDFTables(self.config)
        (self.cdfs, self.lengths, self.offsets) = ([], [], [])
        for logit in np.linspace(-self.binary_logit_limit, self.binary_logit_limit, self.binary_bins, dtype=np.float64):
            probability_zero = 1.0 / (1.0 + math.exp(float(logit)))
            zero = min(65534, max(1, int(math.floor(65535 * probability_zero + 0.5))))
            self.cdfs.append([0, zero, 65535, 65536])
            self.lengths.append(4)
            self.offsets.append(0)
        self.cdfs.extend(self.logistic.cdfs)
        self.lengths.extend(self.logistic.lengths)
        self.offsets.extend(self.logistic.offsets)

    def binary_indexes(self, logits):
        logits = torch.as_tensor(logits)
        if logits.device.type != 'cpu':
            raise ValueError('Coding parameter quantization is CPU-only')
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite binary logits')
        position = (logits.double().clamp(-self.binary_logit_limit, self.binary_logit_limit) + self.binary_logit_limit) * ((self.binary_bins - 1) / (2 * self.binary_logit_limit))
        return torch.floor(position + 0.5).to(torch.int64).clamp(0, self.binary_bins - 1)
    coset_indexes = binary_indexes

    def coordinate_indexes(self, means, scales):
        (indexes, integer) = self.logistic.indexes(means, scales)
        return (indexes + self.binary_bins, integer)

    def subset(self, indexes):
        values = torch.as_tensor(indexes).reshape(-1).to(torch.int64).tolist()
        active = sorted(set(values))
        if any((value < 0 or value >= len(self.cdfs) for value in active)):
            raise ValueError('Invalid CDF index')
        mapping = {value: local for (local, value) in enumerate(active)}
        return ([mapping[value] for value in values], [self.cdfs[value] for value in active], [self.lengths[value] for value in active], [self.offsets[value] for value in active])

    def bit_cost(self, residuals, indexes):
        residuals = torch.as_tensor(residuals).cpu().to(torch.int64).reshape(-1)
        indexes = torch.as_tensor(indexes).cpu().to(torch.int64).reshape(-1)
        if residuals.shape != indexes.shape:
            raise ValueError('Symbol and CDF index counts differ')
        if bool(((indexes < 0) | (indexes >= len(self.cdfs))).any()):
            raise ValueError('Invalid CDF index')
        binary = indexes < self.binary_bins
        bits = 0.0
        if bool(binary.any()):
            (values, ids) = (residuals[binary].tolist(), indexes[binary].tolist())
            if any((value not in (0, 1) for value in values)):
                raise ValueError('Binary symbols must be zero or one')
            bits = sum((16 - math.log2(self.cdfs[index][value + 1] - self.cdfs[index][value]) for (value, index) in zip(values, ids)))
        if bool((~binary).any()):
            bits += self.logistic.bit_cost(residuals[~binary], indexes[~binary] - self.binary_bins)
        return bits

class CosetEntropyModel(nn.Module):
    """Predict all coset-coordinate distributions once from scale context.

    ``nll_bits`` returns one total per vector.  With ``zero_flag=True``, b and
    all eight coordinates are charged only for nonzero vectors.  The special
    final coordinate uses the same gap-removing wire transform as the coder.
    """
    VERSION = 1

    def __init__(self, groups, hidden=128, embedding_dim=8, zero_flag=True, learn_mean=True, cdf_config=None, binary_bins=256, binary_logit_limit=12.0, mean_correction_limit=0.25):
        super().__init__()
        if not isinstance(groups, int) or groups < 1 or hidden < 1 or (embedding_dim < 1):
            raise ValueError('Group count and network widths must be positive integers')
        if mean_correction_limit < 0:
            raise ValueError('mean_correction_limit must be nonnegative')
        (self.groups, self.hidden, self.embedding_dim) = (groups, hidden, embedding_dim)
        (self.zero_flag, self.learn_mean) = (bool(zero_flag), bool(learn_mean))
        (self.binary_bins, self.binary_logit_limit) = (int(binary_bins), float(binary_logit_limit))
        if self.binary_bins < 2 or self.binary_logit_limit <= 0:
            raise ValueError('Invalid binary CDF grid')
        self.mean_correction_limit = float(mean_correction_limit)
        if isinstance(cdf_config, dict):
            cdf_config = CDFTableConfig(**cdf_config)
        self.cdf_config = CDFTableConfig() if cdf_config is None else cdf_config
        self.cdf_config.validate()
        self.register_buffer('rotations', torch.eye(8).repeat(groups, 1, 1))
        self.register_buffer('gains', torch.ones(groups))
        self.group_embedding = nn.Embedding(groups, embedding_dim)
        self.network = nn.Sequential(nn.Linear(16 + embedding_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 40))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self._coding_tables = None

    def configuration(self):
        return dict(groups=self.groups, hidden=self.hidden, embedding_dim=self.embedding_dim, zero_flag=self.zero_flag, learn_mean=self.learn_mean, cdf_config=asdict(self.cdf_config), binary_bins=self.binary_bins, binary_logit_limit=self.binary_logit_limit, mean_correction_limit=self.mean_correction_limit)

    @torch.no_grad()
    def configure_geometry(self, rotations, gains):
        rotations = torch.as_tensor(rotations, device=self.rotations.device, dtype=self.rotations.dtype)
        gains = torch.as_tensor(gains, device=self.gains.device, dtype=self.gains.dtype)
        if rotations.shape != (self.groups, 8, 8) or gains.shape != (self.groups,):
            raise ValueError('Geometry must have shapes [G,8,8] and [G]')
        if not bool(torch.isfinite(rotations).all() & torch.isfinite(gains).all()):
            raise ValueError('Nonfinite geometry')
        if bool((gains <= 0).any()):
            raise ValueError('Geometry gains must be positive')
        identity = torch.eye(8, device=rotations.device, dtype=rotations.dtype).expand_as(rotations)
        if not torch.allclose(rotations @ rotations.transpose(-1, -2), identity, rtol=0.0001, atol=0.0001):
            raise ValueError('Geometry rotations must be orthogonal')
        self.rotations.copy_(rotations)
        self.gains.copy_(gains)
        return self

    def _validate_context(self, log_scales, group_ids):
        if log_scales.ndim != 2 or log_scales.shape[1] != 8 or len(log_scales) < 1:
            raise ValueError('log_scales must have nonempty shape [N,8]')
        if group_ids.shape != (len(log_scales),):
            raise ValueError('group_ids must have shape [N]')
        if not bool(torch.isfinite(log_scales).all()):
            raise ValueError('Nonfinite log scales')
        groups = _integers(group_ids, 'group_ids')
        if bool(((groups < 0) | (groups >= self.groups)).any()):
            raise ValueError('Invalid group identifier')
        device = self.network[0].weight.device
        if log_scales.device != device or groups.device != device:
            raise ValueError('Context and entropy model must share a device')
        return groups

    def canonical_log_scales(self, log_scales, group_ids):
        """Known diagonal canonical std from the frozen rotation/gain.

        ``rhat = point @ rotation * gain``; hence canonical variance i is
        ``sum_j rotation[i,j]**2 * sigma[j]**2 / gain**2``.  A log-sum-exp
        implementation avoids overflow without changing that formula.
        """
        groups = self._validate_context(log_scales, group_ids)
        return self._canonical_log_scales(log_scales, groups)

    def _canonical_log_scales(self, log_scales, groups):
        logs = log_scales.to(self.network[0].weight.dtype)
        weight = self.rotations[groups].square()
        return 0.5 * torch.logsumexp(2 * logs[:, None, :] + weight.log(), dim=-1) - self.gains[groups, None].log()

    def prepare(self, log_scales, group_ids, *, validate=True):
        """Predict a full front once, with no dependency on coded symbols.

        ``validate=False`` is an explicit fast path for context supplied by an
        already-validated frozen image model.  It avoids CUDA-to-host boolean
        checks; the caller guarantees finite [N,8] context, valid [N] group
        identifiers, and a shared device.  Training and public helpers retain
        validation by default.
        """
        groups = self._validate_context(log_scales, group_ids) if validate else group_ids.to(torch.int64)
        dtype = self.network[0].weight.dtype
        logs = log_scales.to(dtype)
        canonical_logs = self._canonical_log_scales(log_scales, groups)
        features = torch.cat((logs.clamp(-8.0, 7.0), canonical_logs.clamp(-8.0, 7.0), self.group_embedding(groups)), -1)
        output = self.network(features)
        reference = canonical_logs.clamp(math.log(self.cdf_config.scale_min) - 5, math.log(self.cdf_config.scale_max) + 5).exp()
        coordinate = output[:, 2:30].reshape(-1, 2, 7, 2)
        final = output[:, 30:38].reshape(-1, 2, 2, 2)
        special = output[:, 38:40]
        correction = self.mean_correction_limit if self.learn_mean else 0.0
        b = torch.arange(2, device=logs.device, dtype=dtype)
        parity = torch.arange(2, device=logs.device, dtype=dtype)
        coordinate_means = -0.5 * b[None, :, None] + correction * reference[:, None, :7] * coordinate[..., 0].tanh()
        last_means = (-0.5 * b[:, None] - parity[None, :])[None] / 2 + correction * reference[:, None, None, 7] / 2 * final[..., 0].tanh()
        coordinate_scales = (reference[:, None, :7] * LOGISTIC_STD_TO_SCALE * coordinate[..., 1].clamp(-5.0, 5.0).exp()).clamp(self.cdf_config.scale_min, self.cdf_config.scale_max)
        last_scales = (reference[:, None, None, 7] * (LOGISTIC_STD_TO_SCALE / 2) * final[..., 1].clamp(-5.0, 5.0).exp()).clamp(self.cdf_config.scale_min, self.cdf_config.scale_max)
        return dict(nonzero_logits=output[:, 0], coset_logits=output[:, 1], coordinate_means=coordinate_means, coordinate_scales=coordinate_scales, last_means=last_means, last_scales=last_scales, special_last_mean=-0.5 + correction * reference[:, 7] / 2 * special[:, 0].tanh(), special_last_scale=(reference[:, 7] * (LOGISTIC_STD_TO_SCALE / 2) * special[:, 1].clamp(-5.0, 5.0).exp()).clamp(self.cdf_config.scale_min, self.cdf_config.scale_max))

    @staticmethod
    def select_coordinates(prepared, coset):
        coset = _integers(coset, 'coset')
        if coset.ndim != 1 or bool(((coset < 0) | (coset > 1)).any()):
            raise ValueError('coset must have shape [N] with zero/one values')
        return CosetEntropyModel._select_coordinates(prepared, coset)

    @staticmethod
    def _select_coordinates(prepared, coset):
        rows = torch.arange(len(coset), device=coset.device)
        return (prepared['coordinate_means'][rows, coset], prepared['coordinate_scales'][rows, coset])

    def select_last(self, prepared, coset, first):
        (coset, first) = (_integers(coset, 'coset'), _integers(first, 'first'))
        if coset.ndim != 1 or first.shape != (len(coset), 7) or bool(((coset < 0) | (coset > 1)).any()):
            raise ValueError('coset/first must have shapes [N], [N,7] with zero/one cosets')
        return self._select_last(prepared, coset, first)

    def _select_last(self, prepared, coset, first):
        rows = torch.arange(len(coset), device=coset.device)
        parity = first.sum(-1).remainder(2)
        (means, scales) = (prepared['last_means'][rows, coset, parity], prepared['last_scales'][rows, coset, parity])
        if self.zero_flag:
            special = _special_last(coset, first)
            means = torch.where(special, prepared['special_last_mean'], means)
            scales = torch.where(special, prepared['special_last_scale'], scales)
        return (means, scales)

    def nll_bits(self, symbols, log_scales, group_ids, *, validate=True):
        """Per-vector wire-symbol NLL, optionally trusting a preflighted cache.

        ``validate=False`` skips symbol/context range and shape checks, allowing
        cached int64 ``CosetSymbols`` on GPU to train without host boolean
        synchronizations.  Validate the entire immutable cache once before
        selecting this path; both paths use identical probability arithmetic.
        """
        if validate:
            (coset, first, last) = _symbols(symbols)
            if len(coset) != len(log_scales) or coset.device != log_scales.device:
                raise ValueError('Symbols and context must have matching batch/device')
        else:
            (coset, first, last) = (value.to(torch.int64) for value in symbols)
        prepared = self.prepare(log_scales, group_ids, validate=validate)
        (means, scales) = self._select_coordinates(prepared, coset)
        bits = F.binary_cross_entropy_with_logits(prepared['coset_logits'], coset.to(means.dtype), reduction='none') / LOG2
        bits = bits - _positive_logistic_log_mass(first, means, scales).sum(-1) / LOG2
        (means, scales) = self._select_last(prepared, coset, first)
        if self.zero_flag:
            active = (coset != 0) | first.ne(0).any(-1) | (last != 0)
            coded_last = last - (_special_last(coset, first) & (last > 0)).to(last.dtype)
        else:
            (coded_last, active) = (last, torch.ones_like(coset, dtype=torch.bool))
        bits = bits - _positive_logistic_log_mass(coded_last, means, scales) / LOG2
        bits = torch.where(active, bits, torch.zeros_like(bits))
        if self.zero_flag:
            bits = bits + F.binary_cross_entropy_with_logits(prepared['nonzero_logits'], active.to(bits.dtype), reduction='none') / LOG2
        return bits
    forward = nll_bits

    def build_coding_tables(self):
        if self._coding_tables is None:
            self._coding_tables = CosetCodingTables(self.cdf_config, self.binary_bins, self.binary_logit_limit)
        return self._coding_tables

    def cdf_indexes(self, means, scales):
        return self.build_coding_tables().coordinate_indexes(means, scales)

    def binary_cdf_indexes(self, logits):
        return self.build_coding_tables().binary_indexes(logits)

    @torch.no_grad()
    def ideal_bits(self, symbols, log_scales, group_ids):
        return float(self.nll_bits(symbols, log_scales, group_ids).double().sum())

    @torch.no_grad()
    def _coding_batches(self, symbols, prepared):
        """Synthetic codec helper; the image codec should share its rANS state."""
        (coset, first, last) = _symbols(symbols)
        if coset.device.type != 'cpu':
            raise ValueError('rANS symbols must be on CPU')
        tables = self.build_coding_tables()
        if self.zero_flag:
            active = nonzero_mask(symbols)
            yield (active.to(torch.int64), tables.binary_indexes(prepared['nonzero_logits']))
        else:
            active = torch.ones_like(coset, dtype=torch.bool)
        if not bool(active.any()):
            return
        prepared = {key: value[active] for (key, value) in prepared.items()}
        (coset, first, last) = (coset[active], first[active], last[active])
        yield (coset, tables.binary_indexes(prepared['coset_logits']))
        (means, scales) = self.select_coordinates(prepared, coset)
        (indexes, integer) = tables.coordinate_indexes(means, scales)
        yield ((first - integer).reshape(-1), indexes.reshape(-1))
        (means, scales) = self.select_last(prepared, coset, first)
        (indexes, integer) = tables.coordinate_indexes(means, scales)
        value = wire_last(coset, first, last) if self.zero_flag else last
        yield (value - integer, indexes)

    @torch.no_grad()
    def estimated_bits(self, symbols, log_scales, group_ids):
        prepared = {key: value.cpu() for (key, value) in self.prepare(log_scales, group_ids).items()}
        tables = self.build_coding_tables()
        return sum((tables.bit_cost(values, indexes) for (values, indexes) in self._coding_batches(symbols, prepared)))

    @torch.no_grad()
    def encode_symbols(self, symbols, log_scales, group_ids):
        from compressai.ans import BufferedRansEncoder
        prepared = {key: value.cpu() for (key, value) in self.prepare(log_scales, group_ids).items()}
        (encoder, tables) = (BufferedRansEncoder(), self.build_coding_tables())
        for (values, indexes) in self._coding_batches(symbols, prepared):
            if bool((values.abs() > SAFE_RESIDUAL_LIMIT).any()):
                raise ValueError('Symbol outside safe bypass range')
            (local, cdfs, lengths, offsets) = tables.subset(indexes)
            encoder.encode_with_indexes(values.tolist(), local, cdfs, lengths, offsets)
        return encoder.flush()

    @torch.no_grad()
    def decode_symbols(self, stream, log_scales, group_ids):
        from compressai.ans import RansDecoder
        if not isinstance(stream, (bytes, bytearray)) or len(stream) < 8:
            raise ValueError('A rANS stream must contain at least its 64-bit state')
        prepared = {key: value.cpu() for (key, value) in self.prepare(log_scales, group_ids).items()}
        (decoder, tables) = (RansDecoder(), self.build_coding_tables())
        decoder.set_stream(stream)

        def decode(indexes):
            (local, cdfs, lengths, offsets) = tables.subset(indexes)
            return torch.tensor(decoder.decode_stream(local, cdfs, lengths, offsets), dtype=torch.int64)
        count = len(log_scales)
        result = CosetSymbols(torch.zeros(count, dtype=torch.int64), torch.zeros((count, 7), dtype=torch.int64), torch.zeros(count, dtype=torch.int64))
        active = decode(tables.binary_indexes(prepared['nonzero_logits'])).bool() if self.zero_flag else torch.ones(count, dtype=torch.bool)
        if not bool(active.any()):
            return result
        prepared = {key: value[active] for (key, value) in prepared.items()}
        coset = decode(tables.binary_indexes(prepared['coset_logits']))
        (means, scales) = self.select_coordinates(prepared, coset)
        (indexes, integer) = tables.coordinate_indexes(means, scales)
        first = decode(indexes.reshape(-1)).reshape(-1, 7) + integer
        (means, scales) = self.select_last(prepared, coset, first)
        (indexes, integer) = tables.coordinate_indexes(means, scales)
        last = decode(indexes) + integer
        if self.zero_flag:
            last = restore_last(coset, first, last)
        (result.coset[active], result.first[active], result.last[active]) = (coset, first, last)
        return result
