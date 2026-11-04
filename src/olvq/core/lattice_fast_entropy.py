# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from olvq.core.e8_coset_entropy import CosetCodingTables, LOGISTIC_STD_TO_SCALE, _integers, _positive_logistic_log_mass
from olvq.core.e8_discrete_entropy import CDFTableConfig, LOG2, SAFE_RESIDUAL_LIMIT
DIMENSION = 8

def validate_symbols(symbols):
    """Return exact int64 basis indexes, rejecting rounded or nonfinite input."""
    symbols = _integers(symbols, 'symbols')
    if symbols.ndim != 2 or symbols.shape[1] != DIMENSION or len(symbols) < 1:
        raise ValueError('symbols must have nonempty shape [N,8]')
    return symbols

def wire_last(first, last):
    """Map the last coordinate of a nonzero vector to a full integer domain."""
    (first, last) = (_integers(first, 'first'), _integers(last, 'last'))
    if first.ndim != 2 or first.shape[1] != 7 or last.shape != (len(first),):
        raise ValueError('first/last must have shapes [N,7] and [N]')
    special = first.eq(0).all(-1)
    if bool((special & last.eq(0)).any()):
        raise ValueError('The nonzero branch cannot contain the zero vector')
    return last - (special & last.gt(0)).to(last.dtype)

def restore_last(first, wire):
    """Invert :func:`wire_last`; the special branch can never restore zero."""
    (first, wire) = (_integers(first, 'first'), _integers(wire, 'wire'))
    if first.ndim != 2 or first.shape[1] != 7 or wire.shape != (len(first),):
        raise ValueError('first/wire must have shapes [N,7] and [N]')
    return wire + (first.eq(0).all(-1) & wire.ge(0)).to(wire.dtype)

class IntegerLatticeEntropyModel(nn.Module):
    """A context-only entropy head shared by the I8-based lattice routes."""
    VERSION = 1

    def __init__(self, groups, hidden=128, embedding_dim=8, zero_flag=True, learn_mean=True, cdf_config=None, binary_bins=256, binary_logit_limit=12.0, mean_correction_limit=0.25):
        super().__init__()
        if any((not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (groups, hidden, embedding_dim))):
            raise ValueError('Group count and network widths must be positive integers')
        if not math.isfinite(mean_correction_limit) or mean_correction_limit < 0:
            raise ValueError('mean_correction_limit must be finite and nonnegative')
        if not isinstance(binary_bins, int) or isinstance(binary_bins, bool) or binary_bins < 2 or (not math.isfinite(binary_logit_limit)) or (binary_logit_limit <= 0):
            raise ValueError('Invalid binary CDF grid')
        (self.groups, self.hidden, self.embedding_dim) = (groups, hidden, embedding_dim)
        (self.zero_flag, self.learn_mean) = (bool(zero_flag), bool(learn_mean))
        (self.binary_bins, self.binary_logit_limit) = (binary_bins, float(binary_logit_limit))
        self.mean_correction_limit = float(mean_correction_limit)
        if isinstance(cdf_config, dict):
            cdf_config = CDFTableConfig(**cdf_config)
        self.cdf_config = CDFTableConfig() if cdf_config is None else cdf_config
        self.cdf_config.validate()
        self.register_buffer('bases', torch.eye(8).repeat(groups, 1, 1))
        self.register_buffer('inverse_bases', torch.eye(8).repeat(groups, 1, 1))
        self.group_embedding = nn.Embedding(groups, embedding_dim)
        self.network = nn.Sequential(nn.Linear(16 + embedding_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 19))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self._coding_tables = None

    def configuration(self):
        return dict(groups=self.groups, hidden=self.hidden, embedding_dim=self.embedding_dim, zero_flag=self.zero_flag, learn_mean=self.learn_mean, cdf_config=asdict(self.cdf_config), binary_bins=self.binary_bins, binary_logit_limit=self.binary_logit_limit, mean_correction_limit=self.mean_correction_limit)

    @torch.no_grad()
    def configure_geometry(self, bases):
        """Freeze a nonsingular row basis per group; no quantizer is called."""
        bases = torch.as_tensor(bases, device=self.bases.device, dtype=self.bases.dtype)
        if bases.shape != (self.groups, 8, 8) or not bool(torch.isfinite(bases).all()):
            raise ValueError('Geometry bases must be finite with shape [G,8,8]')
        try:
            inverse = torch.linalg.inv(bases.double()).to(bases.dtype)
        except RuntimeError as error:
            raise ValueError('Geometry bases must be nonsingular') from error
        if not bool(torch.isfinite(inverse).all()):
            raise ValueError('Geometry inverse must be finite')
        self.bases.copy_(bases)
        self.inverse_bases.copy_(inverse)
        return self
    validate_symbols = staticmethod(validate_symbols)

    def _validate_context(self, log_scales, group_ids):
        if log_scales.ndim != 2 or log_scales.shape[1] != DIMENSION or len(log_scales) < 1 or (not log_scales.is_floating_point()):
            raise ValueError('log_scales must have nonempty floating shape [N,8]')
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

    def _coordinate_log_scales(self, log_scales, groups):
        logs = log_scales.to(self.network[0].weight.dtype)
        weights = self.inverse_bases[groups].transpose(-1, -2).square()
        return 0.5 * torch.logsumexp(2 * logs[:, None, :] + weights.log(), dim=-1)

    def coordinate_log_scales(self, log_scales, group_ids):
        """Diagonal index std implied by decoder-known scales and row basis."""
        return self._coordinate_log_scales(log_scales, self._validate_context(log_scales, group_ids))

    def prepare(self, log_scales, group_ids, *, validate=True):
        """Prepare every distribution in one network call, with no symbols.

        ``validate=False`` is only for a previously checked immutable cache or
        frozen-model context. It avoids CUDA boolean synchronization while
        using exactly the same probability arithmetic.
        """
        groups = self._validate_context(log_scales, group_ids) if validate else group_ids.to(torch.int64)
        logs = log_scales.to(self.network[0].weight.dtype)
        coordinate_logs = self._coordinate_log_scales(log_scales, groups)
        output = self.network(torch.cat((logs.clamp(-8.0, 7.0), coordinate_logs.clamp(-8.0, 7.0), self.group_embedding(groups)), dim=-1))
        reference = coordinate_logs.clamp(math.log(self.cdf_config.scale_min) - 5, math.log(self.cdf_config.scale_max) + 5).exp()
        parameters = output[:, 1:17].reshape(-1, 8, 2)
        correction = self.mean_correction_limit if self.learn_mean else 0.0
        means = correction * reference * parameters[..., 0].tanh()
        scales = (reference * LOGISTIC_STD_TO_SCALE * parameters[..., 1].clamp(-5.0, 5.0).exp()).clamp(self.cdf_config.scale_min, self.cdf_config.scale_max)
        special_mean = -0.5 + correction * reference[:, 7] * output[:, 17].tanh()
        special_scale = (reference[:, 7] * LOGISTIC_STD_TO_SCALE * output[:, 18].clamp(-5.0, 5.0).exp()).clamp(self.cdf_config.scale_min, self.cdf_config.scale_max)
        return dict(nonzero_logits=output[:, 0], means=means, scales=scales, special_last_mean=special_mean, special_last_scale=special_scale)

    def select_last(self, prepared, first):
        (means, scales) = (prepared['means'][:, 7], prepared['scales'][:, 7])
        if self.zero_flag:
            special = first.eq(0).all(-1)
            means = torch.where(special, prepared['special_last_mean'], means)
            scales = torch.where(special, prepared['special_last_scale'], scales)
        return (means, scales)

    def nll_bits(self, symbols, log_scales, group_ids, *, validate=True):
        """Return normalized continuous-logistic discrete NLL per vector."""
        symbols = validate_symbols(symbols) if validate else symbols.to(torch.int64)
        if validate and (symbols.shape != log_scales.shape or symbols.device != log_scales.device):
            raise ValueError('Symbols and context must have matching shape/device')
        prepared = self.prepare(log_scales, group_ids, validate=validate)
        (first, last) = (symbols[:, :7], symbols[:, 7])
        bits = -_positive_logistic_log_mass(first, prepared['means'][:, :7], prepared['scales'][:, :7]).sum(-1) / LOG2
        (means, scales) = self.select_last(prepared, first)
        active = symbols.ne(0).any(-1)
        value = last - (first.eq(0).all(-1) & last.gt(0)).to(last.dtype) if self.zero_flag else last
        bits = bits - _positive_logistic_log_mass(value, means, scales) / LOG2
        if self.zero_flag:
            bits = torch.where(active, bits, torch.zeros_like(bits))
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
        """CPU batches for a shared image rANS state, in decoder order."""
        symbols = validate_symbols(symbols)
        if symbols.device.type != 'cpu':
            raise ValueError('rANS symbols must be on CPU')
        if len(symbols) != len(prepared['means']):
            raise ValueError('Symbol and prepared-context counts differ')
        tables = self.build_coding_tables()
        if not self.zero_flag:
            (indexes, integer) = tables.coordinate_indexes(prepared['means'], prepared['scales'])
            yield ((symbols - integer).reshape(-1), indexes.reshape(-1))
            return
        active = symbols.ne(0).any(-1)
        yield (active.to(torch.int64), tables.binary_indexes(prepared['nonzero_logits']))
        if not bool(active.any()):
            return
        prepared = {key: value[active] for (key, value) in prepared.items()}
        (first, last) = (symbols[active, :7], symbols[active, 7])
        (indexes, integer) = tables.coordinate_indexes(prepared['means'][:, :7], prepared['scales'][:, :7])
        yield ((first - integer).reshape(-1), indexes.reshape(-1))
        (means, scales) = self.select_last(prepared, first)
        (indexes, integer) = tables.coordinate_indexes(means, scales)
        yield (wire_last(first, last) - integer, indexes)

    @torch.no_grad()
    def estimated_bits(self, symbols, log_scales, group_ids):
        prepared = {key: value.cpu() for (key, value) in self.prepare(log_scales, group_ids).items()}
        tables = self.build_coding_tables()
        return sum((tables.bit_cost(values, indexes) for (values, indexes) in self._coding_batches(validate_symbols(symbols).cpu(), prepared)))

    @torch.no_grad()
    def encode_symbols(self, symbols, log_scales, group_ids):
        from compressai.ans import BufferedRansEncoder
        prepared = {key: value.cpu() for (key, value) in self.prepare(log_scales, group_ids).items()}
        (encoder, tables) = (BufferedRansEncoder(), self.build_coding_tables())
        for (values, indexes) in self._coding_batches(validate_symbols(symbols).cpu(), prepared):
            if bool((values.abs() > SAFE_RESIDUAL_LIMIT).any()):
                raise ValueError('Symbol outside safe bypass range')
            (local, cdfs, lengths, offsets) = tables.subset(indexes)
            encoder.encode_with_indexes(values.tolist(), local, cdfs, lengths, offsets)
        return encoder.flush()

    @torch.no_grad()
    def decode_front(self, decoder, prepared):
        """Decode one already-prepared CPU front from a shared rANS decoder."""
        tables = self.build_coding_tables()

        def decode(indexes):
            (local, cdfs, lengths, offsets) = tables.subset(indexes)
            values = torch.tensor(decoder.decode_stream(local, cdfs, lengths, offsets), dtype=torch.int64)
            if bool((values.abs() > SAFE_RESIDUAL_LIMIT).any()):
                raise ValueError('Decoded symbol outside safe bypass range')
            return values
        count = len(prepared['means'])
        if not self.zero_flag:
            (indexes, integer) = tables.coordinate_indexes(prepared['means'], prepared['scales'])
            return decode(indexes.reshape(-1)).reshape(count, 8) + integer
        flags = decode(tables.binary_indexes(prepared['nonzero_logits']))
        if bool(((flags < 0) | (flags > 1)).any()):
            raise ValueError('Decoded nonzero flag is not binary')
        (active, result) = (flags.bool(), torch.zeros(count, 8, dtype=torch.int64))
        if not bool(active.any()):
            return result
        prepared = {key: value[active] for (key, value) in prepared.items()}
        (indexes, integer) = tables.coordinate_indexes(prepared['means'][:, :7], prepared['scales'][:, :7])
        first = decode(indexes.reshape(-1)).reshape(-1, 7) + integer
        (means, scales) = self.select_last(prepared, first)
        (indexes, integer) = tables.coordinate_indexes(means, scales)
        last = restore_last(first, decode(indexes) + integer)
        result[active] = torch.cat((first, last[:, None]), dim=-1)
        return result

    @torch.no_grad()
    def decode_symbols(self, stream, log_scales, group_ids):
        from compressai.ans import RansDecoder
        if not isinstance(stream, (bytes, bytearray)) or len(stream) < 8:
            raise ValueError('A rANS stream must contain at least its 64-bit state')
        prepared = {key: value.cpu() for (key, value) in self.prepare(log_scales, group_ids).items()}
        decoder = RansDecoder()
        decoder.set_stream(stream)
        return self.decode_front(decoder, prepared)
