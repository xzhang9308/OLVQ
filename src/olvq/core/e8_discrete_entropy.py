# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
LOG2 = math.log(2.0)
DIMENSION = 8
SAFE_RESIDUAL_LIMIT = 1 << 24

def logistic_log_mass(symbols, means, scales):
    """Stable log P(symbol-.5 <= Logistic(mean,scale) < symbol+.5)."""
    if not torch.is_floating_point(means) or not torch.is_floating_point(scales):
        raise TypeError('Logistic parameters must be floating point tensors')
    if bool((scales <= 0).any()):
        raise ValueError('Logistic scales must be strictly positive')
    centered = symbols.to(means.dtype) - means
    upper = (centered + 0.5) / scales
    lower = (centered - 0.5) / scales
    return F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(-1.0 / scales))

@dataclass(frozen=True)
class CDFTableConfig:
    precision: int = 16
    scale_bins: int = 96
    mean_fraction_bins: int = 16
    scale_min: float = 0.08
    scale_max: float = 128.0
    support_scale: float = 10.0
    support_min: int = 4
    support_max: int = 1024

    def validate(self):
        if self.precision != 16:
            raise ValueError('The CompressAI rANS backend requires precision=16')
        if self.scale_bins < 2 or self.mean_fraction_bins < 1:
            raise ValueError('CDF table grid sizes are invalid')
        if not 0 < self.scale_min < self.scale_max:
            raise ValueError('CDF scale bounds are invalid')
        if not 1 <= self.support_min <= self.support_max:
            raise ValueError('CDF support bounds are invalid')
        if self.support_scale <= 0:
            raise ValueError('support_scale must be positive')
        if 2 * self.support_max + 2 >= 1 << self.precision:
            raise ValueError('CDF support leaves no positive frequency budget')

def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    term = math.exp(value)
    return term / (1.0 + term)

def _positive_integer_cdf(probabilities, precision):
    """Normalize and quantize a PMF with a positive frequency for every bin."""
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 1 or not np.isfinite(probability).all() or (probability < 0).any() or (probability.sum() <= 0):
        raise ValueError('Invalid discrete PMF')
    total = 1 << precision
    if len(probability) >= total:
        raise ValueError('Too many symbols for the CDF precision')
    probability = probability / probability.sum()
    scaled = probability * (total - len(probability))
    floored = np.floor(scaled).astype(np.int64)
    frequency = floored + 1
    missing = total - int(frequency.sum())
    if missing:
        order = np.argsort(-(scaled - floored), kind='stable')
        frequency[order[:missing]] += 1
    cdf = np.concatenate(([0], np.cumsum(frequency))).astype(np.int64)
    if cdf[-1] != total or not (np.diff(cdf) > 0).all():
        raise RuntimeError('Integer CDF normalization failed')
    return cdf.tolist()

class DiscreteCDFTables:
    """Shared finite CDF bank plus CompressAI-compatible tail bypass.

    A table represents integer residuals -R through R, followed by one escape
    symbol.  All remaining integers use the backend's native signed bypass.
    Means are split into an integer shift and a quantized fractional part.
    """

    def __init__(self, config: Optional[CDFTableConfig]=None):
        self.config = CDFTableConfig() if config is None else config
        self.config.validate()
        c = self.config
        (log_min, log_max) = (math.log(c.scale_min), math.log(c.scale_max))
        self.scales = [math.exp(log_min + i * (log_max - log_min) / (c.scale_bins - 1)) for i in range(c.scale_bins)]
        (self.cdfs, self.lengths, self.offsets) = ([], [], [])
        for scale in self.scales:
            radius = min(c.support_max, max(c.support_min, int(math.ceil(c.support_scale * scale + 1))))
            for fraction_index in range(c.mean_fraction_bins):
                mean = fraction_index / c.mean_fraction_bins
                probabilities = []
                for symbol in range(-radius, radius + 1):
                    upper = (symbol + 0.5 - mean) / scale
                    lower = (symbol - 0.5 - mean) / scale
                    mass = _sigmoid(upper) * _sigmoid(-lower) * -math.expm1(-1.0 / scale)
                    probabilities.append(mass)
                tail = _sigmoid((-radius - 0.5 - mean) / scale) + _sigmoid(-(radius + 0.5 - mean) / scale)
                probabilities.append(tail)
                cdf = _positive_integer_cdf(probabilities, c.precision)
                self.cdfs.append(cdf)
                self.lengths.append(len(cdf))
                self.offsets.append(-radius)
        longest = max(self.lengths)
        self._frequency = np.ones((len(self.cdfs), longest - 1), dtype=np.int32)
        for (index, cdf) in enumerate(self.cdfs):
            self._frequency[index, :len(cdf) - 1] = np.diff(cdf)
        self._offset = np.asarray(self.offsets, dtype=np.int64)
        self._maximum = np.asarray(self.lengths, dtype=np.int64) - 2

    def indexes(self, means, scales):
        if means.device.type != 'cpu' or scales.device.type != 'cpu':
            raise ValueError('Coding parameter quantization is CPU-only')
        if means.shape != scales.shape:
            raise ValueError('Means and scales must have the same shape')
        if not bool(torch.isfinite(means).all() & torch.isfinite(scales).all()):
            raise ValueError('Nonfinite coding parameters')
        if bool((scales <= 0).any()):
            raise ValueError('Nonpositive coding scale')
        if bool((means.abs() > SAFE_RESIDUAL_LIMIT).any()):
            raise ValueError('Mean outside safe integer offset range')
        c = self.config
        (means, scales) = (means.double(), scales.double())
        integer = torch.floor(means).to(torch.int64)
        fraction = torch.floor((means - integer) * c.mean_fraction_bins + 0.5).to(torch.int64)
        integer = integer + torch.div(fraction, c.mean_fraction_bins, rounding_mode='floor')
        fraction = fraction.remainder(c.mean_fraction_bins)
        position = (scales.clamp(c.scale_min, c.scale_max).log() - math.log(c.scale_min)) * ((c.scale_bins - 1) / math.log(c.scale_max / c.scale_min))
        scale_index = torch.floor(position + 0.5).to(torch.int64).clamp(0, c.scale_bins - 1)
        return (scale_index * c.mean_fraction_bins + fraction, integer)

    def subset(self, indexes):
        """Only marshal tables actually used by one rANS call across pybind.

        The selected global table IDs are decoder-known; both ends derive the
        same local remapping, so this optimization needs no side information.
        """
        values = torch.as_tensor(indexes).reshape(-1).to(torch.int64).tolist()
        active = sorted(set(values))
        if any((value < 0 or value >= len(self.cdfs) for value in active)):
            raise ValueError('Invalid CDF index')
        remapping = {value: local for (local, value) in enumerate(active)}
        return ([remapping[value] for value in values], [self.cdfs[value] for value in active], [self.lengths[value] for value in active], [self.offsets[value] for value in active])

    def bit_cost(self, residuals, indexes):
        """Ideal bits for the actual integer CDF, including native bypass bits."""
        residual = np.asarray(torch.as_tensor(residuals).cpu(), dtype=np.int64).reshape(-1)
        index = np.asarray(torch.as_tensor(indexes).cpu(), dtype=np.int64).reshape(-1)
        if len(residual) != len(index):
            raise ValueError('Symbol and CDF index counts differ')
        if (index < 0).any() or (index >= len(self.cdfs)).any() or (np.abs(residual) > SAFE_RESIDUAL_LIMIT).any():
            raise ValueError('Symbol or CDF index outside safe coding range')
        value = residual - self._offset[index]
        maximum = self._maximum[index]
        (lower, upper) = (value < 0, value >= maximum)
        escape = lower | upper
        coded = np.where(escape, maximum, value)
        frequency = self._frequency[index, coded]
        bits = float((self.config.precision - np.log2(frequency.astype(np.float64))).sum())
        if escape.any():
            raw = np.where(lower, -2 * value - 1, 2 * (value - maximum))[escape]
            for raw_value in raw.tolist():
                n_bypass = (int(raw_value).bit_length() + 3) // 4
                bits += 4 * (n_bypass + n_bypass // 15 + 1)
        return bits

class ConditionedEntropyModel(nn.Module):
    """Causal coordinate entropy model; all non-entropy quantities are inputs."""
    VERSION = 1

    def __init__(self, groups, hidden=64, embedding_dim=4, cdf_config=None):
        super().__init__()
        if not isinstance(groups, int) or groups < 1:
            raise ValueError('groups must be a positive integer')
        if hidden < 1 or embedding_dim < 1:
            raise ValueError('Network widths must be positive')
        (self.groups, self.hidden, self.embedding_dim) = (groups, hidden, embedding_dim)
        if isinstance(cdf_config, dict):
            cdf_config = CDFTableConfig(**cdf_config)
        self.cdf_config = CDFTableConfig() if cdf_config is None else cdf_config
        self.cdf_config.validate()
        self.group_embedding = nn.Embedding(groups, embedding_dim)
        self.coordinate_embedding = nn.Embedding(DIMENSION, embedding_dim)
        self.network = nn.Sequential(nn.Linear(2 * DIMENSION + 2 * embedding_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.prefix_mean = nn.Parameter(torch.zeros(groups, DIMENSION, DIMENSION))
        self.register_buffer('causal_mask', torch.tril(torch.ones(DIMENSION, DIMENSION), -1))
        self._coding_tables = None

    def configuration(self):
        return dict(groups=self.groups, hidden=self.hidden, embedding_dim=self.embedding_dim, cdf_config=asdict(self.cdf_config))

    def _validate(self, symbols, log_scales, group_ids):
        if log_scales.ndim != 2 or log_scales.shape[1] != DIMENSION:
            raise ValueError('log_scales must have shape [N,8]')
        if len(log_scales) < 1:
            raise ValueError('An entropy-model batch cannot be empty')
        if group_ids.ndim != 1 or len(group_ids) != len(log_scales):
            raise ValueError('group_ids must have shape [N]')
        if bool(((group_ids < 0) | (group_ids >= self.groups)).any()):
            raise ValueError('Invalid group identifier')
        if not bool(torch.isfinite(log_scales).all()):
            raise ValueError('Nonfinite log scales')
        if symbols is not None and symbols.shape != log_scales.shape:
            raise ValueError('symbols must have shape [N,8]')

    @staticmethod
    def _reference(log_scales):
        logs = log_scales.clamp(-8.0, 7.0)
        reference = torch.exp(0.5 * (torch.logsumexp(2 * logs, -1) - math.log(DIMENSION)))
        return (logs, reference.clamp_min(0.05))

    def forward(self, symbols, log_scales, group_ids):
        self._validate(symbols, log_scales, group_ids)
        dtype = self.network[0].weight.dtype
        (logs, reference) = self._reference(log_scales.to(dtype))
        symbols = symbols.to(dtype)
        mask = self.causal_mask.to(dtype)
        prefix = symbols[:, None, :] * mask[None]
        normalized = torch.asinh(prefix / reference[:, None, None])
        group_embedding = self.group_embedding(group_ids.long())[:, None, :].expand(-1, 8, -1)
        coordinate_embedding = self.coordinate_embedding.weight[None].expand(len(symbols), -1, -1)
        features = torch.cat((logs[:, None].expand(-1, 8, -1), normalized, group_embedding, coordinate_embedding), -1)
        output = self.network(features)
        direct = (prefix * self.prefix_mean[group_ids.long()]).sum(-1)
        means = direct + reference[:, None] * output[..., 0]
        scales = (reference[:, None] * output[..., 1].clamp(-4.0, 4.0).exp()).clamp(self.cdf_config.scale_min, self.cdf_config.scale_max)
        return (means, scales)

    def predict_coordinate(self, prefix, log_scales, group_ids, k):
        self._validate(None, log_scales, group_ids)
        if not isinstance(k, int) or not 0 <= k < DIMENSION:
            raise ValueError('Coordinate must be in [0,8)')
        if prefix.ndim != 2 or len(prefix) != len(log_scales) or prefix.shape[1] not in (k, 8):
            raise ValueError('prefix must have shape [N,k] or [N,8]')
        dtype = self.network[0].weight.dtype
        (logs, reference) = self._reference(log_scales.to(dtype))
        known = torch.zeros_like(logs)
        if k:
            known[:, :k] = prefix[:, :k].to(dtype)
        normalized = torch.asinh(known / reference[:, None])
        coordinate = self.coordinate_embedding.weight[k].expand(len(logs), -1)
        features = torch.cat((logs, normalized, self.group_embedding(group_ids.long()), coordinate), -1)
        output = self.network(features)
        direct = (known * self.prefix_mean[group_ids.long(), k]).sum(-1)
        means = direct + reference * output[:, 0]
        scales = (reference * output[:, 1].clamp(-4.0, 4.0).exp()).clamp(self.cdf_config.scale_min, self.cdf_config.scale_max)
        return (means, scales)

    def predict(self, prefix, log_scales, group_ids, coordinate):
        return self.predict_coordinate(prefix, log_scales, group_ids, coordinate)

    def nll_bits(self, symbols, log_scales, group_ids):
        (means, scales) = self(symbols, log_scales, group_ids)
        return -logistic_log_mass(symbols, means, scales) / LOG2

    def _require_cpu(self):
        if next(self.parameters()).device.type != 'cpu':
            raise ValueError('The prototype entropy coder requires a CPU model')

    def build_coding_tables(self):
        if self._coding_tables is None:
            self._coding_tables = DiscreteCDFTables(self.cdf_config)
        return self._coding_tables

    def cdf_indexes(self, means, scales):
        return self.build_coding_tables().indexes(means, scales)

    @staticmethod
    def continuous_bits_from_parameters(symbols, means, scales):
        with torch.no_grad():
            return float((-logistic_log_mass(symbols, means, scales).double() / LOG2).sum())

    def estimated_bits_from_parameters(self, symbols, means, scales):
        with torch.no_grad():
            (indexes, integer_means) = self.cdf_indexes(means, scales)
            return self.build_coding_tables().bit_cost(symbols.to(torch.int64) - integer_means, indexes)

    @torch.no_grad()
    def ideal_bits(self, symbols, log_scales, group_ids):
        return float(self.nll_bits(symbols, log_scales, group_ids).double().sum())

    @torch.no_grad()
    def estimated_bits(self, symbols, log_scales, group_ids):
        self._require_cpu()
        bits = 0.0
        for k in range(DIMENSION):
            (means, scales) = self.predict_coordinate(symbols[:, :k], log_scales, group_ids, k)
            bits += self.estimated_bits_from_parameters(symbols[:, k], means, scales)
        return bits

    @torch.no_grad()
    def encode_symbols(self, symbols, log_scales, group_ids):
        """Standalone stream helper; a full image should share a single coder."""
        self._require_cpu()
        self._validate(symbols, log_scales, group_ids)
        if symbols.device.type != 'cpu' or log_scales.device.type != 'cpu':
            raise ValueError('Coding symbols and features must be on CPU')
        if torch.is_floating_point(symbols) and (not bool((symbols == symbols.round()).all())):
            raise ValueError('Coding symbols must be integers')
        from compressai.ans import BufferedRansEncoder
        encoder = BufferedRansEncoder()
        tables = self.build_coding_tables()
        for k in range(DIMENSION):
            (means, scales) = self.predict_coordinate(symbols[:, :k], log_scales, group_ids, k)
            (indexes, integer_means) = self.cdf_indexes(means, scales)
            residuals = symbols[:, k].to(torch.int64) - integer_means
            if bool((residuals.abs() > SAFE_RESIDUAL_LIMIT).any()):
                raise ValueError('Symbol outside safe bypass range')
            (local, cdfs, lengths, offsets) = tables.subset(indexes)
            encoder.encode_with_indexes(residuals.tolist(), local, cdfs, lengths, offsets)
        return encoder.flush()

    @torch.no_grad()
    def decode_symbols(self, stream, log_scales, group_ids):
        self._require_cpu()
        self._validate(None, log_scales, group_ids)
        if log_scales.device.type != 'cpu':
            raise ValueError('Coding features must be on CPU')
        if not isinstance(stream, (bytes, bytearray)) or len(stream) < 8:
            raise ValueError('A rANS stream must contain at least its 64-bit state')
        from compressai.ans import RansDecoder
        decoder = RansDecoder()
        decoder.set_stream(stream)
        tables = self.build_coding_tables()
        symbols = torch.zeros((len(log_scales), DIMENSION), dtype=torch.int64)
        for k in range(DIMENSION):
            (means, scales) = self.predict_coordinate(symbols[:, :k], log_scales, group_ids, k)
            (indexes, integer_means) = self.cdf_indexes(means, scales)
            (local, cdfs, lengths, offsets) = tables.subset(indexes)
            values = decoder.decode_stream(local, cdfs, lengths, offsets)
            symbols[:, k] = torch.tensor(values, dtype=torch.int64) + integer_means
        return symbols
