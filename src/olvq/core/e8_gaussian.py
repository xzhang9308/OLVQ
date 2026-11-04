from __future__ import annotations
import math
import torch
import torch.nn as nn
from olvq.core import quantizers
from compressai.models.google import FactorizedPrior, JointAutoregressiveHierarchicalPriors, MeanScaleHyperprior, ScaleHyperprior
from compressai.models.waseda import Cheng2020Anchor, Cheng2020Attention
from olvq.core.e8_layout import ADAPTIVE_GROUPING_MODES, MbtOLVQBase

class FixedE8GaussianMixin:
    """Reuse the validated MBT E8 cell integration for other backbones."""
    _flatten = MbtOLVQBase._flatten
    _unflatten = MbtOLVQBase._unflatten
    _adaptive_grouping_index = MbtOLVQBase._adaptive_grouping_index
    _quantization_diagnostics = MbtOLVQBase._quantization_diagnostics

    def _setup_fixed_e8(self, channels: int, channel_permutation=None, adaptive_grouping: str='none', voronoi_mc_samples: int=2048, voronoi_mc_samples_train: int=2048, voronoi_likelihood_chunk: int=4096, voronoi_eval_seed: int=104729, ar_context_iterations: int=2, e8_scale: float=1.0, e8_learn_scale: bool=False, e8_per_group_scale: bool=False, companding_law: str='none', companding_strength_init: float=0.01, companding_spatial_slope_init: float=0.0, companding_loading: float=4.0) -> None:
        self.channels = int(channels)
        self.lattice_dim = 8
        if self.channels % self.lattice_dim:
            raise ValueError(f'channels={channels} is not divisible by 8')
        self.lattice_groups = self.channels // self.lattice_dim
        if channel_permutation is None:
            permutation = torch.arange(self.channels, dtype=torch.long)
        else:
            permutation = torch.as_tensor(channel_permutation, dtype=torch.long)
        if permutation.numel() != self.channels or not torch.equal(torch.sort(permutation).values, torch.arange(self.channels)):
            raise ValueError('channel_permutation is not a complete permutation')
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(self.channels)
        self.register_buffer('channel_permutation', permutation)
        self.register_buffer('inverse_channel_permutation', inverse)
        self.e8_per_group_scale = bool(e8_per_group_scale)
        self.quantizer = quantizers.E8LatticeQuantizer(dim=8, init_scale=float(e8_scale), learn_scale=bool(e8_learn_scale), num_groups=self.lattice_groups if self.e8_per_group_scale else 1, per_group=self.e8_per_group_scale, num_shape_groups=self.lattice_groups)
        self.quantizer.skew_raw.requires_grad_(False)
        self.voronoi_mc_samples = int(voronoi_mc_samples)
        self.voronoi_mc_samples_train = int(voronoi_mc_samples_train)
        self.voronoi_likelihood_chunk = int(voronoi_likelihood_chunk)
        self.voronoi_sampler = 'sobol'
        self.voronoi_antithetic = True
        self.voronoi_eval_seed = int(voronoi_eval_seed)
        self.olvq_quantization_mode = 'mean_centered'
        self.ar_context_iterations = int(ar_context_iterations)
        self.adaptive_grouping = str(adaptive_grouping)
        if self.adaptive_grouping not in ADAPTIVE_GROUPING_MODES:
            raise ValueError(f'adaptive_grouping must be one of {ADAPTIVE_GROUPING_MODES}, got {self.adaptive_grouping!r}')
        self.conditional_shape_alpha = 0.0
        self.conditional_shape_log_limit = 2.0
        self.radial_companding_beta = 0.0
        self.radial_companding_threshold = 0.0
        self.companding_law = str(companding_law)
        self.companding_strength_init = float(companding_strength_init)
        self.companding_loading = float(companding_loading)
        self.entropy_constrained_weight = 0.0
        if self.companding_law not in ('none', 'alaw', 'mulaw'):
            raise ValueError(f'unsupported companding_law: {self.companding_law}')
        if self.companding_law != 'none' and self.olvq_quantization_mode != 'mean_centered':
            raise ValueError('adaptive companding requires mean_centered quantization')
        if self.companding_strength_init <= 0 or self.companding_loading <= 0:
            raise ValueError('companding strength and loading must be positive')
        if self.companding_law != 'none':
            if self.companding_law == 'alaw':
                target = max(self.companding_strength_init - 1.0, 0.0001)
            else:
                target = self.companding_strength_init
            self.companding_strength_raw = nn.Parameter(torch.tensor(math.log(math.expm1(target))))
            self.companding_spatial_slope = nn.Parameter(torch.tensor(float(companding_spatial_slope_init)))

    def _e8_gaussian(self, y, scales_hat, means_hat):
        scales_hat = self.gaussian_conditional.lower_bound_scale(scales_hat)
        (y_hat, indices, _) = self._olvq_quantize_with_means(y, means_hat, scales_hat)
        likelihoods = self._estimate_y_likelihoods(y_hat, scales_hat, means_hat)
        return (y_hat, likelihoods, indices)
