# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from olvq.core.gaussian_rectangle import gaussian_rectangle_logmass, sobol_uniforms
from olvq.core.backbone_adapter import ENTROPY, RotationLIC

class Lattice8Cell(nn.Module):
    """An independent dense eight-dimensional basis per scale-ranked group."""

    def __init__(self, channels, gaussian_conditional, mc=8192, train_mc=512, chunk=64, integration_checkpoint=True):
        super().__init__()
        if channels <= 0 or channels % 8:
            raise ValueError('The channel count must be a positive multiple of eight')
        if min(mc, train_mc, chunk) <= 0:
            raise ValueError('Integration budgets and chunk size must be positive')
        self.channels = int(channels)
        self.lattice_groups = channels // 8
        self.gaussian_conditional = gaussian_conditional
        self.rotation_params = nn.Parameter(torch.zeros(self.lattice_groups, 28))
        self.log_scale = nn.Parameter(torch.zeros(self.lattice_groups, 8))
        generators = torch.zeros(28, 8, 8)
        (row, col) = torch.triu_indices(8, 8, offset=1)
        generators[torch.arange(28), row, col] = 1
        generators[torch.arange(28), col, row] = -1
        self.register_buffer('skew_generators', generators)
        self.mc_samples = int(mc)
        self.train_mc_samples = int(train_mc)
        self.eval_seed = 104729
        self.integration_chunk = int(chunk)
        self.integration_checkpoint = bool(integration_checkpoint)
        self.adaptive_grouping = 'sigma_sort'
        self.factor_floor = float(gaussian_conditional.likelihood_lower_bound.bound.detach().cpu()) if gaussian_conditional.use_likelihood_bound else 0.0
        self._uniform_key = None
        self._uniform_cache = None
        self._last_diagnostics = None

    def geometry(self):
        skew = torch.einsum('gp,pij->gij', self.rotation_params, self.skew_generators)
        rotation = torch.matrix_exp(skew)
        bound = math.log(2.0)
        steps = (bound * torch.tanh(self.log_scale / bound)).exp()
        return (rotation, steps)

    def generator_matrix(self):
        (rotation, steps) = self.geometry()
        return steps[:, :, None] * rotation

    def set_mc(self, count, seed):
        if count <= 0:
            raise ValueError('Integration sample count must be positive')
        (self.mc_samples, self.eval_seed) = (int(count), int(seed))

    def _uniforms(self, device):
        count = self.train_mc_samples if self.training else self.mc_samples
        seed = int(torch.initial_seed() % (2 ** 31 - 1)) if self.training else self.eval_seed
        key = (count, seed, str(device))
        if self._uniform_key != key:
            self._uniform_cache = sobol_uniforms(count, 7, device=device, dtype=torch.float64, seed=seed, antithetic=True)
            self._uniform_key = key
        return self._uniform_cache

    @staticmethod
    def quantize_flat(residual, rotation, steps, training=False):
        """Hard exact CVP; encoder STE plus the genuine basis derivative.

        The integer coefficient is detached. Reconstruction differentiates
        through its basis. The optional identity input derivative is separate,
        so it does not cancel or duplicate that basis derivative.
        """
        with torch.no_grad():
            canonical = torch.bmm(residual[:, None], rotation.transpose(1, 2)).squeeze(1) / steps
            symbols = torch.round(canonical)
        reconstruction = torch.bmm((symbols * steps)[:, None], rotation).squeeze(1)
        if training:
            reconstruction = reconstruction + (residual - residual.detach())
        return (reconstruction, symbols)

    def _chunk(self, residual, sigma, rotation, steps, uniforms):
        (reconstruction, symbols) = self.quantize_flat(residual, rotation, steps, self.training)
        center = symbols.to(torch.float64)
        if self.training:
            input_ste = residual - residual.detach()
            center = center + (torch.bmm(input_ste[:, None], rotation.detach().transpose(1, 2)).squeeze(1) / steps.detach()).to(torch.float64)
        (rd, sd, dd) = (rotation.double(), sigma.double(), steps.double())
        covariance = torch.bmm(rd * sd.square()[:, None, :], rd.transpose(1, 2))
        covariance = covariance / dd[:, :, None] / dd[:, None, :]
        cholesky = torch.linalg.cholesky(covariance)
        (_, diagnostics) = gaussian_rectangle_logmass(center - 0.5, center + 0.5, cholesky, uniforms, factor_floor=self.factor_floor, return_diagnostics=True, return_factors=True)
        factors = diagnostics['log_factors']
        prefix = torch.logsumexp(factors.cumsum(-1), dim=1) - math.log(factors.shape[1])
        per_channel_logmass = torch.diff(torch.cat([torch.zeros_like(prefix[:, :1]), prefix], dim=-1), dim=-1)
        likelihood = per_channel_logmass.clamp(max=0).exp().to(residual.dtype)
        if self.gaussian_conditional.use_likelihood_bound:
            likelihood = self.gaussian_conditional.likelihood_lower_bound(likelihood)
        count = len(residual)
        stats = torch.stack([likelihood.new_tensor(float(count), dtype=torch.float64), diagnostics['floor_fraction'].detach().double().sum(), diagnostics['floor_logmass_delta'].detach().double().sum(), diagnostics['floor_logmass_delta'].detach().double().max(), diagnostics['raw_logmass'].detach().double().max()])
        return (reconstruction, likelihood, stats)

    def forward(self, y, scales, means):
        if y.ndim != 4 or y.shape != scales.shape or y.shape != means.shape:
            raise ValueError('y, scales, and means must have matching NCHW shapes')
        if y.shape[1] != self.channels:
            raise ValueError('Unexpected channel count')
        bounded = self.gaussian_conditional.lower_bound_scale(scales)
        (batch, channels, height, width) = y.shape
        sigma = bounded.permute(0, 2, 3, 1).reshape(-1, channels)
        residual = (y - means).permute(0, 2, 3, 1).reshape(-1, channels)
        with torch.no_grad():
            order = torch.argsort(sigma.detach(), dim=-1, descending=True, stable=True)
            inverse = torch.argsort(order, dim=-1)
        sigma = sigma.gather(1, order).reshape(-1, 8)
        residual = residual.gather(1, order).reshape(-1, 8)
        (rotation, steps) = self.geometry()
        uniforms = self._uniforms(y.device)
        (reconstruction_parts, likelihood_parts, statistics) = ([], [], [])
        for start in range(0, len(residual), self.integration_chunk):
            stop = min(start + self.integration_chunk, len(residual))
            group_ids = torch.arange(start, stop, device=y.device) % self.lattice_groups
            inputs = (residual[start:stop], sigma[start:stop], rotation[group_ids], steps[group_ids], uniforms)
            if self.training and self.integration_checkpoint and torch.is_grad_enabled():
                (reconstruction, likelihood, stats) = checkpoint(self._chunk, *inputs, use_reentrant=False)
            else:
                (reconstruction, likelihood, stats) = self._chunk(*inputs)
            reconstruction_parts.append(reconstruction)
            likelihood_parts.append(likelihood)
            statistics.append(stats)

        def restore(values):
            flattened = torch.cat(values).reshape(-1, channels).gather(1, inverse)
            return flattened.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        stats = torch.stack(statistics)
        self._last_diagnostics = torch.cat([stats[:, :3].sum(0), stats[:, 3:].max(0).values]).detach()
        return (restore(reconstruction_parts) + means, restore(likelihood_parts))

    def numerical_diagnostics(self):
        if self._last_diagnostics is None:
            return None
        (count, floors, delta, maximum_delta, maximum_logmass) = self._last_diagnostics.cpu().tolist()
        return {'vectors': int(count), 'conditional_floor_fraction': floors / count, 'mean_floor_logmass_delta': delta / count, 'max_floor_logmass_delta': maximum_delta, 'max_raw_logmass': maximum_logmass}
