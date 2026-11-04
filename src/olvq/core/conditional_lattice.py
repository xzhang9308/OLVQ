# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from olvq.core.quantizers import _closest_e8_point
from olvq.core.voronoi_sampling import sample_unit_cube

class ConditionalShape(nn.Module):

    def __init__(self, groups, mode='conditional', components=2, log_bound=0.7):
        super().__init__()
        if mode not in ('fixed', 'static', 'conditional', 'rotation', 'diagonal'):
            raise ValueError(mode)
        self.mode = mode
        self.groups = groups
        self.log_bound = float(log_bound)
        self.base = nn.Parameter(torch.zeros(groups, 8, 8))
        self.templates = nn.Parameter(torch.zeros(groups, components, 8, 8))
        self.gate_weight = nn.Parameter(torch.randn(groups, 9, components) * 0.1)
        self.gate_bias = nn.Parameter(torch.zeros(groups, components))
        self.register_buffer('eye', torch.eye(8))
        self.base.requires_grad_(mode != 'fixed')
        for p in (self.templates, self.gate_weight, self.gate_bias):
            p.requires_grad_(mode == 'conditional')

    def forward(self, scales, group_ids):
        if self.mode == 'fixed':
            return self.eye.expand(len(scales), -1, -1)
        per_group = self.mode in ('static', 'rotation', 'diagonal')
        raw = self.base if per_group else self.base[group_ids]
        if self.mode == 'conditional':
            logs = scales.clamp_min(1e-08).log()
            mean = logs.mean(-1, keepdim=True)
            features = torch.cat((logs - mean, mean), -1).clamp(-4, 4)
            gate = torch.tanh(torch.einsum('ni,nik->nk', features, self.gate_weight[group_ids]) + self.gate_bias[group_ids])
            raw = raw + torch.einsum('nk,nkij->nij', gate, self.templates[group_ids])
        if self.mode == 'rotation':
            generator = (raw - raw.transpose(-1, -2)) * 0.5
        else:
            generator = (raw + raw.transpose(-1, -2)) * 0.5
            if self.mode == 'diagonal':
                generator = torch.diag_embed(generator.diagonal(dim1=-2, dim2=-1))
            trace = generator.diagonal(dim1=-2, dim2=-1).mean(-1)
            generator = generator - trace[:, None, None] * self.eye
        norm = torch.linalg.matrix_norm(generator, ord='fro', dim=(-2, -1))
        generator = generator / (norm / self.log_bound).clamp_min(1)[:, None, None]
        result = torch.matrix_exp(generator)
        return result[group_ids] if per_group else result

class ConditionalLatticeMixin:

    def setup_conditional(self, mode='conditional', components=2, log_bound=0.7, integration_chunk=64, integration_checkpoint=True):
        self.lattice_shape = ConditionalShape(self.lattice_groups, mode, components, log_bound)
        self.integration_chunk = int(integration_chunk)
        self.integration_checkpoint = bool(integration_checkpoint)

    def _geometry(self, scales, start=0):
        ids = (torch.arange(len(scales), device=scales.device) + start) % self.lattice_groups
        shape = self.lattice_shape(scales, ids)
        scale = self.quantizer.scale
        scale = scale.expand(self.lattice_groups) if scale.ndim == 0 else scale
        return (shape, scale[ids])

    def _olvq_quantize_with_means(self, y, means, scales=None):
        if scales is None:
            raise ValueError('conditional lattice requires decoder-known scales')
        scales = self.gaussian_conditional.lower_bound_scale(scales)
        grouping = self._adaptive_grouping_index(scales)
        (r, layout) = self._flatten(y - means, grouping)
        (sf, _) = self._flatten(scales, grouping)
        (a, s) = self._geometry(sf)
        transformed = self._component_compand_flat(r, sf) if self.companding_law != 'none' else r
        with torch.no_grad():
            canonical = torch.linalg.solve(a.transpose(-1, -2), transformed.unsqueeze(-1)).squeeze(-1) / s[:, None]
            point = _closest_e8_point(canonical)
            idx = torch.round(point @ self.quantizer.basis_inv)
        rhat = torch.bmm(point.unsqueeze(1), a).squeeze(1) * s[:, None]
        if self.companding_law != 'none':
            rhat = self._component_expand_flat(rhat, sf)
        if self.training:
            rhat = r + (rhat - r).detach() + rhat - rhat.detach()
        yhat = self._unflatten(rhat, layout, grouping) + means
        (b, h, w, _) = layout
        return (yhat, idx.reshape(b, h, w, self.lattice_groups, 8), means)

    def canonical_noise(self, samples, device, dtype, seed=None):
        u = sample_unit_cube(samples, 8, device=device, dtype=dtype, sampler='sobol', antithetic=True, seed=seed)
        p = u @ self.quantizer.basis.to(dtype=dtype)
        return p - _closest_e8_point(p)

    def _cell_logmass(self, center, scales, a, s, noise):
        warped_noise = torch.einsum('jd,ndk->njk', noise, a) * s[:, None, None]
        if self.companding_law == 'none':
            residual = center[:, None, :] + warped_noise
            jacobian = 0.0
        else:
            center_u = self._component_compand_flat(center, scales)
            (residual, jacobian) = self._component_expand_flat(center_u[:, None, :] + warped_noise, scales[:, None, :], return_logdet=True)
        logpdf = -0.5 * (residual / scales[:, None, :]).square().sum(-1) - 4 * math.log(2 * math.pi) - scales.log().sum(-1)[:, None] + jacobian
        logvol = 8 * s.log() + torch.linalg.slogdet(a)[1]
        return logvol + torch.logsumexp(logpdf, 1) - math.log(len(noise))

    def _estimate_y_likelihoods(self, y_hat, scales_hat, means_hat, num_samples=None, seed=None, return_diagnostics=False):
        scales_hat = self.gaussian_conditional.lower_bound_scale(scales_hat)
        grouping = self._adaptive_grouping_index(scales_hat)
        (center, layout) = self._flatten(y_hat - means_hat, grouping)
        (sf, _) = self._flatten(scales_hat, grouping)
        n = num_samples or (self.voronoi_mc_samples_train if self.training else self.voronoi_mc_samples)
        if seed is None and (not self.training):
            seed = self.voronoi_eval_seed
        noise = self.canonical_noise(n, center.device, center.dtype, seed)
        masses = []
        geometry = self._geometry(sf) if self.training else None
        for start in range(0, len(center), self.integration_chunk):
            c = center[start:start + self.integration_chunk]
            sc = sf[start:start + self.integration_chunk]
            (a, s) = tuple((v[start:start + self.integration_chunk] for v in geometry)) if geometry is not None else self._geometry(sc, start)
            if self.training and self.integration_checkpoint and torch.is_grad_enabled():
                lm = checkpoint(self._cell_logmass, c, sc, a, s, noise, use_reentrant=False)
            else:
                lm = self._cell_logmass(c, sc, a, s, noise)
            masses.append(lm)
        logmass = torch.cat(masses)
        per_channel = (logmass.clamp(max=0) / 8).exp()[:, None].expand(-1, 8)
        likelihood = self._unflatten(per_channel, layout, grouping)
        likelihood = self.gaussian_conditional.likelihood_lower_bound(likelihood)
        if return_diagnostics:
            return (likelihood, {'mc_samples': n, 'mass_above_one_fraction': float((logmass > 0).float().mean())})
        return likelihood

    def compress(self, x):
        raise NotImplementedError('Use the public encode CLI for supported backbones')

    def decompress(self, strings, shape):
        raise NotImplementedError('Use the public decode CLI for supported backbones')
