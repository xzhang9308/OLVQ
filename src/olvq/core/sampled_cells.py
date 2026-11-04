# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from olvq.core.e8_integration import mis_cell_logmass
from olvq.core.lattice_cells import GeometryRateMixin, Nonorthogonal8Cell, SubsampledOrthogonal8Cell, clear_training_rate, training_rate_bits
from olvq.core.backbone_adapter import ENTROPY, RotationCell, RotationLIC
from olvq.core.nonorthogonal_lattice8 import nearest_lattice_point
ROUTES = {'e8': 'e8', 'e8_rotation': 'e8', 'orth': 'orthogonal', 'orthogonal': 'orthogonal', 'nonorth': 'nonorthogonal', 'nonorthogonal': 'nonorthogonal', 'scalar': 'scalar'}

def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value

class SampledRotationCell(RotationCell):
    """Rotated E8 assignment with double-precision MIS cell masses.

The detached importance proposal has uniform
support on the full E8 cell and a truncated Gaussian component for narrow
conditional densities. Geometry and the encoder STE differentiate the same
physical integration domain as the E8 conditional implementation.
"""
    _cell_logmass = mis_cell_logmass

    def __init__(self, *args, rate_positions=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_positions = rate_positions
        self.training_rate_terms = []
        self.last_stats = {}
        self._noise_key = None
        self._noise = None

    def _estimate_y_likelihoods(self, y_hat, scales_hat, means_hat, num_samples=None, seed=None, return_diagnostics=False):
        bounded = self.gaussian_conditional.lower_bound_scale(scales_hat)
        grouping = self._adaptive_grouping_index(bounded)
        (center, layout) = self._flatten(y_hat - means_hat, grouping)
        (sigma, _) = self._flatten(bounded, grouping)
        count = int(num_samples or (self.voronoi_mc_samples_train if self.training else self.voronoi_mc_samples))
        if seed is None:
            seed = int(torch.initial_seed() % (2 ** 31 - 1)) if self.training else int(self.voronoi_eval_seed)
        key = (count, seed, str(center.device))
        if self._noise_key != key:
            self._noise = self.canonical_noise(count, center.device, torch.float64, seed)
            self._noise_key = key
        noise = self._noise
        positions = len(center) // self.lattice_groups
        selected = []
        for group in range(self.lattice_groups):
            if self.training and self.rate_positions is not None:
                index = torch.randperm(positions, device=center.device)[:min(self.rate_positions, positions)]
            else:
                index = torch.arange(positions, device=center.device)
            selected.append(index * self.lattice_groups + group)
        selected = torch.cat(selected)
        ids = selected % self.lattice_groups
        selected_center = center[selected].double()
        if self.training:
            central = (selected_center.detach() == 0).all(-1, keepdim=True)
            selected_center = torch.where(central, selected_center.detach(), selected_center)
        selected_sigma = sigma[selected].double()
        shapes = self.lattice_shape(selected_sigma, ids).double()
        scales = self.quantizer.scale
        scales = scales.expand(self.lattice_groups) if scales.ndim == 0 else scales
        scales = scales[ids].double()
        parts = []
        maxima = []
        for first in range(0, len(selected), self.integration_chunk):
            last = first + self.integration_chunk
            args = (selected_center[first:last], selected_sigma[first:last], shapes[first:last], scales[first:last], noise)
            if self.training and self.integration_checkpoint and torch.is_grad_enabled():
                logmass = checkpoint(self._cell_logmass, *args, use_reentrant=False)
            else:
                logmass = self._cell_logmass(*args)
            if not bool(torch.isfinite(logmass.detach()).all()):
                raise RuntimeError('Nonfinite E8 MIS probability')
            maxima.append(logmass.detach().max())
            part = (logmass.clamp(max=0) / 8).exp()[:, None].expand(-1, 8)
            if self.gaussian_conditional.use_likelihood_bound:
                part = self.gaussian_conditional.likelihood_lower_bound(part)
            parts.append(part)
        probability = torch.cat(parts)
        if self.training:
            self.training_rate_terms.append(-probability.log2().sum() * positions / (len(selected) / self.lattice_groups))
        complete = torch.ones(center.shape, device=center.device, dtype=probability.dtype)
        complete = complete.index_copy(0, selected, probability)
        self.last_stats = {'samples': count, 'integrator': 'mis', 'dimension': 8, 'groups': self.lattice_groups, 'max_raw_logmass': float(torch.stack(maxima).max())}
        likelihood = self._unflatten(complete, layout, grouping)
        return (likelihood, dict(self.last_stats)) if return_diagnostics else likelihood

class CentralNonorthogonal8Cell(Nonorthogonal8Cell):
    """Preserve input-gradient symmetry for the central lattice symbol."""

    def _mass(self, symbols, *args, **kwargs):
        central = (symbols.detach() == 0).all(dim=-1, keepdim=True)
        effective = torch.where(central, symbols.detach(), symbols)
        return super()._mass(effective, *args, **kwargs)

def _reconstruct_only(cell, y, scales, means):
    """Exact assignment for causal capture; probability is integrated later."""
    if isinstance(cell, SampledRotationCell):
        return cell._olvq_quantize_with_means(y, means, scales)[0]
    (batch, channels, height, width) = y.shape
    sigma = cell.gaussian_conditional.lower_bound_scale(scales)
    sigma = sigma.permute(0, 2, 3, 1).reshape(-1, channels)
    residual = (y - means).permute(0, 2, 3, 1).reshape(-1, channels)
    order = torch.argsort(sigma, dim=-1, descending=True, stable=True)
    inverse = torch.argsort(order, dim=-1)
    groups = channels // 8
    values = residual.gather(1, order).reshape(-1, groups, 8)
    reconstructed = []
    if isinstance(cell, Nonorthogonal8Cell):
        bases = cell.basis()
        facets = cell._group_facets(bases)
        for group in range(groups):
            reconstructed.append(nearest_lattice_point(values[:, group], bases[group], facets[group])[0].to(y.dtype))
    else:
        (rotations, steps) = cell.geometry()
        for group in range(groups):
            count = len(values)
            reconstructed.append(cell.quantize_flat(values[:, group], rotations[group].expand(count, 8, 8), steps[group].expand(count, 8), False)[0])
    packed = torch.stack(reconstructed, 1).reshape(-1, channels).gather(1, inverse)
    return packed.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous() + means
