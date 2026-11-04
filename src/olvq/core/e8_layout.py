# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from olvq.core import quantizers
from olvq.core.voronoi_sampling import SAMPLERS, sample_unit_cube
QUANTIZATION_MODES = ('direct', 'mean_centered')
COMPANDING_LAWS = ('none', 'alaw', 'mulaw')
ADAPTIVE_GROUPING_MODES = ('none', 'sigma_sort', 'sigma_interleave')

class MbtOLVQBase:

    def _adaptive_grouping_index(self, scales):
        """Per-position channel ordering derived only from the coded scales.

        The static ``channel_permutation`` groups channels by their scale
        averaged over a calibration set.  Predicted scales vary by orders of
        magnitude across positions, so a single average ordering still mixes
        near-dead and active channels inside one lattice cell.  Ordering per
        position stays decoder-reproducible because the scales are available
        on both sides before the position is decoded.
        """
        mode = self.adaptive_grouping
        if mode == 'none':
            return None
        if scales is None:
            raise ValueError('adaptive grouping requires predicted scales')
        with torch.no_grad():
            bounded = self.gaussian_conditional.lower_bound_scale(scales.detach())
            flat = bounded.permute(0, 2, 3, 1).reshape(-1, self.channels)
            order = torch.argsort(flat, dim=1, descending=True, stable=True)
            if mode == 'sigma_interleave':
                order = order.reshape(-1, self.lattice_dim, self.lattice_groups).transpose(1, 2).reshape(-1, self.channels)
            inverse = torch.argsort(order, dim=1)
        return (order, inverse)

    def _flatten(self, y, grouping=None):
        if grouping is None:
            y = y.index_select(1, self.channel_permutation)
            y = y.permute(0, 2, 3, 1).contiguous()
            y_shape = y.size()
        else:
            y = y.permute(0, 2, 3, 1).contiguous()
            y_shape = y.size()
            y = torch.gather(y.reshape(-1, self.channels), 1, grouping[0])
        y = y.reshape(-1, self.lattice_groups, self.lattice_dim)
        y = y.reshape(-1, self.lattice_dim)
        return (y, y_shape)

    def _unflatten(self, y_hat, y_shape, grouping=None):
        (b, h, w, _) = y_shape
        y_hat = y_hat.reshape(b, h, w, self.lattice_groups, self.lattice_dim)
        y_hat = y_hat.reshape(y_shape)
        if grouping is not None:
            y_hat = torch.gather(y_hat.reshape(-1, self.channels), 1, grouping[1]).reshape(y_shape)
            return y_hat.permute(0, 3, 1, 2).contiguous()
        y_hat = y_hat.permute(0, 3, 1, 2).contiguous()
        y_hat = y_hat.index_select(1, self.inverse_channel_permutation)
        return y_hat

    def _quantization_diagnostics(self, y, y_hat, means_hat):
        """Compare lattice MSE with scalar rounding at matched cell volume."""
        if self.olvq_quantization_mode == 'mean_centered':
            residual = y - means_hat
            residual_hat = y_hat - means_hat
        else:
            residual = y
            residual_hat = y_hat
        (residual_flat, _) = self._flatten(residual)
        (residual_hat_flat, _) = self._flatten(residual_hat)
        grouped = residual_flat.reshape(-1, self.lattice_groups, self.lattice_dim)
        log_volume = self._log_voronoi_volume().to(device=grouped.device, dtype=grouped.dtype)
        if log_volume.ndim == 0:
            log_volume = log_volume.expand(self.lattice_groups)
        scalar_step = torch.exp(log_volume / self.lattice_dim)
        scalar_hat = torch.round(grouped / scalar_step[None, :, None]) * scalar_step[None, :, None]
        return {'lattice_squared_error': (residual_flat - residual_hat_flat).square().sum().reshape(1), 'scalar_squared_error': (grouped - scalar_hat).square().sum().reshape(1), 'num_values': residual_flat.new_tensor([residual_flat.numel()], dtype=torch.long)}
