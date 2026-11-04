# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from compressai import zoo
import compressai.ops

def ste_round(x):
    return x + (torch.round(x) - x).detach()
if not hasattr(compressai.ops, 'ste_round'):
    compressai.ops.ste_round = ste_round
from olvq.core.conditional_lattice import ConditionalLatticeMixin
from olvq.core.e8_gaussian import FixedE8GaussianMixin
CLASSIC = {'bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018', 'cheng2020-anchor', 'cheng2020-attn'}
AR = {'mbt2018', 'cheng2020-anchor', 'cheng2020-attn'}
ENTROPY = ('h_a.', 'h_s.', 'h_mean_s.', 'h_scale_s.', 'entropy_bottleneck.', 'context_prediction.', 'entropy_parameters.', 'cc_mean_transforms.', 'cc_scale_transforms.', 'atten_mean.', 'atten_scale.', 'local_context.', 'channel_context.', 'global_inter_context.', 'global_intra_context.', 'entropy_parameters_anchor.', 'entropy_parameters_nonanchor.')

class RotationCell(ConditionalLatticeMixin, FixedE8GaussianMixin, nn.Module):

    def __init__(self, channels, gc, mc=8192, train_mc=512, chunk=128):
        nn.Module.__init__(self)
        self._setup_fixed_e8(channels, adaptive_grouping='sigma_sort', e8_learn_scale=True, e8_per_group_scale=True, voronoi_mc_samples=mc, voronoi_mc_samples_train=train_mc)
        self.gaussian_conditional = gc
        self.setup_conditional('rotation', integration_chunk=chunk)

    def forward(self, y, scales, means):
        return self._e8_gaussian(y, scales, means)[:2]

class RotationLIC(nn.Module):

    def quantize(self, y, s, m, index=0, mask=None):
        if mask is not None:
            (b, c, h, w) = y.shape
            selected = mask.flatten().nonzero().flatten()
            yp = y.flatten(2).index_select(2, selected).unsqueeze(2)
            sp = s.flatten(2).index_select(2, selected).unsqueeze(2)
            mp = m.flatten(2).index_select(2, selected).unsqueeze(2)
            (q, p) = self.quantize(yp, sp, mp, index)
            qfull = torch.zeros_like(y).flatten(2).index_copy(2, selected, q.squeeze(2)).reshape_as(y)
            pfull = torch.ones_like(y).flatten(2).index_copy(2, selected, p.squeeze(2)).reshape_as(y)
            return (qfull, pfull)
        if self.variant == 'scalar':
            return self.net.gaussian_conditional(y, s, m, training=False)
        return self.cells[index](y, s, m)

    def _z(self, y, absolute=False, ste=False):
        n = self.net
        z = n.h_a(y.abs() if absolute else y)
        (zh, zp) = n.entropy_bottleneck(z)
        if ste:
            offset = n.entropy_bottleneck._get_medians()
            zh = ste_round(z - offset) + offset
        return (zh, zp)

    def _ar(self, y, params):
        n = self.net
        (b, c, h, w) = y.shape
        if self.training:
            yh = ste_round(y)
            for _ in range(2):
                (s, m) = n.entropy_parameters(torch.cat([params, n.context_prediction(yh)], 1)).chunk(2, 1)
                (yh, p) = self.quantize(y, s, m)
            return (yh, p)
        k = n.context_prediction.kernel_size[0]
        pad = k // 2
        weights = n.context_prediction.weight * n.context_prediction.mask
        yh = torch.zeros_like(y)
        prob = torch.ones_like(y)
        for t in range((pad + 1) * (h - 1) + w):
            rows = torch.arange(h, device=y.device)
            cols = t - (pad + 1) * rows
            keep = (cols >= 0) & (cols < w)
            rows = rows[keep]
            cols = cols[keep]
            if len(rows) == 0:
                continue
            patches = F.pad(yh, (pad, pad, pad, pad)).unfold(2, k, 1).unfold(3, k, 1)
            chosen = patches[:, :, rows, cols].permute(0, 2, 1, 3, 4).reshape(b * len(rows), -1)
            ctx = F.linear(chosen, weights.reshape(weights.shape[0], -1), n.context_prediction.bias)
            ctx = ctx.reshape(b, len(rows), -1).permute(0, 2, 1).unsqueeze(-1)
            hyper = params[:, :, rows, cols].unsqueeze(-1)
            (s, m) = n.entropy_parameters(torch.cat([hyper, ctx], 1)).chunk(2, 1)
            (q, p) = self.quantize(y[:, :, rows, cols].unsqueeze(-1), s, m)
            yh[:, :, rows, cols] = q.squeeze(-1)
            prob[:, :, rows, cols] = p.squeeze(-1)
        return (yh, prob)

    def _slices(self, x):
        n = self.net
        y = n.g_a(x)
        (h, w) = y.shape[-2:]
        (zh, zp) = self._z(y, ste=True)
        ls = n.h_scale_s(zh)
        lm = n.h_mean_s(zh)
        hats = []
        probs = []
        for (i, ys) in enumerate(y.chunk(n.num_slices, 1)):
            support = hats if n.max_support_slices < 0 else hats[:n.max_support_slices]
            ms = torch.cat([lm] + support, 1)
            ss = torch.cat([ls] + support, 1)
            if self.family == 'tcm':
                ms = n.atten_mean[i](ms)
                ss = n.atten_scale[i](ss)
            m = n.cc_mean_transforms[i](ms)[:, :, :h, :w]
            s = n.cc_scale_transforms[i](ss)[:, :, :h, :w]
            (q, p) = self.quantize(ys, s, m, i)
            q = q + 0.5 * torch.tanh(n.lrp_transforms[i](torch.cat([ms, q], 1)))
            hats.append(q)
            probs.append(p)
        yh = torch.cat(hats, 1)
        xh = n.g_s(yh)
        return {'x_hat': xh, 'likelihoods': {'y': torch.cat(probs, 1), 'z': zp}}

    def forward(self, x):
        if self.family == 'tcm':
            return self._slices(x)
        n = self.net
        y = n.g_a(x)
        (zh, zp) = self._z(y, self.family == 'bmshj2018-hyperprior')
        params = n.h_s(zh)
        if self.family in AR:
            (yh, yp) = self._ar(y, params)
        else:
            if self.family == 'bmshj2018-hyperprior':
                s = params
                m = torch.zeros_like(s)
            else:
                (s, m) = params.chunk(2, 1)
            (yh, yp) = self.quantize(y, s, m)
        return {'x_hat': n.g_s(yh), 'likelihoods': {'y': yp, 'z': zp}}
