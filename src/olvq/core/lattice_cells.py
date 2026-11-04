# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from olvq.core.orthogonal_cell import Lattice8Cell
from olvq.core.nonorthogonal_lattice8 import NonorthogonalBasis8, certified_facets, nearest_lattice_point
from olvq.core.gaussian_voronoi import conditional_cell_proposal, conditional_cell_proposal_grouped, gaussian_voronoi_logmass

def _validate_direction_chunk(value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError('direction_chunk must be a positive integer')

def _validate_training_options(group_proposals, train_order, eval_order, train_proposal_chains, train_proposal_sweeps):
    if not isinstance(group_proposals, bool):
        raise ValueError('group_proposals must be a boolean')
    for (name, value, optional) in (('train_order', train_order, False), ('eval_order', eval_order, False), ('train_proposal_chains', train_proposal_chains, True), ('train_proposal_sweeps', train_proposal_sweeps, True)):
        if optional and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(name + ' must be a positive integer')

class Nonorthogonal8Cell(nn.Module):

    def __init__(self, channels, gc, samples=8192, train_samples=8192, chunk=8, proposal_chains=128, proposal_sweeps=64, direction_chunk=2048, group_proposals=False, train_order=64, eval_order=64, train_proposal_chains=None, train_proposal_sweeps=None):
        super().__init__()
        _validate_direction_chunk(direction_chunk)
        _validate_training_options(group_proposals, train_order, eval_order, train_proposal_chains, train_proposal_sweeps)
        if channels % 8:
            raise ValueError('Channel count must be divisible by eight')
        (self.channels, self.groups) = (channels, channels // 8)
        self.gaussian_conditional = gc
        self.basis = NonorthogonalBasis8(self.groups).double()
        with torch.no_grad():
            self.basis.shape.copy_(self.basis.shape[:1].expand_as(self.basis.shape).clone())
        (self.samples, self.train_samples, self.integration_chunk) = (samples, train_samples, chunk)
        (self.proposal_chains, self.proposal_sweeps) = (proposal_chains, proposal_sweeps)
        self.direction_chunk = direction_chunk
        self.group_proposals = group_proposals
        (self.train_order, self.eval_order) = (train_order, eval_order)
        self.train_proposal_chains = train_proposal_chains
        self.train_proposal_sweeps = train_proposal_sweeps
        self.seed = 104729
        self._facet_key = None
        self._facets = None
        self.last_stats = {}
        self.rate_positions = None
        self.training_rate_terms = []

    def set_mc(self, samples, seed):
        (self.samples, self.seed) = (int(samples), int(seed))

    def _mass(self, symbols, scales, basis, facets, proposal, count, seed):
        return gaussian_voronoi_logmass(symbols, scales, basis, facets, samples=count, order=self.train_order if self.training else self.eval_order, seed=seed, chunk=self.direction_chunk, proposal=proposal)[0]

    def _group_facets(self, bases):
        parameters = (self.basis.shape, self.basis.log_step)
        key = tuple(((id(p), p._version, p.device, p.dtype) for p in parameters))
        if key != self._facet_key:
            host_bases = bases.detach().double().cpu().numpy()
            unique = {}
            facets = []
            for basis in host_bases:
                identity = basis.tobytes()
                if identity not in unique:
                    unique[identity] = certified_facets(basis)[0]
                facets.append(unique[identity])
            self._facets = facets
            self._facet_key = key
        return self._facets

    def forward(self, y, scales, means):
        (batch, channels, height, width) = y.shape
        sigma = self.gaussian_conditional.lower_bound_scale(scales).permute(0, 2, 3, 1).reshape(-1, channels)
        residual = (y - means).permute(0, 2, 3, 1).reshape(-1, channels)
        with torch.no_grad():
            order = torch.argsort(sigma, dim=-1, descending=True, stable=True)
            inverse = torch.argsort(order, dim=-1)
        sigma = sigma.gather(1, order).reshape(-1, self.groups, 8)
        residual = residual.gather(1, order).reshape(-1, self.groups, 8)
        bases = self.basis()
        group_facets = self._group_facets(bases)
        reconstructions = []
        probabilities = []
        count = self.train_samples if self.training else self.samples
        seed = int(torch.initial_seed() % (2 ** 31 - 1)) if self.training else self.seed
        maximum = bases.new_tensor(-float('inf'))
        all_finite = torch.ones((), device=bases.device, dtype=torch.bool)
        proposal_chains = self.train_proposal_chains if self.training and self.train_proposal_chains is not None else self.proposal_chains
        proposal_sweeps = self.train_proposal_sweeps if self.training and self.train_proposal_sweeps is not None else self.proposal_sweeps
        grouped = self.training and self.group_proposals and (self.rate_positions is not None) and (min(self.rate_positions, len(residual)) <= 64)

        def prepared_groups():
            batch_groups = 4 if grouped else 1
            for first in range(0, self.groups, batch_groups):
                prepared = []
                for group in range(first, min(first + batch_groups, self.groups)):
                    (values, ss) = (residual[:, group], sigma[:, group])
                    b = bases[group]
                    facets = group_facets[group]
                    (reconstructed, symbols, _) = nearest_lattice_point(values, b, facets)
                    if self.training:
                        reconstructed = reconstructed + (values - values.detach())
                    if self.training and self.rate_positions is not None:
                        selected = torch.randperm(len(values), device=values.device)[:min(self.rate_positions, len(values))]
                    else:
                        selected = torch.arange(len(values), device=values.device)
                    (selected_symbols, selected_scales) = (symbols[selected], ss[selected])
                    selected_effective = selected_symbols
                    if self.training:
                        selected_values = values[selected]
                        selected_effective = selected_symbols + (selected_values - selected_values.detach()).double() @ torch.linalg.inv(b.detach())
                    prepared.append((group, reconstructed, selected, selected_symbols, selected_scales, selected_effective))
                if grouped:
                    chart = conditional_cell_proposal_grouped(torch.stack([item[3] for item in prepared]), torch.stack([item[4] for item in prepared]), bases[first:first + len(prepared)], group_facets[first:first + len(prepared)], chains=proposal_chains, sweeps=proposal_sweeps, shared_uniforms=True)
                    for (index, item) in enumerate(prepared):
                        yield (item, {key: chart[key][index] for key in ('origin', 'factor', 'mode')})
                else:
                    yield (prepared[0], None)
        for (prepared, full_chart) in prepared_groups():
            (group, reconstructed, selected, selected_symbols, selected_scales, selected_effective) = prepared
            b = bases[group]
            facets = group_facets[group]
            if full_chart is None:
                charts = []
                for offset in range(0, len(selected), 64):
                    charts.append(conditional_cell_proposal(selected_symbols[offset:offset + 64], selected_scales[offset:offset + 64], b, facets, chains=proposal_chains, sweeps=proposal_sweeps, shared_uniforms=True))
                full_chart = {key: charts[0][key] for key in ('origin', 'factor', 'mode')} if len(charts) == 1 else {key: torch.cat([c[key] for c in charts]) for key in ('origin', 'factor', 'mode')}
            qs = []
            for start in range(0, len(selected), self.integration_chunk):
                stop = start + self.integration_chunk
                proposal = {key: value[start:stop] for (key, value) in full_chart.items()}
                call = partial(self._mass, facets=facets, proposal=proposal, count=count, seed=seed)
                inputs = (selected_effective[start:stop], selected_scales[start:stop], b)
                logmass = checkpoint(call, *inputs, use_reentrant=False) if self.training else call(*inputs)
                detached = logmass.detach()
                all_finite = all_finite & torch.isfinite(detached).all()
                maximum = torch.maximum(maximum, detached.max())
                qs.append((logmass / 8).exp()[:, None].expand(-1, 8))
            reconstructions.append(reconstructed.to(y.dtype))
            sampled_probability = torch.cat(qs)
            if self.training and self.rate_positions is not None:
                self.training_rate_terms.append(-sampled_probability.log2().sum() * len(residual) / len(selected))
            probability = torch.ones((len(residual), 8), dtype=sampled_probability.dtype, device=y.device)
            probabilities.append(probability.index_copy(0, selected, sampled_probability))
        if not bool(all_finite & (maximum <= 1e-08)):
            raise RuntimeError('Nonfinite or superunit cell probability')

        def restore(parts):
            packed = torch.stack(parts, dim=1).reshape(-1, channels).gather(1, inverse)
            return packed.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        self.last_stats = dict(samples=count, dimension=8, max_logmass=float(maximum), groups=self.groups, proposal_chains=proposal_chains, proposal_sweeps=proposal_sweeps)
        return (restore(reconstructions) + means, restore(probabilities))

class GeometryRateMixin:

    def quantize(self, y, s, m, index=0, mask=None):
        if mask is None:
            return super().quantize(y, s, m, index)
        selected = mask.flatten().nonzero().flatten()
        packed = [v.flatten(2).index_select(2, selected).unsqueeze(2) for v in (y, s, m)]
        (q, p) = self.quantize(*packed, index)
        qfull = torch.zeros_like(y).flatten(2).index_copy(2, selected, q.squeeze(2)).reshape_as(y)
        pfull = torch.ones(y.shape, dtype=p.dtype, device=p.device).flatten(2).index_copy(2, selected, p.squeeze(2)).reshape_as(y)
        return (qfull, pfull)

    def _ar(self, y, params):
        from torch.nn import functional as F
        n = self.net
        (b, c, h, w) = y.shape
        if self.training:
            yh = y + (y.round() - y).detach()
            for _ in range(2):
                (s, m) = n.entropy_parameters(torch.cat([params, n.context_prediction(yh)], 1)).chunk(2, 1)
                for cell in self.cells:
                    cell.training_rate_terms = []
                (yh, p) = self.quantize(y, s, m)
            return (yh, p)
        k = n.context_prediction.kernel_size[0]
        pad = k // 2
        weights = n.context_prediction.weight * n.context_prediction.mask
        yh = torch.zeros_like(y)
        prob = torch.ones(y.shape, dtype=torch.double, device=y.device)
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

class SubsampledOrthogonal8Cell(Lattice8Cell):
    """Same full reconstruction and unbiased rate-position sampling as dense8."""

    def forward(self, y, scales, means):
        if not self.training or getattr(self, 'rate_positions', None) is None:
            return super().forward(y, scales, means)
        (batch, channels, height, width) = y.shape
        ss = self.gaussian_conditional.lower_bound_scale(scales).permute(0, 2, 3, 1).reshape(-1, channels)
        values = (y - means).permute(0, 2, 3, 1).reshape(-1, channels)
        order = torch.argsort(ss.detach(), dim=-1, descending=True, stable=True)
        inverse = torch.argsort(order, dim=-1)
        ss = ss.gather(1, order).reshape(-1, self.lattice_groups, 8)
        values = values.gather(1, order).reshape_as(ss)
        (rotations, steps) = self.geometry()
        uniforms = self._uniforms(y.device)
        qs = []
        ps = []
        for group in range(self.lattice_groups):
            (v, s) = (values[:, group], ss[:, group])
            r = rotations[group].expand(len(v), 8, 8)
            d = steps[group].expand(len(v), 8)
            (q, _) = self.quantize_flat(v, r, d, True)
            selected = torch.randperm(len(v), device=y.device)[:min(self.rate_positions, len(v))]
            inputs = (v[selected], s[selected], r[selected], d[selected], uniforms)
            (_, p, _) = checkpoint(self._chunk, *inputs, use_reentrant=False)
            self.training_rate_terms.append(-p.double().log2().sum() * len(v) / len(selected))
            pp = torch.ones_like(v).index_copy(0, selected, p)
            qs.append(q)
            ps.append(pp)

        def restore(parts):
            flat = torch.stack(parts, dim=1).reshape(-1, channels).gather(1, inverse)
            return flat.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return (restore(qs) + means, restore(ps))

def clear_training_rate(model):
    for cell in model.cells:
        cell.training_rate_terms = []

def training_rate_bits(model):
    terms = [term for cell in model.cells for term in cell.training_rate_terms]
    if not terms:
        raise RuntimeError('Missing sampled rate terms')
    return torch.stack(terms).sum()
TRAINING_PROFILES = {'strict': dict(train_samples=32768, train_order=64, train_proposal_chains=128, train_proposal_sweeps=64), 'conservative': dict(train_samples=16384, train_order=64, train_proposal_chains=128, train_proposal_sweeps=16), 'balanced': dict(train_samples=8192, train_order=48, train_proposal_chains=64, train_proposal_sweeps=16), 'fast': dict(train_samples=4096, train_order=32, train_proposal_chains=64, train_proposal_sweeps=8)}
