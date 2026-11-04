# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.optimize import LinearConstraint, minimize
from olvq.core.gaussian_rectangle import sobol_uniforms, _conditional_interval
from olvq.core.nonorthogonal_lattice8 import _legendre, nearest_lattice_point

@torch.no_grad()
def gaussian_cell_proposal(symbols, sigma, basis, facets, interior=0.95):
    """Gaussian mode constrained to the cell; strict-interior radial origin.

    Proposal construction is detached. Changing a valid integration origin or
    coordinate chart leaves the exact integral unchanged, so its derivative
    need not include the optimizer's derivative. Finite QMC error DOES depend
    on the chart, and must be checked with independent scrambles/refinement.
    """
    if not 0 <= interior < 1:
        raise ValueError('The integration origin must stay strictly interior')
    (b, s, n) = (basis.double(), sigma.double(), symbols.double())
    if n.ndim != 2 or s.shape != n.shape or n.shape[1] != len(b):
        raise ValueError('Expected symbols and positive scales [batch,dimension]')
    if not bool(torch.isfinite(s).all() & (s > 0).all()):
        raise ValueError('Nonfinite or nonpositive Gaussian scales')
    coefficients = torch.as_tensor(facets, device=b.device, dtype=b.dtype)
    normals = coefficients @ b
    half = 0.5 * normals.square().sum(-1)
    centers = n @ b
    (vv, hh) = (normals.cpu().numpy(), half.cpu().numpy())
    modes = []
    for (cc, ss) in zip(centers.cpu().numpy(), s.cpu().numpy()):
        if np.max(-vv @ cc - hh) <= 0:
            modes.append(np.zeros_like(cc))
            continue
        constraint = LinearConstraint(vv * ss[None], -np.inf, hh + vv @ cc)
        fit = minimize(lambda z: 0.5 * z @ z, cc / ss, jac=lambda z: z, constraints=constraint, method='SLSQP', options={'ftol': 1e-10, 'maxiter': 500})
        violation = np.max(vv * ss[None] @ fit.x - hh - vv @ cc)
        if not fit.success or violation > 1e-07:
            raise RuntimeError('Constrained Gaussian mode did not converge: ' + fit.message)
        modes.append(fit.x * ss)
    mode = torch.as_tensor(np.asarray(modes), device=b.device, dtype=b.dtype)
    cell_scale = torch.linalg.det(b).abs().pow(1 / len(b))
    chart_scale = torch.minimum(s, cell_scale * 0.5)
    origin = interior * (mode - centers)
    margin = half[None] - origin @ normals.T
    if not bool((margin > 0).all()):
        raise RuntimeError('Integration origin is not strictly inside the cell')
    return {'origin': origin.detach(), 'scale': chart_scale.detach(), 'mode': mode.detach()}

@torch.no_grad()
def conditional_cell_proposal(symbols, sigma, basis, facets, chains=128, sweeps=64, seed=67867967, shared_uniforms=False):
    """Learn an integration chart from truncated-Gaussian Gibbs proposals.

    The Gibbs samples NEVER estimate the mass. They only select an interior
    origin and a full covariance chart. Even an unconverged chain cannot change
    the exact radial integral or importance target; it can worsen variance.
    Fixed seeds make proposal construction replayable. No per-model parameters.
    """
    (b, s, n) = (basis.double(), sigma.double(), symbols.double())
    d = len(b)
    coeff = torch.as_tensor(facets, device=b.device, dtype=b.dtype)
    normals = coeff @ b
    center = n @ b
    limits = 0.5 * normals.square().sum(-1)[None] + center @ normals.T
    a = normals[None] * s[:, None]
    z = (center / s)[:, None].expand(-1, chains, -1).clone()
    generator = torch.Generator(device=b.device).manual_seed(seed)
    uniforms = torch.rand(sweeps, d, 1 if shared_uniforms else len(n), chains, generator=generator, device=b.device, dtype=b.dtype)
    eps = torch.finfo(b.dtype).eps / 2
    uniforms = uniforms.clamp(eps, 1 - eps)
    slack = limits[:, None] - torch.einsum('ncd,nfd->ncf', z, a)
    for sweep in range(sweeps):
        for axis in range(d):
            component = a[:, None, :, axis]
            divisor = torch.where(component.abs() > 1e-14, component, torch.ones_like(component))
            ratios = slack / divisor
            low = torch.where(component < -1e-14, ratios, -torch.inf).amax(-1) + z[:, :, axis]
            high = torch.where(component > 1e-14, ratios, torch.inf).amin(-1) + z[:, :, axis]
            (_, sample) = _conditional_interval(low, high, uniforms[sweep, axis])
            delta = sample - z[:, :, axis]
            z[:, :, axis] = sample
            slack -= delta[:, :, None] * component
    physical = z * s[:, None]
    mean = physical.mean(1)
    residual = physical - mean[:, None]
    covariance = residual.transpose(1, 2) @ residual / max(chains - 1, 1)
    diagonal = s.square().clamp_max(torch.linalg.det(b).abs().pow(2 / d)) * 1e-05
    covariance = 1.5 * covariance + torch.diag_embed(diagonal)
    factor = torch.linalg.cholesky(covariance)
    origin = 0.999 * (mean - center)
    if not bool(torch.isfinite(factor).all()):
        raise RuntimeError('Nonfinite conditional chart covariance')
    margin = 0.5 * normals.square().sum(-1)[None] - origin @ normals.T
    if not bool((margin > 0).all()):
        raise RuntimeError('Conditional chart origin is not inside the cell')
    return {'origin': origin, 'factor': factor, 'mode': center + origin, 'chains': chains, 'sweeps': sweeps, 'proposal_seed': seed}

@torch.no_grad()
def conditional_cell_proposal_grouped(symbols, sigma, bases, facets, *, chains=128, sweeps=64, seed=67867967, shared_uniforms=True, group_chunk=None):
    """Batch independent conditional charts without changing their Gibbs draws.

    Inputs have shapes [groups, positions, dimension] and [groups, dimension,
    dimension]; ``facets`` contains one certified coefficient array per group.
    All groups replay the same private RNG stream, exactly as separate calls to
    :func:`conditional_cell_proposal` with the same seed. Padded constraints are
    excluded from both Gibbs bounds and the final strict-interior check.

    This only changes scheduling, not the sampling budget or FP64 arithmetic.
    ``group_chunk`` caps groups processed together; the default limits each
    slack/ratio array to approximately 64 MiB (at least one group). Charts are
    detached and cannot propagate a derivative through proposal construction.
    """
    if any((not isinstance(value, torch.Tensor) for value in (symbols, sigma, bases))):
        raise ValueError('Expected tensor symbols, scales, and bases')
    if symbols.ndim != 3 or sigma.shape != symbols.shape or bases.ndim != 3 or (bases.shape[0] != symbols.shape[0]) or (bases.shape[1:] != (symbols.shape[-1], symbols.shape[-1])) or (min(symbols.shape) <= 0):
        raise ValueError('Expected symbols/scales [groups,positions,dimension] and square bases')
    if symbols.device != bases.device or sigma.device != bases.device:
        raise ValueError('Grouped proposal tensors must share a device')
    if not isinstance(chains, int) or isinstance(chains, bool) or chains <= 0 or (not isinstance(sweeps, int)) or isinstance(sweeps, bool) or (sweeps <= 0):
        raise ValueError('Proposal chains and sweeps must be positive integers')
    if group_chunk is not None and (not isinstance(group_chunk, int) or isinstance(group_chunk, bool) or group_chunk <= 0):
        raise ValueError('Proposal group chunk must be a positive integer')
    (b, s, n) = (bases.double(), sigma.double(), symbols.double())
    (groups, positions, d) = n.shape
    if len(facets) != groups:
        raise ValueError('Expected one facet array per group')
    determinant = torch.linalg.det(b).abs()
    if not bool(torch.isfinite(n).all() & torch.isfinite(s).all() & (s > 0).all() & torch.isfinite(b).all() & torch.isfinite(determinant).all() & (determinant > 0).all()):
        raise ValueError('Expected finite symbols, positive scales, and nonsingular bases')
    coefficient_arrays = []
    for group_facets in facets:
        coefficients = torch.as_tensor(group_facets, device=b.device, dtype=b.dtype)
        if coefficients.ndim != 2 or coefficients.shape[1] != d or len(coefficients) == 0:
            raise ValueError('Each facet array must have shape [facets,dimension]')
        if not bool(torch.isfinite(coefficients).all() & (coefficients != 0).any(-1).all()):
            raise ValueError('Facet coefficients must be finite and nonzero')
        coefficient_arrays.append(coefficients)
    maximum_facets = max((len(coefficients) for coefficients in coefficient_arrays))
    if group_chunk is None:
        group_chunk = max(1, 8 * 1024 * 1024 // (positions * chains * maximum_facets))
    generator = torch.Generator(device=b.device).manual_seed(seed)
    uniforms = torch.rand(sweeps, d, 1 if shared_uniforms else positions, chains, generator=generator, device=b.device, dtype=b.dtype)
    eps = torch.finfo(b.dtype).eps / 2
    uniforms = uniforms.clamp(eps, 1 - eps)
    (origins, factors, modes) = ([], [], [])
    for start in range(0, groups, group_chunk):
        stop = min(start + group_chunk, groups)
        (block_b, block_s, block_n) = (b[start:stop], s[start:stop], n[start:stop])
        counts = [len(coefficients) for coefficients in coefficient_arrays[start:stop]]
        coefficients = b.new_zeros(stop - start, max(counts), d)
        valid = torch.zeros(stop - start, max(counts), device=b.device, dtype=torch.bool)
        for (local, (count, values)) in enumerate(zip(counts, coefficient_arrays[start:stop])):
            coefficients[local, :count] = values
            valid[local, :count] = True
        normals = coefficients @ block_b
        center = block_n @ block_b
        half = 0.5 * normals.square().sum(-1)
        limits = half[:, None] + center @ normals.transpose(1, 2)
        a = normals[:, None] * block_s[:, :, None]
        z = (center / block_s)[:, :, None].expand(-1, -1, chains, -1).clone()
        slack = limits[:, :, None] - z @ a.transpose(-1, -2)
        for sweep in range(sweeps):
            for axis in range(d):
                component = a[:, :, None, :, axis]
                divisor = torch.where(component.abs() > 1e-14, component, torch.ones_like(component))
                ratios = slack / divisor
                low = torch.where(component < -1e-14, ratios, -torch.inf).amax(-1) + z[..., axis]
                high = torch.where(component > 1e-14, ratios, torch.inf).amin(-1) + z[..., axis]
                (_, sample) = _conditional_interval(low, high, uniforms[sweep, axis][None])
                delta = sample - z[..., axis]
                z[..., axis] = sample
                slack -= delta[..., None] * component
        physical = z * block_s[:, :, None]
        mean = physical.mean(2)
        residual = physical - mean[:, :, None]
        covariance = residual.transpose(-1, -2) @ residual / max(chains - 1, 1)
        diagonal = torch.minimum(block_s.square(), determinant[start:stop, None, None].pow(2 / d)) * 1e-05
        covariance = 1.5 * covariance + torch.diag_embed(diagonal)
        factor = torch.linalg.cholesky(covariance)
        origin = 0.999 * (mean - center)
        if not bool(torch.isfinite(factor).all()):
            raise RuntimeError('Nonfinite conditional chart covariance')
        margin = half[:, None] - origin @ normals.transpose(1, 2)
        if not bool(((margin > 0) | ~valid[:, None]).all()):
            raise RuntimeError('Conditional chart origin is not inside the cell')
        origins.append(origin)
        factors.append(factor)
        modes.append(center + origin)
    return {'origin': torch.cat(origins), 'factor': torch.cat(factors), 'mode': torch.cat(modes), 'chains': chains, 'sweeps': sweeps, 'proposal_seed': seed}

def gaussian_voronoi_logmass(symbols, sigma, basis, facets, *, samples=4096, order=32, seed=104729, chunk=1024, proposal=None):
    """Positive, density-adapted spherical radial integral, with no mass floor.

    x = nB + origin + diag(chart_scale) * (r * direction).
    Radial end points come from ALL certified Voronoi inequalities. The exact
    Jacobian and Gaussian density are used, including the origin displacement.
    Log-space quadrature supports tails whose probabilities underflow in FP64.
    """
    (b, s, n) = (basis.double(), sigma.double(), symbols.double())
    d = len(b)
    proposal = proposal or gaussian_cell_proposal(n, s, b, facets)
    origin = proposal['origin']
    factor = proposal.get('factor')
    if factor is None:
        factor = torch.diag_embed(proposal['scale'])
    coefficients = torch.as_tensor(facets, device=b.device, dtype=b.dtype)
    normals = coefficients @ b
    margin = 0.5 * normals.square().sum(-1)[None] - origin @ normals.T
    if not bool((margin > 0).all()):
        raise RuntimeError('Fixed proposal origin left the cell')
    u = sobol_uniforms(samples, d, device=b.device, dtype=b.dtype, seed=seed)
    directions = torch.special.ndtri(u)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    (t, w) = _legendre(order)
    t = torch.as_tensor(t, device=b.device, dtype=b.dtype)
    w = torch.as_tensor(w, device=b.device, dtype=b.dtype)
    log_d = math.log(d)
    log_w = w.log()
    tail_transform = t / (1 - t)
    tail_log_jacobian = 2 * torch.log1p(-t)
    log_weights = (d - 1) * t.log() + log_w + log_d
    log_ball = 0.5 * d * math.log(math.pi) - math.lgamma(1 + 0.5 * d)
    logconstant = log_ball + torch.linalg.slogdet(factor)[1] - s.log().sum(-1) - 0.5 * d * math.log(2 * math.pi)
    center = n @ b + origin
    standardized_center = center / s
    sums = []
    outside_sums = []
    all_radii_valid = torch.ones((), device=b.device, dtype=torch.bool)
    for direction in directions.split(chunk):
        physical = direction[None] @ factor.transpose(1, 2)
        products = physical @ normals.T
        radii = torch.where(products > 0, margin[:, None] / products.clamp_min(1e-300), torch.inf).amin(-1)
        checked_radii = radii.detach()
        all_radii_valid = all_radii_valid & (torch.isfinite(checked_radii) & (checked_radii > 0)).all()
        standardized_ray = physical / s[:, None]
        ray_norm = torch.linalg.vector_norm(standardized_ray, dim=-1)
        unit_ray = standardized_ray / ray_norm[:, :, None]
        projection = (standardized_center[:, None] * unit_ray).sum(-1)
        perpendicular = standardized_center[:, None] - projection[:, :, None] * unit_ray
        perpendicular_sq = perpendicular.square().sum(-1)
        radial_projection = projection[:, :, None] + (radii * ray_norm)[:, :, None] * t[None, None]
        squared_distance = radial_projection.square() + perpendicular_sq[:, :, None]
        logterms = -0.5 * squared_distance + log_weights[None, None] + d * radii.log()[:, :, None]
        sums.append(torch.logsumexp(logterms.flatten(1), -1))
        tail_radius = radii[:, :, None] + tail_transform[None, None] / (1 + radii[:, :, None])
        tail_projection = projection[:, :, None] + tail_radius * ray_norm[:, :, None]
        tail_squared_distance = tail_projection.square() + perpendicular_sq[:, :, None]
        tail_weights = (d - 1) * tail_radius.log() + log_w[None, None] - tail_log_jacobian[None, None] - torch.log1p(radii)[:, :, None] + log_d
        tail_terms = -0.5 * tail_squared_distance + tail_weights
        outside_sums.append(torch.logsumexp(tail_terms.flatten(1), -1))
    if not bool(all_radii_valid):
        raise RuntimeError('Invalid integration radius')
    direct = torch.logsumexp(torch.stack(sums, -1), -1) - math.log(samples) + logconstant
    outside = torch.logsumexp(torch.stack(outside_sums, -1), -1) - math.log(samples) + logconstant
    use_complement = direct > -math.log(2)
    if bool((use_complement & (outside >= 0)).any()):
        raise RuntimeError('Complement integral exceeds total Gaussian mass; refine quadrature')
    safe_outside = torch.where(outside < 0, outside, torch.full_like(outside, -1.0))
    complement = torch.log(-torch.expm1(safe_outside))
    result = torch.where(use_complement, complement, direct)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError('Nonfinite radial integral')
    return (result, {'samples': samples, 'radial_order': order, 'seed': seed, 'max_logmass': float(result.detach().max()), 'proposal': proposal, 'used_complement': use_complement.detach()})

@torch.no_grad()
def gaussian_voronoi_mixture(symbols, sigma, basis, facets, *, samples=262144, seed=15485863, chunk=2048, proposal=None):
    """Independent defensive importance oracle (NOT a training gradient).

    Half the points are exactly uniform in the Voronoi cell: fold a uniform
    fundamental parallelotope by exact CVP. Half come from a mode-centered
    Gaussian. Balance weights include BOTH normalized proposal densities.
    This avoids the vanishing acceptance of a loose enclosing parallelotope.
    """
    if samples < 2 or samples % 2:
        raise ValueError('Mixture sample budget must be positive and even')
    (b, s, n) = (basis.double(), sigma.double(), symbols.double())
    d = len(b)
    proposal = proposal or gaussian_cell_proposal(n, s, b, facets)
    mean = proposal['mode']
    factor = proposal.get('factor')
    if factor is None:
        factor = torch.diag_embed(proposal['scale'])
    center = n @ b
    coefficients = torch.as_tensor(facets, device=b.device, dtype=b.dtype)
    normals = coefficients @ b
    half = 0.5 * normals.square().sum(-1)
    logvolume = torch.linalg.slogdet(b)[1]
    lognormal = -s.log().sum(-1) - 0.5 * d * math.log(2 * math.pi)
    logproposal = -torch.linalg.slogdet(factor)[1] - 0.5 * d * math.log(2 * math.pi)
    u = sobol_uniforms(samples // 2, d, device=b.device, dtype=b.dtype, seed=seed)
    v = sobol_uniforms(samples // 2, d, device=b.device, dtype=b.dtype, seed=seed + 7919)
    sums = []
    accepted = torch.zeros(len(n), device=b.device, dtype=torch.long)
    for (uniform, gaussian) in zip(u.split(chunk), v.split(chunk)):
        raw = (uniform - 0.5) @ b
        (q, _, _) = nearest_lattice_point(raw, b, facets)
        physical_uniform = center[:, None] + (raw - q)[None]
        physical_gaussian = mean[:, None] + torch.special.ndtri(gaussian)[None] @ factor.transpose(1, 2)
        inside = ((physical_gaussian - center[:, None]) @ normals.T <= half + 1e-11).all(-1)
        accepted += inside.sum(-1)
        for (physical, mask) in [(physical_uniform, torch.ones_like(inside)), (physical_gaussian, inside)]:
            logtarget = -0.5 * (physical / s[:, None]).square().sum(-1) + lognormal[:, None]
            standardized = torch.linalg.solve_triangular(factor, (physical - mean[:, None]).transpose(1, 2), upper=False).transpose(1, 2)
            logq = -0.5 * standardized.square().sum(-1) + logproposal[:, None]
            logmix = torch.logaddexp(torch.full_like(logq, -logvolume - math.log(2)), logq - math.log(2))
            weights = logtarget - logmix
            sums.append(torch.logsumexp(torch.where(mask, weights, -torch.inf), -1))
    result = torch.logsumexp(torch.stack(sums, -1), -1) - math.log(samples)
    return (result, {'gaussian_acceptance': accepted.double() / (samples // 2), 'samples': samples, 'seed': seed, 'training_gradient_supported': False})
