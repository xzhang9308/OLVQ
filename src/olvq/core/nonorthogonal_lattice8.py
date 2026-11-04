# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
from functools import lru_cache
import itertools
import math
import numpy as np
import torch
from torch import nn
from olvq.core.gaussian_rectangle import gaussian_rectangle_logmass, sobol_uniforms, _conditional_interval

def _checked_basis(basis):
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != b.shape[1] or (not 1 <= len(b) <= 8):
        raise ValueError('Expected a square basis of dimension 1..8')
    if not np.isfinite(b).all() or np.linalg.cond(b) > 10:
        raise ValueError('Nonfinite or ill-conditioned basis (condition > 10)')
    return b

def sphere_points(basis, target, radius, max_nodes=2000000):
    """Enumerate ALL integer coefficients in a closed Euclidean sphere.

    Independent CPU reference, with an explicit failure on resource exhaustion.
    QR turns the row-basis norm into a triangular branch-and-bound problem.
    """
    b = _checked_basis(basis)
    (q, r) = np.linalg.qr(b.T)
    y = q.T @ np.asarray(target, dtype=np.float64)
    (d, coefficients, found, nodes) = (len(b), np.zeros(len(b), dtype=np.int64), [], 0)
    tolerance = 1e-11 * max(1.0, radius * radius)

    def visit(k, remaining):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError('Sphere enumeration exceeded its certified node budget')
        if k < 0:
            found.append(coefficients.copy())
            return
        center = (y[k] - r[k, k + 1:] @ coefficients[k + 1:]) / r[k, k]
        width = math.sqrt(max(0.0, remaining + tolerance)) / abs(r[k, k])
        (lo, hi) = (math.ceil(center - width), math.floor(center + width))
        for value in range(lo, hi + 1):
            coefficients[k] = value
            cost = (r[k, k] * (value - center)) ** 2
            if cost <= remaining + tolerance:
                visit(k - 1, remaining - cost)
    visit(d - 1, float(radius) ** 2)
    return (np.asarray(found, dtype=np.int64).reshape(-1, d), nodes)

def reference_cvp(target, basis):
    b = _checked_basis(basis)
    seed = np.rint(np.asarray(target) @ np.linalg.inv(b))
    radius = np.linalg.norm(seed @ b - target)
    (coefficients, _) = sphere_points(b, target, radius)
    if not len(coefficients):
        raise RuntimeError('Empty certified nearest-point search')
    distances = ((coefficients @ b - target) ** 2).sum(-1)
    return coefficients[distances.argmin()]

def certified_facets(basis):
    """Return integer Voronoi normals and an enumeration certificate.

    Each nonzero parity coset has a representative in {-1,0,1}^d. The largest
    of their minimum norms bounds ALL coset minima. Enumerate that entire
    sphere, then keep shortest vectors per coset. At ties all tied vectors
    are retained: redundant constraints are harmless, omitted facets are not.
    Exact orthogonal bases use just their 2d provably sufficient basis normals.
    Reference: Micciancio & Voulgaris, Voronoi algorithm, Theorem 2.3.
    """
    b = _checked_basis(basis.detach().double().cpu().numpy() if isinstance(basis, torch.Tensor) else basis)
    d = len(b)
    gram = b @ b.T
    off = gram - np.diag(np.diag(gram))
    if np.max(np.abs(off)) <= 8 * np.finfo(np.float64).eps * np.max(np.diag(gram)):
        coefficients = np.concatenate((np.eye(d, dtype=np.int64), -np.eye(d, dtype=np.int64)))
        return (coefficients, {'dimension': d, 'constraints': len(coefficients), 'enumeration_nodes': 0, 'orthogonal_exact': True})
    powers = 1 << np.arange(d)
    initial = np.asarray(list(itertools.product((-1, 0, 1), repeat=d)), dtype=np.int64)
    parity = initial % 2 @ powers
    lengths = ((initial @ b) ** 2).sum(-1)
    bounds = np.full(2 ** d, np.inf)
    np.minimum.at(bounds, parity, lengths)
    radius = math.sqrt(bounds[1:].max())
    (candidates, nodes) = sphere_points(b, np.zeros(d), radius)
    lengths = ((candidates @ b) ** 2).sum(-1)
    parity = candidates % 2 @ powers
    minima = np.full(2 ** d, np.inf)
    np.minimum.at(minima, parity, lengths)
    if not np.isfinite(minima).all():
        raise RuntimeError('Enumeration missed a parity coset')
    keep = (parity != 0) & (lengths <= minima[parity] + 1e-10 * max(1.0, radius ** 2))
    coefficients = candidates[keep]
    return (coefficients, {'dimension': d, 'constraints': len(coefficients), 'enumeration_nodes': nodes, 'radius': radius, 'condition': float(np.linalg.cond(b)), 'orthogonal_exact': False})

def nearest_lattice_point(values, basis, facet_coefficients, max_steps=64):
    """GPU-capable Voronoi descent; a facet-membership certificate terminates it.

    Inverse-rounding supplies only the initial point. Each violated inequality
    supplies a strictly closer point. Exhausting max_steps raises an error.
    Integer decisions are detached; reconstruction retains the basis derivative.
    """
    with torch.no_grad():
        (b, x) = (basis.detach().double(), values.detach().double())
        normals = torch.as_tensor(facet_coefficients, device=x.device, dtype=x.dtype)
        vectors = normals @ b
        lengths = vectors.square().sum(-1)
        symbols = torch.round(x @ torch.linalg.inv(b))
        tolerance = 2e-11 * lengths.max().clamp_min(1.0)
        for step in range(max_steps + 1):
            error = x - symbols @ b
            violations = 2 * error @ vectors.T - lengths
            (gain, choice) = violations.max(-1)
            active = gain > tolerance
            if not bool(active.any()):
                break
            if step == max_steps:
                raise RuntimeError('Voronoi descent did not certify the nearest point')
            symbols = symbols + normals[choice] * active[..., None]
    symbols = symbols.to(basis.dtype)
    return (symbols @ basis, symbols, {'descent_steps': step, 'max_facet_violation': float(gain.max().cpu())})

class NonorthogonalBasis8(nn.Module):
    """Full 8D basis with separate volume and volume-preserving shape.

    exp(A-tr(A)/8 I) has determinant one and includes shear and rotation.
    Shape is initialized by the SAME fixed small dense perturbation per group;
    seed is shared across models. Bounds are checked, not projected silently.
    """

    def __init__(self, groups=1, perturbation=0.03, seed=42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        shape = torch.randn(groups, 8, 8, generator=generator) * perturbation
        shape.diagonal(dim1=-2, dim2=-1).zero_()
        self.shape = nn.Parameter(shape)
        self.log_step = nn.Parameter(torch.zeros(groups))

    def forward(self):
        trace = self.shape.diagonal(dim1=-2, dim2=-1).mean(-1)
        eye = torch.eye(8, device=self.shape.device, dtype=self.shape.dtype)
        shape = self.shape - trace[:, None, None] * eye
        return self.log_step.exp()[:, None, None] * torch.matrix_exp(shape)

@lru_cache(maxsize=16)
def _legendre(order):
    (nodes, weights) = np.polynomial.legendre.leggauss(order)
    return ((nodes + 1) / 2, weights / 2)

def cube_face_rays(dimension, samples_per_face, seed, device=None):
    """Balanced cube-surface QMC; opposite faces are antithetic pairs."""
    if dimension < 2:
        raise ValueError('Radial estimator requires dimension >= 2')
    points = sobol_uniforms(samples_per_face, dimension - 1, dtype=torch.float64, device=device, seed=seed, antithetic=False) - 0.5
    faces = []
    for axis in range(dimension):
        face = torch.empty(samples_per_face, dimension, device=device, dtype=torch.float64)
        face[:, axis] = 0.5
        face[:, [i for i in range(dimension) if i != axis]] = points
        faces.extend((face, -face))
    return torch.cat(faces)

def prepare_radial_cell(basis, facet_coefficients, samples_per_face=128, seed=104729):
    """Differentiable ray lengths to the actual Voronoi boundary.

    x=z@A maps [-1/2,1/2]^d onto the enclosing dual-basis parallelotope.
    All supplied normals come from the COMPLETE certified facet set.
    """
    b = basis.double()
    d = len(b)
    coefficients = torch.as_tensor(facet_coefficients, device=b.device, dtype=b.dtype)
    norms = b.square().sum(-1)
    a = torch.diag(norms) @ torch.linalg.inv(b).T
    rays = cube_face_rays(d, samples_per_face, seed, device=b.device)
    normals = coefficients @ b
    half_norms = 0.5 * normals.square().sum(-1)
    products = rays @ (norms[:, None] * coefficients.T)
    ratio = torch.where(products > 0, half_norms / products.clamp_min(1e-300), torch.inf)
    radius = ratio.amin(-1).clamp_max(1.0)
    if not bool(torch.isfinite(radius).all() & (radius > 0).all()):
        raise RuntimeError('Invalid Voronoi radial boundary')
    return {'basis': b, 'a': a, 'rays': rays @ a, 'radius': radius, 'logdet': torch.linalg.slogdet(a)[1], 'dimension': d, 'samples_per_face': samples_per_face, 'seed': seed}

def prepare_spherical_cell(basis, facet_coefficients, samples=4096, seed=104729):
    """Uniform sphere directions for cells far from a rectangular shape.

    The cube-shell control variate becomes inefficient when its enclosing box
    has vastly more volume than the Voronoi cell. This positive estimator has
    no subtraction and is independent of the choice of lattice basis vectors.
    """
    b = basis.double()
    d = len(b)
    u = sobol_uniforms(samples, d, device=b.device, dtype=b.dtype, seed=seed)
    directions = torch.special.ndtri(u)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    coefficients = torch.as_tensor(facet_coefficients, device=b.device, dtype=b.dtype)
    normals = coefficients @ b
    products = directions @ normals.T
    half_norm = 0.5 * normals.square().sum(-1)
    radius = torch.where(products > 0, half_norm / products.clamp_min(1e-300), torch.inf).amin(-1)
    log_ball_volume = 0.5 * d * math.log(math.pi) - math.lgamma(1 + 0.5 * d)
    return {'basis': b, 'rays': directions, 'radius': radius, 'logdet': b.new_tensor(log_ball_volume), 'dimension': d, 'samples_per_face': samples // (2 * d), 'seed': seed}

def _radial_logmass(center, sigma, cell, order, shell=False, ray_chunk=256):
    (d, radius, rays) = (cell['dimension'], cell['radius'], cell['rays'])
    (t, w) = _legendre(order)
    t = torch.as_tensor(t, device=center.device, dtype=center.dtype)
    w = torch.as_tensor(w, device=center.device, dtype=center.dtype)
    log_normalizer = -sigma.log().sum(-1) - 0.5 * d * math.log(2 * math.pi)
    parts = []
    for start in range(0, len(rays), ray_chunk):
        (r, directions) = (radius[start:start + ray_chunk], rays[start:start + ray_chunk])
        if shell:
            width = 1 - r.pow(d)
            radial = (r[:, None].pow(d) + width[:, None] * t).pow(1.0 / d)
            weights = width[:, None] * w
        else:
            radial = r[:, None] * t
            weights = d * r[:, None].pow(d) * t.pow(d - 1) * w
        offsets = radial[..., None] * directions[:, None]
        z = (center[:, None, None, :] + offsets[None]) / sigma[:, None, None, :]
        logpdf = -0.5 * z.square().sum(-1) + log_normalizer[:, None, None]
        shift = logpdf.amax((1, 2)).detach()
        total = ((logpdf - shift[:, None, None]).exp() * weights[None]).sum((1, 2))
        parts.append((total, shift))
    shifts = torch.stack([p[1] for p in parts], -1)
    maximum = shifts.amax(-1).detach()
    total = sum((value * (shift - maximum).exp() for (value, shift) in parts))
    safe = total.clamp_min(torch.finfo(total.dtype).tiny)
    result = safe.log() + maximum + cell['logdet'] - math.log(len(rays))
    return (result, total, maximum + cell['logdet'] - math.log(len(rays)))

def voronoi_logmass(symbols, sigma, basis, facet_coefficients, *, samples_per_face=128, radial_order=16, seed=104729, rectangle_samples=2048, prepared=None, method='auto', positive_domain='cube'):
    """Gaussian mass of nB+V(B), with raw diagnostics and no likelihood floor.

    'control' = enclosing Gaussian rectangle minus its excluded radial shell.
    'positive' = direct positive radial integral, independent of rectangle CDF.
    'auto' uses control only if the excluded mass is <= half the enclosing mass.
    Choosing an estimator does NOT certify its error: callers must independently
    vary angular scramble/sample count AND radial quadrature order before use.
    """
    if method not in ('auto', 'control', 'positive'):
        raise ValueError('Unknown estimator')
    if positive_domain not in ('cube', 'sphere'):
        raise ValueError('Unknown radial integration domain')
    (b, sig, n) = (basis.double(), sigma.double(), symbols.double())
    if n.ndim != 2 or sig.shape != n.shape or n.shape[1] != len(b):
        raise ValueError('symbols and sigma must be [batch,dimension]')
    if not bool(torch.isfinite(sig).all() & (sig > 0).all()):
        raise ValueError('sigma must be positive and finite')
    cell = prepared or prepare_radial_cell(b, facet_coefficients, samples_per_face, seed)
    center = n @ b
    inverse = torch.linalg.inv(cell['a'])
    rectangle_center = center @ inverse
    covariance = inverse.T[None] * sig[:, None, :].square() @ inverse[None]
    chol = torch.linalg.cholesky(covariance)
    uniforms = sobol_uniforms(rectangle_samples, len(b) - 1, device=b.device, dtype=b.dtype, seed=seed + 7919)
    rectangle = gaussian_rectangle_logmass(rectangle_center - 0.5, rectangle_center + 0.5, chol, uniforms)
    positive_cell = prepare_spherical_cell(b, facet_coefficients, 2 * len(b) * samples_per_face, seed=seed + 49999) if positive_domain == 'sphere' else cell
    (positive, _, _) = _radial_logmass(center, sig, positive_cell, radial_order)
    (shell_log, shell_total, shell_scale) = _radial_logmass(center, sig, cell, radial_order, shell=True)
    fraction = shell_total * (shell_scale - rectangle).exp()
    valid = torch.isfinite(fraction) & (fraction >= 0) & (fraction < 1)
    safe_fraction = torch.where(valid, fraction, torch.zeros_like(fraction))
    control = rectangle + torch.log1p(-safe_fraction)
    use_control = valid & (fraction <= 0.5)
    if method == 'control':
        if not bool(valid.all()):
            raise RuntimeError('Shell subtraction invalid: refine estimator, do not clamp')
        result = control
    elif method == 'positive':
        result = positive
    else:
        result = torch.where(use_control, control, positive)
    if not bool(torch.isfinite(result).all()) or bool((result > 1e-08).any()):
        raise RuntimeError('Invalid Gaussian mass; numerical acceptance failed')
    diagnostics = {'logmass_positive': positive.detach(), 'logmass_control': torch.where(valid, control.detach(), torch.full_like(control, float('nan'))), 'logmass_rectangle': rectangle.detach(), 'shell_fraction': fraction.detach(), 'used_control': use_control.detach() if method == 'auto' else torch.full_like(use_control, method == 'control'), 'volume': (cell['logdet'].exp() * cell['radius'].pow(len(b)).mean()).detach(), 'true_volume': torch.linalg.det(b).abs().detach(), 'positive_volume': (positive_cell['logdet'].exp() * positive_cell['radius'].pow(len(b)).mean()).detach(), 'positive_domain': positive_domain, 'samples_per_face': cell['samples_per_face'], 'radial_order': radial_order, 'rectangle_samples': rectangle_samples, 'probability_floor': 0.0}
    return (result, diagnostics)

@torch.no_grad()
def voronoi_logmass_rejection(symbols, sigma, basis, facet_coefficients, *, samples=262144, seed=15485863, chunk=1024, tilt='none'):
    """Independent evaluation oracle: Genz importance proposal + cell membership.

    Samples a Gaussian inside the enclosing parallelotope and rejects points
    outside the true Voronoi cell, retaining the Genz importance weight. This
    integrates the exact event without radial quadrature, shell subtraction or
    uniform angular sampling. It is NOT differentiable and must NOT be silently
    substituted for a training rate gradient. No accepted point is an error.
    """
    if tilt not in ('none', 'mode'):
        raise ValueError('Unknown importance tilt')
    (b, sig, n) = (basis.double(), sigma.double(), symbols.double())
    d = len(b)
    norms = b.square().sum(-1)
    a = torch.diag(norms) @ torch.linalg.inv(b).T
    inverse = torch.linalg.inv(a)
    center = n @ b @ inverse
    covariance = inverse.T[None] * sig[:, None, :].square() @ inverse[None]
    chol = torch.linalg.cholesky(covariance)
    coefficients = torch.as_tensor(facet_coefficients, device=b.device, dtype=b.dtype)
    normals = coefficients @ b
    half_norm = 0.5 * normals.square().sum(-1)
    constraint = norms[:, None] * coefficients.T
    proposal_mean = torch.zeros_like(n)
    if tilt == 'mode':
        from scipy.optimize import minimize, LinearConstraint
        (vv, hh) = (normals.cpu().numpy(), half_norm.cpu().numpy())
        centers = (n @ b).cpu().numpy()
        means = []
        for (cc, ss) in zip(centers, sig.cpu().numpy()):
            constraints = LinearConstraint(vv * ss[None], -np.inf, hh + vv @ cc)
            solution = minimize(lambda z: 0.5 * z @ z, cc / ss, jac=lambda z: z, constraints=constraints, method='SLSQP', options={'ftol': 1e-10, 'maxiter': 500})
            violation = np.max(vv * ss[None] @ solution.x - hh - vv @ cc)
            if not solution.success or violation > 1e-07:
                raise RuntimeError('Gaussian-mode importance tilt failed: ' + solution.message)
            means.append(solution.x * ss)
        proposal_mean = torch.as_tensor(np.asarray(means), device=b.device, dtype=b.dtype)
    proposal_center = proposal_mean @ inverse
    (lower, upper) = (center - proposal_center - 0.5, center - proposal_center + 0.5)
    uniforms = sobol_uniforms(samples, d, seed=seed, device=b.device, dtype=b.dtype, antithetic=True)
    (sums, rectangle_sums, accepted) = ([], [], torch.zeros(len(n), device=b.device, dtype=torch.int64))
    for points in uniforms.split(chunk):
        (standard, coordinates, weights) = ([], [], [])
        for axis in range(d):
            shift = (torch.stack(standard, -1) * chol[:, None, axis, :axis]).sum(-1) if axis else torch.zeros(len(n), len(points), device=b.device, dtype=b.dtype)
            diagonal = chol[:, None, axis, axis]
            (logwidth, z) = _conditional_interval((lower[:, None, axis] - shift) / diagonal, (upper[:, None, axis] - shift) / diagonal, points[None, :, axis])
            standard.append(z)
            coordinates.append(shift + diagonal * z)
            weights.append(logwidth)
        sampled = torch.stack(coordinates, -1) + proposal_center[:, None]
        positions = sampled - center[:, None]
        inside = (positions @ constraint <= half_norm + 1e-12).all(-1)
        accepted += inside.sum(-1)
        logweight = torch.stack(weights, -1).sum(-1)
        if tilt != 'none':
            physical = sampled @ a
            logweight = logweight - (physical * proposal_mean[:, None] / sig[:, None].square()).sum(-1) + 0.5 * (proposal_mean / sig).square().sum(-1)[:, None]
        sums.append(torch.logsumexp(torch.where(inside, logweight, -torch.inf), -1))
        rectangle_sums.append(torch.logsumexp(logweight, -1))
    if bool((accepted == 0).any()):
        raise RuntimeError('Independent importance estimator has zero accepted points; refine')
    result = torch.logsumexp(torch.stack(sums, -1), -1) - math.log(samples)
    return (result, {'accepted_fraction': accepted.double() / samples, 'logmass_rectangle': torch.logsumexp(torch.stack(rectangle_sums, -1), -1) - math.log(samples), 'samples': samples, 'seed': seed, 'tilt': tilt, 'proposal_mean': proposal_mean, 'training_gradient_supported': False})
