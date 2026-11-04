# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import math
import torch
_LOG2PI = math.log(2.0 * math.pi)

def sobol_uniforms(samples, dimension, *, device=None, dtype=torch.float32, seed=104729, antithetic=True):
    """Scrambled Sobol points in the open cube, optionally antithetic.

    ``dimension`` is the number of uniforms, normally d-1. Endpoint protection
    is limited to half a machine epsilon; it never clips a Gaussian tail CDF.
    """
    if samples < 1 or dimension < 1:
        raise ValueError('samples and dimension must be positive')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Gaussian rectangle integration requires float32 or float64')
    half = (samples + 1) // 2 if antithetic else samples
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=int(seed))
    points = engine.draw(half, dtype=dtype).to(device=device)
    if antithetic:
        points = torch.cat((points, 1.0 - points), 0)
    eps = torch.finfo(dtype).eps / 2
    return points[:samples].clamp(eps, 1.0 - eps)

def _log1mexp(x):
    """log(1-exp(x)) for x<0, including values very close to zero."""
    return torch.log(-torch.expm1(x))

class _InverseLogNormalCDF(torch.autograd.Function):
    """ndtri(exp(logp)) without probability underflow or tail-gradient overflow."""

    @staticmethod
    def forward(ctx, logp):
        flip = logp > -math.log(2.0)
        lp = torch.where(flip, _log1mexp(logp), logp)
        direct = torch.special.ndtri(lp.clamp_min(-32.0).exp())
        tail_size = (-2.0 * lp).clamp_min(64.0)
        x = -torch.sqrt(tail_size - tail_size.log() - _LOG2PI)
        for _ in range(2):
            lcdf = torch.special.log_ndtr(x)
            logpdf = -0.5 * x.square() - 0.5 * _LOG2PI
            inv_t2 = x.square().clamp_min(1.0).reciprocal()
            mills = (-x).clamp_min(1.0).reciprocal() * (1 - inv_t2 + 3 * inv_t2.square() - 15 * inv_t2.pow(3) + 105 * inv_t2.pow(4) - 945 * inv_t2.pow(5))
            ratio = torch.where(x < -10.0, mills, (lcdf - logpdf).exp())
            x = x - (lcdf - lp) * ratio
        x = torch.where(lp >= -32.0, direct, x)
        value = torch.where(flip, -x, x)
        ctx.save_for_backward(logp, value)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        (logp, x) = ctx.saved_tensors
        derivative = (logp + 0.5 * x.square() + 0.5 * _LOG2PI).exp()
        inv_t2 = x.square().clamp_min(1.0).reciprocal()
        mills = (-x).clamp_min(1.0).reciprocal() * (1 - inv_t2 + 3 * inv_t2.square() - 15 * inv_t2.pow(3) + 105 * inv_t2.pow(4) - 945 * inv_t2.pow(5))
        derivative = torch.where(x < -10.0, mills, derivative)
        return grad_output * derivative

def inverse_log_normal_cdf(logp):
    """Public, testable log-probability inverse; inputs must be finite and <0."""
    return _InverseLogNormalCDF.apply(logp)

class _LogFactorFloor(torch.autograd.Function):
    """log(LowerBound(exp(logp),floor)), with CompressAI's recovery gradient."""

    @staticmethod
    def forward(ctx, logp, logfloor):
        ctx.save_for_backward(logp)
        ctx.logfloor = logfloor
        return logp.clamp_min(logfloor)

    @staticmethod
    def backward(ctx, grad_output):
        (logp,) = ctx.saved_tensors
        logfloor = ctx.logfloor
        pass_through = (logp >= logfloor) | (grad_output < 0)
        multiplier = (logp - logfloor).clamp_max(0).exp()
        return (grad_output * multiplier * pass_through, None)

def _conditional_interval(a, b, u=None):
    reflected = a + b > 0
    left = torch.where(reflected, -b, a)
    right = torch.where(reflected, -a, b)
    loglo = torch.special.log_ndtr(left)
    loghi = torch.special.log_ndtr(right)
    logwidth = loghi + _log1mexp(loglo - loghi)
    if u is None:
        return (logwidth, None)
    uniform = torch.where(reflected, 1.0 - u, u)
    logp = torch.logaddexp(loglo + torch.log1p(-uniform), loghi + uniform.log())
    z = inverse_log_normal_cdf(logp)
    return (logwidth, torch.where(reflected, -z, z))

def gaussian_rectangle_logmass(lower, upper, chol, uniforms, *, factor_floor=0.0, return_diagnostics=False, return_factors=False, validate_args=False):
    """Estimate log P(lower < X < upper), X~N(0, chol@chol.T).

    Args:
        lower, upper: broadcastable [...,d] finite bounds, strictly ordered.
        chol: broadcastable [...,d,d] lower triangular, positive diagonal.
        uniforms: [samples,d-1] (or [samples,d]) points in the open unit cube.
        factor_floor: optional per-conditional numerical floor, 0 <= floor < 1.
        return_diagnostics / return_factors: return ``(logmass, diagnostics)``;
            factors add effective ``log_factors[...,samples,d]`` to the dict.
        validate_args: check finite input values/triangularity; intended for
            preflight, as tensor value assertions synchronize accelerators.

    Diagnostics tensors are detached except raw_logmass and returned factors.
    No joint probability floor is applied. Input dtype is preserved.
    """
    if lower.dtype not in (torch.float32, torch.float64):
        raise ValueError('Gaussian rectangle integration requires float32 or float64')
    if upper.dtype != lower.dtype or chol.dtype != lower.dtype:
        raise ValueError('bounds and chol must have the same dtype')
    if lower.device != upper.device or lower.device != chol.device:
        raise ValueError('bounds and chol must have the same device')
    d = lower.shape[-1]
    if upper.shape[-1] != d or chol.shape[-2:] != (d, d):
        raise ValueError('inconsistent rectangle/Cholesky dimensions')
    if uniforms.ndim != 2 or uniforms.shape[1] < d - 1 or uniforms.shape[0] < 1:
        raise ValueError('uniforms must be [samples,d-1] or [samples,d]')
    if not 0 <= factor_floor < 1:
        raise ValueError('factor_floor must be in [0,1)')
    shape = torch.broadcast_shapes(lower.shape[:-1], upper.shape[:-1], chol.shape[:-2])
    lo = lower.expand(*shape, d).reshape(-1, d)
    hi = upper.expand(*shape, d).reshape(-1, d)
    factor = chol.expand(*shape, d, d).reshape(-1, d, d)
    u = uniforms.to(device=lower.device, dtype=lower.dtype)
    if validate_args:
        if not bool(torch.isfinite(lo).all() & torch.isfinite(hi).all() & torch.isfinite(factor).all()):
            raise ValueError('bounds and chol must be finite')
        if not bool((hi > lo).all() & (factor.diagonal(dim1=-2, dim2=-1) > 0).all()):
            raise ValueError('upper must exceed lower, with positive Cholesky diagonal')
        if not bool((factor.triu(1) == 0).all()):
            raise ValueError('chol must be lower triangular')
        if not bool(torch.isfinite(u).all() & (u > 0).all() & (u < 1).all()):
            raise ValueError('uniforms must be finite and strictly between zero and one')
    (n, count) = (lo.shape[0], u.shape[0])
    (z, raw_factors, effective_factors) = ([], [], [])
    logfloor = math.log(factor_floor) if factor_floor else None
    for i in range(d):
        if i:
            conditional_mean = (torch.stack(z, dim=-1) * factor[:, None, i, :i]).sum(-1)
        else:
            conditional_mean = lo.new_zeros(n, count)
        diagonal = factor[:, None, i, i]
        a = (lo[:, None, i] - conditional_mean) / diagonal
        b = (hi[:, None, i] - conditional_mean) / diagonal
        (logwidth, sample) = _conditional_interval(a, b, u[None, :, i] if i < d - 1 else None)
        raw_factors.append(logwidth)
        effective_factors.append(_LogFactorFloor.apply(logwidth, logfloor) if logfloor is not None else logwidth)
        if sample is not None:
            z.append(sample)
    raw = torch.stack(raw_factors, dim=-1)
    effective = torch.stack(effective_factors, dim=-1)
    raw_logmass = torch.logsumexp(raw.sum(-1), dim=1) - math.log(count)
    logmass = torch.logsumexp(effective.sum(-1), dim=1) - math.log(count)
    logmass = logmass.reshape(shape)
    if not (return_diagnostics or return_factors):
        return logmass
    diagnostics = {'raw_logmass': raw_logmass.reshape(shape), 'floor_logmass_delta': (logmass - raw_logmass.reshape(shape)).detach(), 'floor_fraction': (raw < logfloor).to(raw.dtype).mean((1, 2)).reshape(shape).detach() if logfloor is not None else logmass.detach().new_zeros(shape), 'min_raw_logfactor': raw.detach().amin((1, 2)).reshape(shape), 'samples': count, 'dimension': d, 'factor_floor': float(factor_floor)}
    if return_factors:
        diagnostics['log_factors'] = effective.reshape(*shape, count, d)
    return (logmass, diagnostics)
