# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
import math
import torch
from olvq.core.voronoi_sampling import sample_unit_cube
from olvq.core.quantizers import _closest_e8_point

def importance_logmass(center, scales, a, s, uniform_noise, uniforms):
    (n, count) = (len(center), len(uniform_noise))
    assert len(uniforms) == count
    with torch.no_grad():
        mu = -torch.bmm(center[:, None, :], a.transpose(1, 2)).squeeze(1) / s[:, None]
        sd = torch.sqrt(torch.bmm(scales.square()[:, None, :], a.square().transpose(1, 2)).squeeze(1)) / s[:, None]
        mu = mu.clamp(-0.95, 0.95)
        sd = sd.clamp(0.08, 2.0)
        lo = torch.special.ndtr((-1 - mu) / sd)
        hi = torch.special.ndtr((1 - mu) / sd)
        width = (hi - lo).clamp_min(1e-12)
        cdf = lo[:, None, :] + uniforms[None, :, :] * width[:, None, :]
        eps = torch.finfo(cdf.dtype).eps
        proposal = mu[:, None, :] + sd[:, None, :] * torch.special.ndtri(cdf.clamp(eps, 1 - eps))
        proposal = proposal.clamp(-1, 1)
        inside = (_closest_e8_point(proposal.reshape(-1, 8)) == 0).all(-1).reshape(n, count)
        samples = torch.cat([uniform_noise[None].expand(n, -1, -1), proposal], 1)
        valid = torch.cat([torch.ones_like(inside), inside], 1)
        logq = (-0.5 * ((samples - mu[:, None]) / sd[:, None]).square() - sd[:, None].log() - 0.5 * math.log(2 * math.pi) - width[:, None].log()).sum(-1)
        logmix = torch.logaddexp(torch.zeros_like(logq), logq) - math.log(2)
    residual = center[:, None] + torch.bmm(samples, a) * s[:, None, None]
    logpdf = (-0.5 * (residual / scales[:, None]).square() - scales[:, None].log() - 0.5 * math.log(2 * math.pi)).sum(-1)
    weighted = (logpdf - logmix).masked_fill(~valid, -torch.inf)
    return 8 * s.log() + torch.linalg.slogdet(a)[1] + torch.logsumexp(weighted, 1) - math.log(2 * count)

def mis_cell_logmass(self, center, scales, a, s, noise):
    count = max(8, len(noise) // 2)
    if self.training:
        seed = int(torch.initial_seed() % (2 ** 31 - 1)) + 7919
    else:
        seed = int(self.voronoi_eval_seed) + 7919
    key = (count, str(center.device), center.dtype, seed)
    if getattr(self, '_mis_uniform_key', None) != key:
        self._mis_uniforms = sample_unit_cube(count, 8, device=center.device, dtype=center.dtype, sampler='sobol', antithetic=True, seed=seed)
        self._mis_uniform_key = key
    u = self._mis_uniforms
    half = count // 2
    cell_noise = torch.cat([noise[:half], noise[len(noise) // 2:len(noise) // 2 + count - half]], 0)
    return importance_logmass(center, scales, a, s, cell_noise, u)
