# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
from __future__ import annotations
import torch
SAMPLERS = ('random', 'sobol')

def sample_unit_cube(num_samples: int, dim: int, *, device: torch.device, dtype: torch.dtype, sampler: str='sobol', antithetic: bool=True, seed: int | None=None) -> torch.Tensor:
    """Draw reproducible points in ``[0, 1)^dim``.

    ``num_samples`` is the total count after adding antithetic complements.
    Scrambled Sobol points are generated on CPU and then moved to the target
    device; the transfer is tiny compared with the likelihood integration.
    """
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')
    if dim <= 0:
        raise ValueError('dim must be positive')
    if sampler not in SAMPLERS:
        raise ValueError(f'sampler must be one of {SAMPLERS}, got {sampler!r}')
    base_count = (num_samples + 1) // 2 if antithetic else num_samples
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()).item())
    if sampler == 'sobol':
        engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=int(seed))
        points = engine.draw(base_count, dtype=dtype).to(device=device)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        points = torch.rand((base_count, dim), device=device, dtype=dtype, generator=generator)
    if antithetic:
        points = torch.cat((points, 1.0 - points), dim=0)
    return points[:num_samples]
