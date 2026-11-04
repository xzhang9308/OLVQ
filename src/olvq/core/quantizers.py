# Copyright (c) 2023, Xi Zhang and Xiaolin Wu. BSD-3-Clause.
import math
import torch
import torch.nn as nn

def _closest_d8_point(t: torch.Tensor) -> torch.Tensor:
    """Closest point of D8 (integer vectors with even coordinate sum)."""
    r = torch.round(t)
    diff = t - r
    j = diff.abs().argmax(dim=1, keepdim=True)
    step = torch.where(diff.gather(1, j) >= 0, 1.0, -1.0).to(t.dtype)
    r_alt = r.clone()
    r_alt.scatter_add_(1, j, step)
    odd = (r.sum(dim=1).remainder(2.0) != 0).unsqueeze(1)
    return torch.where(odd, r_alt, r)

def _closest_e8_point(t: torch.Tensor) -> torch.Tensor:
    """Exact closest point of the E8 lattice (Conway & Sloane).

    E8 = D8 union (D8 + 1/2): decode both cosets and keep the closer point.
    """
    a = _closest_d8_point(t)
    b = _closest_d8_point(t - 0.5) + 0.5
    da = ((t - a) ** 2).sum(dim=1, keepdim=True)
    db = ((t - b) ** 2).sum(dim=1, keepdim=True)
    return torch.where(da <= db, a, b)

def det_normalized_linear(matrix: torch.Tensor) -> torch.Tensor:
    """Rescale an 8x8 map to ``|det| = 1`` without an SVD.

    SVD projection is not differentiable at the identity (repeated singular
    values), so RD training uses this path.  Singular-value clamps stay in
    :func:`volume_preserving_linear` and are applied without gradients.
    """
    if matrix.dim() == 2:
        stacked = matrix.unsqueeze(0)
        squeeze = True
    elif matrix.dim() == 3:
        stacked = matrix
        squeeze = False
    else:
        raise ValueError('matrix must have shape [8, 8] or [G, 8, 8]')
    identity = torch.eye(stacked.shape[-1], device=stacked.device, dtype=stacked.dtype)
    (sign, logabsdet) = torch.linalg.slogdet(stacked)
    invalid = (sign == 0) | ~torch.isfinite(logabsdet)
    scale = torch.exp(-logabsdet.clamp(-30.0, 30.0) / stacked.shape[-1])
    normalized = stacked * scale[:, None, None]
    if bool(invalid.any()):
        normalized = torch.where(invalid[:, None, None], identity.expand_as(normalized), normalized)
    return normalized[0] if squeeze else normalized

def volume_preserving_linear(matrix: torch.Tensor, sigma_min: float=0.25, sigma_max: float=4.0) -> torch.Tensor:
    """Project an 8x8 map onto det = 1 with clamped singular values."""
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    if matrix.dim() == 2:
        if not torch.isfinite(matrix).all():
            return identity
        stacked = matrix.unsqueeze(0)
        squeeze = True
    else:
        stacked = matrix.clone()
        squeeze = False
        finite = torch.isfinite(stacked).all(dim=(-2, -1))
        if not bool(finite.all()):
            stacked = torch.where(finite[:, None, None], stacked, identity.expand_as(stacked))
    (u, singular, vh) = torch.linalg.svd(stacked, full_matrices=False)
    singular = singular.clamp(min=float(sigma_min), max=float(sigma_max))
    log_scale = torch.log(singular).mean(dim=-1, keepdim=True)
    singular = torch.exp(torch.log(singular) - log_scale)
    projected = u * singular.unsqueeze(-2) @ vh
    return projected[0] if squeeze else projected

def covariance_shape_matrix(covariance: torch.Tensor, shrinkage: float=0.05, eps: float=0.0001) -> torch.Tensor:
    """Volume-preserving SPD factor ``M`` such that ``y ≈ p @ M``.

    ``covariance`` is ``[8, 8]`` or ``[G, 8, 8]``.  The Cholesky factor is
    rescaled so ``det(M) = 1``; isotropic volume stays with the E8 scale.
    """
    if covariance.dim() == 2:
        covariance = covariance.unsqueeze(0)
        squeeze = True
    elif covariance.dim() == 3:
        squeeze = False
    else:
        raise ValueError('covariance must be [8, 8] or [G, 8, 8]')
    dim = covariance.shape[-1]
    eye = torch.eye(dim, device=covariance.device, dtype=covariance.dtype)
    mean_var = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(eps)
    isotropic = mean_var[:, None, None] * eye
    shrunk = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * isotropic
    shrunk = 0.5 * (shrunk + shrunk.transpose(-1, -2))
    jitter = float(eps) * mean_var[:, None, None] * eye
    factor = torch.linalg.cholesky(shrunk + jitter)
    log_det = torch.log(factor.diagonal(dim1=-2, dim2=-1).clamp_min(eps)).sum(dim=-1)
    factor = factor * torch.exp(-log_det / dim)[:, None, None]
    shape = factor.transpose(-1, -2)
    return shape[0] if squeeze else shape

class E8LatticeQuantizer(nn.Module):
    """Scaled, rotated, and optionally affinely shaped E8 lattice.

    Lattice points are ``m @ B_E8 * scale @ R @ S`` with the fixed E8 basis, a
    scalar ``scale``, a Cayley rotation ``R``, and a volume-preserving shape
    ``S`` (``det S = 1``).  Closest-point search stays exact Conway-Sloane in
    the canonical coordinates ``t = x @ S^{-1} @ R^T / scale``.  Per-group
    ``S`` changes cell shape without changing cell volume; isotropic gain stays
    in ``log_scale``.
    """
    _E8_BASIS = [[2, 0, 0, 0, 0, 0, 0, 0], [-1, 1, 0, 0, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0, 0, 0], [0, 0, -1, 1, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [0, 0, 0, 0, -1, 1, 0, 0], [0, 0, 0, 0, 0, -1, 1, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]

    def __init__(self, dim: int=8, init_codebook_size: int | None=None, init_scale: float=1.0, learn_scale: bool=True, num_groups: int=1, per_group: bool=False, num_shape_groups: int | None=None):
        super().__init__()
        if int(dim) != 8:
            raise ValueError('E8LatticeQuantizer requires dim == 8')
        self.dim = 8
        self.num_groups = int(num_groups)
        self.per_group = bool(per_group)
        if self.num_groups <= 0:
            raise ValueError('num_groups must be positive')
        if not self.per_group and self.num_groups != 1:
            raise ValueError('num_groups must be 1 unless per_group=True')
        if num_shape_groups is None:
            num_shape_groups = self.num_groups
        self.num_shape_groups = int(num_shape_groups)
        if self.num_shape_groups <= 0:
            raise ValueError('num_shape_groups must be positive')
        if init_codebook_size is not None and init_codebook_size > 1:
            root = float(init_codebook_size) ** (1.0 / 8)
            init_scale = 1.0 / max(root - 1.0, 1e-06)
        if init_scale <= 0:
            raise ValueError('init_scale must be positive')
        basis = torch.tensor(self._E8_BASIS, dtype=torch.float32)
        self.register_buffer('basis', basis)
        self.register_buffer('basis_inv', torch.linalg.inv(basis))
        parameter_shape = (self.num_groups,) if self.per_group else ()
        log_scale = torch.full(parameter_shape, math.log(float(init_scale)), dtype=torch.float32)
        if learn_scale:
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer('log_scale', log_scale)
        skew_shape = (self.num_groups, 8, 8) if self.per_group else (8, 8)
        self.skew_raw = nn.Parameter(torch.zeros(skew_shape))
        if self.num_shape_groups == 1:
            shape = torch.eye(8, dtype=torch.float32)
        else:
            shape = torch.eye(8, dtype=torch.float32).expand(self.num_shape_groups, -1, -1).clone()
        self.shape_matrix = nn.Parameter(shape)
        self.shape_matrix.requires_grad_(False)
        axis_shape = (self.num_shape_groups, 8) if self.num_shape_groups > 1 else (8,)
        self.axis_log_scale = nn.Parameter(torch.zeros(axis_shape, dtype=torch.float32))
        self.axis_log_scale.requires_grad_(False)

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    @property
    def rotation(self) -> torch.Tensor:
        """Orthogonal matrix via the Cayley map R = (I + A)^{-1} (I - A)."""
        a = self.skew_raw - self.skew_raw.transpose(-1, -2)
        eye = torch.eye(8, device=a.device, dtype=a.dtype)
        if self.per_group:
            eye = eye.expand(self.num_groups, -1, -1)
        return torch.linalg.solve(eye + a, eye - a)

    def axis_scale(self, *, device=None, dtype=None) -> torch.Tensor:
        """Volume-preserving per-axis stretch, ``prod(scale)=1`` along the last dim."""
        raw = self.axis_log_scale
        if device is not None or dtype is not None:
            raw = raw.to(device=device if device is not None else raw.device, dtype=dtype if dtype is not None else raw.dtype)
        centered = raw - raw.mean(dim=-1, keepdim=True)
        scale = torch.exp(centered.clamp(-2.0, 2.0))
        return scale / scale.prod(dim=-1, keepdim=True).clamp_min(1e-12).pow(1.0 / 8.0)

    def effective_shape(self, *, device=None, dtype=None) -> torch.Tensor:
        rotation = self.rotation
        shape = self.shape_matrix
        axis = torch.diag_embed(self.axis_scale(device=device, dtype=dtype))
        if device is not None or dtype is not None:
            rotation = rotation.to(device=device if device is not None else rotation.device, dtype=dtype if dtype is not None else rotation.dtype)
            shape = shape.to(device=device if device is not None else shape.device, dtype=dtype if dtype is not None else shape.dtype)
        shape = det_normalized_linear(shape)
        return rotation @ axis @ shape

    def generator_matrix(self, *, device=None, dtype=None) -> torch.Tensor:
        basis = self.basis
        scale = self.scale
        shape = self.effective_shape(device=device, dtype=dtype)
        if device is not None or dtype is not None:
            basis = basis.to(device=device if device is not None else basis.device, dtype=dtype if dtype is not None else basis.dtype)
            scale = scale.to(device=device if device is not None else scale.device, dtype=dtype if dtype is not None else scale.dtype)
        generated = basis @ shape
        if scale.ndim == 0:
            return generated * scale
        return generated * scale[:, None, None]

    @property
    def G(self) -> torch.Tensor:
        """Effective generator matrix (rows span the lattice)."""
        return self.generator_matrix()

    def set_shape_matrix(self, shape: torch.Tensor, *, project: bool=True) -> None:
        shape = torch.as_tensor(shape, dtype=self.shape_matrix.dtype)
        if self.num_shape_groups == 1:
            if shape.shape != (8, 8):
                raise ValueError('shape_matrix must have shape [8, 8]')
        elif tuple(shape.shape) != (self.num_shape_groups, 8, 8):
            raise ValueError(f'shape_matrix must have shape [{self.num_shape_groups}, 8, 8]')
        if project:
            shape = volume_preserving_linear(shape)
        with torch.no_grad():
            self.shape_matrix.copy_(shape)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'shape_matrix'
        if key not in state_dict:
            if self.num_shape_groups == 1:
                identity = torch.eye(8, dtype=self.shape_matrix.dtype)
            else:
                identity = torch.eye(8, dtype=self.shape_matrix.dtype).expand(self.num_shape_groups, -1, -1).clone()
            state_dict[key] = identity
        axis_key = prefix + 'axis_log_scale'
        if axis_key not in state_dict:
            state_dict[axis_key] = self.axis_log_scale.detach().new_zeros(self.axis_log_scale.shape)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _group_ids(self, batch: int, device) -> torch.Tensor:
        groups = max(self.num_groups, self.num_shape_groups)
        return torch.arange(batch, device=device) % groups

    def forward(self, x: torch.Tensor):
        if x.dim() != 2:
            raise ValueError(f'x must have shape [N, dim], got {tuple(x.shape)}')
        if x.shape[1] != self.dim:
            raise ValueError(f'Expected x.shape[1] == {self.dim}, got {x.shape[1]}')
        transform = self.effective_shape(device=x.device, dtype=x.dtype)
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        grouped = transform.dim() == 3 or scale.ndim == 1
        if grouped:
            group_ids = self._group_ids(x.shape[0], x.device)
            if transform.dim() == 3:
                transform = transform[group_ids]
            if scale.ndim == 1:
                scale = scale[group_ids]
            else:
                scale = scale.expand(x.shape[0])
        with torch.no_grad():
            if grouped:
                canonical = torch.linalg.solve(transform.transpose(-1, -2), x.detach().unsqueeze(-1)).squeeze(-1) / scale[:, None]
            else:
                canonical = torch.linalg.solve(transform.T, x.detach().T).T / scale
            lattice_point = _closest_e8_point(canonical)
            idx = torch.round(lattice_point @ self.basis_inv.to(dtype=x.dtype))
        scaled = lattice_point * (scale[:, None] if grouped else scale)
        if grouped:
            y_hat = torch.bmm(scaled.unsqueeze(1), transform).squeeze(1)
        else:
            y_hat = scaled @ transform
        return (y_hat, idx)
