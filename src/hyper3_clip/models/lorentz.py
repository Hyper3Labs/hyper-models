from __future__ import annotations

import math

import torch
from torch import Tensor


def lorentz_inner(x: Tensor, y: Tensor) -> Tensor:
    """Compute batched Lorentzian inner product for matching rows."""
    x = x.float()
    y = y.float()
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def pairwise_lorentz_inner(x: Tensor, y: Tensor) -> Tensor:
    """Compute all-pairs Lorentzian inner products."""
    x = x.float()
    y = y.float()
    time = -x[:, :1] @ y[:, :1].T
    space = x[:, 1:] @ y[:, 1:].T
    return time + space


def exp_map0(u: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Exponential map at the origin from tangent space to hyperboloid."""
    u = u.float()
    kappa = kappa.float()
    sqrt_k = torch.sqrt(kappa)
    norm_u = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
    scaled = sqrt_k * norm_u
    clipped_scaled = scaled.clamp_max(math.asinh(2**15))
    time = torch.cosh(clipped_scaled) / sqrt_k
    space = torch.sinh(clipped_scaled) * u / scaled.clamp_min(eps)
    return torch.cat([time, space], dim=-1)


def log_map0(x: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Logarithmic map at the origin from hyperboloid to tangent space.

    Inverts ``exp_map0`` for points on the Lorentz model hyperboloid. Returns
    vectors in the Euclidean tangent space at the origin (no time coordinate).
    """
    x = x.float()
    dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
    kappa = kappa.to(dtype=torch.float32).flatten()

    if x.dim() == 2:
        if kappa.numel() != 1:
            raise ValueError("log_map0 expects scalar kappa for non-product embeddings")
        sqrt_k = torch.sqrt(kappa.reshape(()))
        alpha = torch.acosh((sqrt_k * x[:, 0]).clamp_min(1.0 + dist_eps))
        coef = alpha / torch.sinh(alpha).clamp_min(dist_eps)
        return x[:, 1:] * coef.unsqueeze(-1)

    if x.dim() == 3:
        if kappa.numel() == 1:
            kappa = kappa.expand(x.shape[1])
        if kappa.numel() != x.shape[1]:
            raise ValueError(
                f"Expected {x.shape[1]} curvatures for product space, got {kappa.numel()}"
            )
        sqrt_k = torch.sqrt(kappa).view(1, -1)
        alpha = torch.acosh((sqrt_k * x[..., 0]).clamp_min(1.0 + dist_eps))
        coef = alpha / torch.sinh(alpha).clamp_min(dist_eps)
        return x[..., 1:] * coef.unsqueeze(-1)

    raise ValueError("log_map0 expects [batch, dim + 1] or [batch, factors, dim + 1] tensors")


def pairwise_dist(x: Tensor, y: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Pairwise geodesic distance on the Lorentz model."""
    kappa = kappa.float()
    dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
    prod = (-kappa) * pairwise_lorentz_inner(x, y)
    prod = prod.clamp_min(1.0 + dist_eps)
    return torch.acosh(prod) / torch.sqrt(kappa)


def product_pairwise_dist(
    x: Tensor,
    y: Tensor,
    kappa: Tensor,
    metric: str = "l1",
    eps: float = 1e-8,
) -> Tensor:
    """Pairwise distance in an l1/l2 product of Lorentz factors.

    Inputs have shape ``[batch, factors, dim + 1]``. For ``metric="l1"``, this
    matches the official PHyCLIP implementation's mean distance over factors.
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("product_pairwise_dist expects [batch, factors, dim + 1] tensors")
    if x.shape[1] != y.shape[1] or x.shape[2] != y.shape[2]:
        raise ValueError("Product Lorentz tensors must have matching factor and feature dimensions")
    kappa = _product_kappa(kappa, x.shape[1], x.device).to(dtype=torch.float32)
    dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
    x = x.float()
    y = y.float()
    inner = -x[:, None, :, 0] * y[None, :, :, 0] + torch.einsum(
        "bkd,nkd->bnk", x[..., 1:], y[..., 1:]
    )
    prod = (-kappa.view(1, 1, -1)) * inner
    dist = torch.acosh(prod.clamp_min(1.0 + dist_eps)) / torch.sqrt(kappa).view(1, 1, -1)
    if metric == "l1":
        return dist.mean(dim=-1)
    if metric == "l2":
        return dist.square().mean(dim=-1).sqrt()
    raise ValueError(f"Unsupported product metric {metric!r}; expected 'l1' or 'l2'")


def metric_pairwise_dist(x: Tensor, y: Tensor, kappa: Tensor, product_metric: str = "l1") -> Tensor:
    """Pairwise distance for either a single Lorentz space or a product space."""
    if x.dim() == 3 or y.dim() == 3:
        return product_pairwise_dist(x, y, kappa, metric=product_metric)
    return pairwise_dist(x, y, kappa)


def paired_dist(
    x: Tensor, y: Tensor, kappa: Tensor, product_metric: str = "l1", eps: float = 1e-8
) -> Tensor:
    """Row-wise distance for either a single Lorentz space or a product space."""
    if x.dim() == 3 or y.dim() == 3:
        if x.shape != y.shape:
            raise ValueError("Product paired_dist expects matching tensor shapes")
        kappa = _product_kappa(kappa, x.shape[1], x.device).to(dtype=torch.float32)
        dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
        x = x.float()
        y = y.float()
        inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        prod = (-kappa.view(1, -1)) * inner
        dist = torch.acosh(prod.clamp_min(1.0 + dist_eps)) / torch.sqrt(kappa).view(1, -1)
        if product_metric == "l1":
            return dist.mean(dim=-1)
        if product_metric == "l2":
            return dist.square().mean(dim=-1).sqrt()
        raise ValueError(f"Unsupported product metric {product_metric!r}; expected 'l1' or 'l2'")
    kappa = kappa.float()
    dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
    prod = (-kappa) * lorentz_inner(x, y)
    prod = prod.clamp_min(1.0 + dist_eps)
    return torch.acosh(prod) / torch.sqrt(kappa)


def radial_distance(x: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Geodesic distance from the origin.

    For points on the hyperboloid, the time coordinate satisfies
    ``x0 = cosh(sqrt(kappa) * r) / sqrt(kappa)``, so we can recover the radial
    distance via ``r = arcosh(sqrt(kappa) * x0) / sqrt(kappa)``.
    """
    dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
    x = x.float()
    kappa = kappa.to(dtype=torch.float32).flatten()
    if x.dim() == 2:
        if kappa.numel() != 1:
            raise ValueError("radial_distance expects scalar kappa for non-product embeddings")
        sqrt_k = torch.sqrt(kappa.reshape(()))
        arg = (sqrt_k * x[:, 0]).clamp_min(1.0 + dist_eps)
        return torch.acosh(arg) / sqrt_k
    if x.dim() == 3:
        if kappa.numel() == 1:
            kappa = kappa.expand(x.shape[1])
        if kappa.numel() != x.shape[1]:
            raise ValueError(
                f"Expected {x.shape[1]} curvatures for product space, got {kappa.numel()}"
            )
        sqrt_k = torch.sqrt(kappa).view(1, -1)
        arg = (sqrt_k * x[..., 0]).clamp_min(1.0 + dist_eps)
        dist = torch.acosh(arg) / sqrt_k
        return dist.mean(dim=-1)
    raise ValueError(
        "radial_distance expects [batch, dim + 1] or [batch, factors, dim + 1] tensors"
    )


def metric_similarity(x: Tensor, y: Tensor, kappa: Tensor, product_metric: str = "l1") -> Tensor:
    """Retrieval/classification similarity for single-space and PHyCLIP-style models."""
    if x.dim() == 3 or y.dim() == 3:
        return -product_pairwise_dist(x, y, kappa, metric=product_metric)
    return pairwise_lorentz_inner(x, y)


def half_aperture(
    general: Tensor, kappa: Tensor, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """Cone half-aperture for entailment cone centered at general concept."""
    general = general.float()
    kappa = kappa.float()
    aperture_eps = max(eps, 16.0 * torch.finfo(general.dtype).eps)
    general_norm = torch.linalg.norm(general[:, 1:], dim=-1)
    ratio = (2.0 * min_radius) / (general_norm * torch.sqrt(kappa) + aperture_eps)
    ratio = ratio.clamp(max=1.0 - aperture_eps)
    return torch.asin(ratio)


def oxy_angle(specific: Tensor, general: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Exterior angle between specific point and entailment cone at general point."""
    specific = specific.float()
    general = general.float()
    kappa = kappa.float()
    angle_eps = max(eps, 16.0 * torch.finfo(specific.dtype).eps)
    inner = lorentz_inner(specific, general)
    numerator = specific[:, 0] + kappa * inner * general[:, 0]
    general_norm = torch.linalg.norm(general[:, 1:], dim=-1).clamp_min(angle_eps)
    denom_term = (kappa * inner).pow(2) - 1.0
    denom = general_norm * torch.sqrt(denom_term.clamp_min(angle_eps))
    cosine = (numerator / denom).clamp(min=-1.0 + angle_eps, max=1.0 - angle_eps)
    return torch.acos(cosine)


def pairwise_oxy_angle(
    specific: Tensor, general: Tensor, kappa: Tensor, eps: float = 1e-8
) -> Tensor:
    """All-pairs exterior angle between specific points and entailment cones at general points."""
    specific = specific.float()
    general = general.float()
    kappa = kappa.to(dtype=torch.float32).flatten()
    if kappa.numel() != 1:
        raise ValueError("pairwise_oxy_angle expects scalar kappa for non-product embeddings")
    kappa_scalar = kappa.reshape(())
    angle_eps = max(eps, 16.0 * torch.finfo(specific.dtype).eps)
    inner = -specific[:, None, 0] * general[None, :, 0] + torch.einsum(
        "nd,md->nm", specific[:, 1:], general[:, 1:]
    )
    numerator = specific[:, None, 0] + kappa_scalar * inner * general[None, :, 0]
    general_norm = torch.linalg.norm(general[:, 1:], dim=-1).clamp_min(angle_eps)
    denom_term = (kappa_scalar * inner).pow(2) - 1.0
    denom = general_norm[None, :] * torch.sqrt(denom_term.clamp_min(angle_eps))
    cosine = (numerator / denom).clamp(min=-1.0 + angle_eps, max=1.0 - angle_eps)
    return torch.acos(cosine)


def product_pairwise_oxy_angle(
    specific: Tensor,
    general: Tensor,
    kappa: Tensor,
    metric: str = "l1",
    eps: float = 1e-8,
) -> Tensor:
    """All-pairs exterior angle in an l1/l2 product of Lorentz factors."""
    if specific.dim() != 3 or general.dim() != 3:
        raise ValueError("product_pairwise_oxy_angle expects [batch, factors, dim + 1] tensors")
    if specific.shape[1] != general.shape[1] or specific.shape[2] != general.shape[2]:
        raise ValueError("Product Lorentz tensors must have matching factor and feature dimensions")
    kappa = _product_kappa(kappa, specific.shape[1], specific.device).to(dtype=torch.float32)
    angle_eps = max(eps, 16.0 * torch.finfo(specific.dtype).eps)
    specific = specific.float()
    general = general.float()
    inner = -specific[:, None, :, 0] * general[None, :, :, 0] + torch.einsum(
        "nkd,mkd->nmk",
        specific[..., 1:],
        general[..., 1:],
    )
    numerator = specific[:, None, :, 0] + (kappa.view(1, 1, -1) * inner) * general[None, :, :, 0]
    general_norm = torch.linalg.norm(general[..., 1:], dim=-1).clamp_min(angle_eps)
    denom_term = (kappa.view(1, 1, -1) * inner).pow(2) - 1.0
    denom = general_norm[None, :, :] * torch.sqrt(denom_term.clamp_min(angle_eps))
    cosine = (numerator / denom).clamp(min=-1.0 + angle_eps, max=1.0 - angle_eps)
    angles = torch.acos(cosine)
    if metric == "l1":
        return angles.mean(dim=-1)
    if metric == "l2":
        return angles.square().mean(dim=-1).sqrt()
    raise ValueError(f"Unsupported product metric {metric!r}; expected 'l1' or 'l2'")


def metric_pairwise_oxy_angle(
    specific: Tensor, general: Tensor, kappa: Tensor, product_metric: str = "l1"
) -> Tensor:
    """All-pairs oxy-angle for either a single Lorentz space or a product space."""
    if specific.dim() == 3 or general.dim() == 3:
        return product_pairwise_oxy_angle(specific, general, kappa, metric=product_metric)
    return pairwise_oxy_angle(specific, general, kappa)


def _product_kappa(kappa: Tensor, num_factors: int, device: torch.device) -> Tensor:
    kappa = kappa.to(device=device, dtype=torch.float32).flatten()
    if kappa.numel() == 1:
        return kappa.expand(num_factors)
    if kappa.numel() != num_factors:
        raise ValueError(
            f"Expected {num_factors} curvatures for product space, got {kappa.numel()}"
        )
    return kappa
