from __future__ import annotations

import torch
from torch import Tensor


def hide_reconstruct_embeddings(
    embeddings: Tensor,
    *,
    variance_threshold: float = 0.9,
    detach_pca: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """HiMo-CLIP HiDe: PCA-reconstruct embeddings using top principal components.

    Given a batch of embeddings ``U ∈ R^{B×D}``, compute mean-centered embeddings,
    perform SVD/PCA, choose the smallest number of components whose cumulative
    explained variance exceeds ``variance_threshold``, and reconstruct each
    embedding from this principal subspace:

        u'_i = P^T (P (u_i - ū)) + ū

    where P stacks the selected principal components as rows.
    """
    if embeddings.ndim != 2:
        raise ValueError("hide_reconstruct_embeddings expects a [batch, dim] tensor")
    if not (0.0 < variance_threshold <= 1.0):
        raise ValueError("variance_threshold must be in (0, 1]")
    if embeddings.size(0) < 2:
        return embeddings

    u = embeddings.to(dtype=torch.float32)
    mean = u.mean(dim=0, keepdim=True)
    centered = u - mean
    if detach_pca:
        centered_for_pca = centered.detach()
    else:
        centered_for_pca = centered

    # SVD: centered = U S Vh, principal components are rows of Vh.
    _, s, vh = torch.linalg.svd(centered_for_pca, full_matrices=False)
    if s.numel() == 0 or float((s.square().sum()).item()) <= eps:
        return embeddings

    explained = s.square()
    cumulative = explained.cumsum(dim=0) / explained.sum().clamp_min(eps)
    m = int((cumulative >= variance_threshold).to(dtype=torch.int64).argmax().item()) + 1
    m = max(1, min(m, vh.size(0)))
    p = vh[:m]
    if detach_pca:
        p = p.detach()

    recon = (centered @ p.T) @ p + mean
    return recon.to(dtype=embeddings.dtype)
