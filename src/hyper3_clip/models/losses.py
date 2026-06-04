from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from hyper3_clip.models.lorentz import (
    half_aperture,
    metric_pairwise_dist,
    metric_pairwise_oxy_angle,
    oxy_angle,
    paired_dist,
    radial_distance,
)


def contrastive_ce(
    logits: Tensor, targets: Tensor | None = None, weights: Tensor | None = None
) -> Tensor:
    if targets is None:
        targets = torch.arange(logits.size(0), device=logits.device)
    losses = F.cross_entropy(logits, targets, reduction="none")
    return weighted_mean(losses, weights)


def contrastive_sigmoid(
    logits: Tensor,
    targets: Tensor | None = None,
    weights: Tensor | None = None,
    negative_weight: float = 1.0,
) -> Tensor:
    if targets is None:
        targets = torch.arange(logits.size(0), device=logits.device)
    labels = torch.zeros_like(logits)
    labels[torch.arange(logits.size(0), device=logits.device), targets] = 1.0
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if negative_weight != 1.0:
        element_weights = torch.where(
            labels > 0.0, torch.ones_like(labels), logits.new_full((), negative_weight)
        )
        losses = losses * element_weights
    losses = losses.mean(dim=1)
    return weighted_mean(losses, weights)


def contrastive_siglip(
    logits: Tensor,
    targets: Tensor | None = None,
    weights: Tensor | None = None,
    negative_weight: float = 1.0,
) -> Tensor:
    """SigLIP pairwise sigmoid loss (Zhai et al., ICCV 2023).

    Uses labels in {+1, -1} with a per-row sum (not mean) over pairs:
      L_i = sum_j softplus(- y_ij * logit_ij)
    """
    if logits.ndim != 2:
        raise ValueError("contrastive_siglip expects a [batch, classes] logit matrix")
    if targets is None:
        targets = torch.arange(logits.size(0), device=logits.device)
    labels = logits.new_full(logits.shape, -1.0)
    labels[torch.arange(logits.size(0), device=logits.device), targets] = 1.0
    losses = F.softplus(-(labels * logits))
    if negative_weight != 1.0:
        element_weights = torch.where(
            labels > 0.0, torch.ones_like(labels), logits.new_full((), negative_weight)
        )
        losses = losses * element_weights
    row_losses = losses.sum(dim=1)
    return weighted_mean(row_losses, weights)


def weighted_mean(values: Tensor, weights: Tensor | None = None) -> Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).eps)


def gramian_volume_loss(
    vectors: Tensor, weights: Tensor | None = None, eps: float = 1e-4
) -> Tensor:
    """GRAM-style volume loss for sets of vectors.

    ``vectors`` is expected to have shape ``[batch, k, dim]``. Each set of k
    vectors is L2-normalized along ``dim``, then we compute the Gramian
    ``G = V V^T`` and return ``sqrt(det(G + eps I))`` averaged over the batch.
    """
    if vectors.ndim != 3:
        raise ValueError("gramian_volume_loss expects a [batch, k, dim] tensor")
    if eps <= 0.0:
        raise ValueError("gramian_volume_loss eps must be positive")

    vectors = F.normalize(vectors.float(), dim=-1, eps=1e-8)
    gram = vectors @ vectors.transpose(-1, -2)
    k = gram.size(-1)
    gram = gram + eps * torch.eye(k, device=gram.device, dtype=gram.dtype)
    sign, logabsdet = torch.linalg.slogdet(gram)
    volume = torch.exp(0.5 * logabsdet)
    volume = torch.where(sign > 0, volume, volume.new_ones(volume.shape))
    return weighted_mean(volume, weights)


def radius_order_hinge(
    specific: Tensor,
    general: Tensor,
    kappa: Tensor,
    margin: float,
    weights: Tensor | None = None,
) -> Tensor:
    if specific.shape[0] != general.shape[0]:
        raise ValueError("radius_order_hinge expects matching batch dimensions")
    if margin < 0.0:
        raise ValueError("radius_order_hinge margin must be non-negative")
    specific_radius = radial_distance(specific, kappa)
    general_radius = radial_distance(general, kappa)
    losses = F.relu(float(margin) + general_radius - specific_radius)
    return weighted_mean(losses, weights)


def soft_contrastive_ce(
    logits: Tensor, target_weights: Tensor, weights: Tensor | None = None
) -> Tensor:
    if logits.ndim != 2 or target_weights.ndim != 2:
        raise ValueError("soft_contrastive_ce expects [batch, classes] tensors")
    if logits.shape != target_weights.shape:
        raise ValueError(
            "soft_contrastive_ce requires logits and target_weights to have matching shapes"
        )
    log_probs = F.log_softmax(logits, dim=1)
    losses = -(target_weights.to(dtype=log_probs.dtype) * log_probs).sum(dim=1)
    return weighted_mean(losses, weights)


def beta_cal_loss(
    logits: Tensor,
    *,
    targets: Tensor,
    group_ids: Tensor,
    all_group_ids: Tensor,
    beta: float,
    variant: str,
    weights: Tensor | None = None,
) -> Tensor:
    if beta < 0.0:
        raise ValueError("beta_cal_loss beta must be non-negative")
    if variant not in {"ce", "bce"}:
        raise ValueError("beta_cal_loss variant must be 'ce' or 'bce'")
    if logits.ndim != 2:
        raise ValueError("beta_cal_loss expects a [batch, classes] logit matrix")
    if targets.shape != (logits.size(0),):
        raise ValueError("beta_cal_loss targets must have shape [batch]")
    if group_ids.shape != (logits.size(0),):
        raise ValueError("beta_cal_loss group_ids must have shape [batch]")
    if all_group_ids.shape != (logits.size(1),):
        raise ValueError("beta_cal_loss all_group_ids must have shape [classes]")

    same_group = group_ids[:, None] == all_group_ids[None, :]
    same_pair = targets[:, None] == torch.arange(logits.size(1), device=logits.device)[None, :]

    if variant == "ce":
        target_weights = logits.new_zeros(logits.shape)
        target_weights = torch.where(same_pair, logits.new_ones(()), target_weights)
        target_weights = torch.where(
            same_group & ~same_pair, logits.new_full((), float(beta)), target_weights
        )
        target_weights = target_weights / target_weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(target_weights.dtype).eps
        )
        return soft_contrastive_ce(logits, target_weights, weights)

    labels = same_group.to(dtype=logits.dtype)
    element_weights = logits.new_ones(logits.shape)
    element_weights = torch.where(
        same_group & ~same_pair, logits.new_full((), float(beta)), element_weights
    )
    element_losses = (
        F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * element_weights
    )
    row_losses = element_losses.mean(dim=1)
    return weighted_mean(row_losses, weights)


def compositional_contrastive_loss(
    image_feats: Tensor,
    text_feats: Tensor,
    box_image_feats: Tensor,
    box_text_feats: Tensor,
    kappa: Tensor,
    logit_scale: Tensor,
    all_image_feats: Tensor | None = None,
    all_text_feats: Tensor | None = None,
    targets: Tensor | None = None,
) -> Tensor:
    scale = logit_scale.exp().clamp(max=100.0)
    all_image_feats = image_feats if all_image_feats is None else all_image_feats
    all_text_feats = text_feats if all_text_feats is None else all_text_feats

    logits_i_t = -metric_pairwise_dist(image_feats, all_text_feats, kappa) * scale
    logits_t_i = -metric_pairwise_dist(text_feats, all_image_feats, kappa) * scale
    logits_bi_t = -metric_pairwise_dist(box_image_feats, all_text_feats, kappa) * scale
    logits_bt_i = -metric_pairwise_dist(box_text_feats, all_image_feats, kappa) * scale

    return 0.25 * (
        contrastive_ce(logits_i_t, targets)
        + contrastive_ce(logits_t_i, targets)
        + contrastive_ce(logits_bi_t, targets)
        + contrastive_ce(logits_bt_i, targets)
    )


def multi_part_contrastive_loss(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_feats: Tensor,
    part_text_feats: Tensor,
    part_mask: Tensor,
    kappa: Tensor,
    logit_scale: Tensor,
    all_image_feats: Tensor | None = None,
    all_text_feats: Tensor | None = None,
    targets: Tensor | None = None,
) -> Tensor:
    scale = logit_scale.exp().clamp(max=100.0)
    all_image_feats = image_feats if all_image_feats is None else all_image_feats
    all_text_feats = text_feats if all_text_feats is None else all_text_feats
    if targets is None:
        targets = torch.arange(image_feats.size(0), device=image_feats.device)

    part_image_flat, part_text_flat, part_targets = _flatten_valid_parts(
        part_image_feats, part_text_feats, part_mask, targets
    )

    logits_i_t = -metric_pairwise_dist(image_feats, all_text_feats, kappa) * scale
    logits_t_i = -metric_pairwise_dist(text_feats, all_image_feats, kappa) * scale
    logits_pi_t = -metric_pairwise_dist(part_image_flat, all_text_feats, kappa) * scale
    logits_pt_i = -metric_pairwise_dist(part_text_flat, all_image_feats, kappa) * scale

    return 0.25 * (
        contrastive_ce(logits_i_t, targets)
        + contrastive_ce(logits_t_i, targets)
        + contrastive_ce(logits_pi_t, part_targets)
        + contrastive_ce(logits_pt_i, part_targets)
    )


def packed_part_contrastive_loss(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_feats: Tensor,
    part_text_feats: Tensor,
    part_owner: Tensor,
    kappa: Tensor,
    logit_scale: Tensor,
    all_image_feats: Tensor | None = None,
    all_text_feats: Tensor | None = None,
    targets: Tensor | None = None,
) -> Tensor:
    scale = logit_scale.exp().clamp(max=100.0)
    all_image_feats = image_feats if all_image_feats is None else all_image_feats
    all_text_feats = text_feats if all_text_feats is None else all_text_feats
    if targets is None:
        targets = torch.arange(image_feats.size(0), device=image_feats.device)

    logits_i_t = -metric_pairwise_dist(image_feats, all_text_feats, kappa) * scale
    logits_t_i = -metric_pairwise_dist(text_feats, all_image_feats, kappa) * scale
    global_loss = 0.5 * (contrastive_ce(logits_i_t, targets) + contrastive_ce(logits_t_i, targets))

    if part_image_feats.numel() == 0:
        return global_loss

    part_targets = targets[part_owner]
    logits_pi_t = -metric_pairwise_dist(part_image_feats, all_text_feats, kappa) * scale
    logits_pt_i = -metric_pairwise_dist(part_text_feats, all_image_feats, kappa) * scale
    part_loss = 0.5 * (
        contrastive_ce(logits_pi_t, part_targets) + contrastive_ce(logits_pt_i, part_targets)
    )
    return 0.5 * (global_loss + part_loss)


def factor_oxy_angle(specific: Tensor, general: Tensor, kappa: Tensor) -> Tensor:
    if specific.dim() != 3:
        return oxy_angle(specific=specific, general=general, kappa=kappa)
    batch_size, num_factors, feature_dim = specific.shape
    kappa = _factor_kappa(kappa, num_factors, specific.device)
    factor_kappa = kappa.view(1, num_factors).expand(batch_size, num_factors).reshape(-1)
    return oxy_angle(
        specific=specific.reshape(batch_size * num_factors, feature_dim),
        general=general.reshape(batch_size * num_factors, feature_dim),
        kappa=factor_kappa,
    ).reshape(batch_size, num_factors)


def factor_half_aperture(general: Tensor, kappa: Tensor) -> Tensor:
    if general.dim() != 3:
        return half_aperture(general=general, kappa=kappa)
    batch_size, num_factors, feature_dim = general.shape
    kappa = _factor_kappa(kappa, num_factors, general.device)
    factor_kappa = kappa.view(1, num_factors).expand(batch_size, num_factors).reshape(-1)
    return half_aperture(
        general=general.reshape(batch_size * num_factors, feature_dim),
        kappa=factor_kappa,
    ).reshape(batch_size, num_factors)


def _factor_kappa(kappa: Tensor, num_factors: int, device: torch.device) -> Tensor:
    kappa = kappa.to(device=device, dtype=torch.float32).flatten()
    if kappa.numel() == 1:
        return kappa.expand(num_factors)
    if kappa.numel() != num_factors:
        raise ValueError(
            f"Expected {num_factors} curvatures for product space, got {kappa.numel()}"
        )
    return kappa


def entailment_residual(
    specific: Tensor,
    general: Tensor,
    kappa: Tensor,
    aperture_scale: float,
) -> Tensor:
    angles = factor_oxy_angle(specific=specific, general=general, kappa=kappa)
    apertures = factor_half_aperture(general=general, kappa=kappa)
    return torch.clamp(angles - (aperture_scale * apertures), min=0.0).mean()


def weighted_entailment_residual(
    specific: Tensor,
    general: Tensor,
    kappa: Tensor,
    aperture_scale: float,
    weights: Tensor | None = None,
) -> Tensor:
    angles = factor_oxy_angle(specific=specific, general=general, kappa=kappa)
    apertures = factor_half_aperture(general=general, kappa=kappa)
    residuals = torch.clamp(angles - (aperture_scale * apertures), min=0.0)
    if residuals.dim() == 2:
        residuals = residuals.mean(dim=-1)
    return weighted_mean(residuals, weights)


def compositional_entailment_loss(
    image_feats: Tensor,
    text_feats: Tensor,
    box_image_feats: Tensor,
    box_text_feats: Tensor,
    kappa: Tensor,
    inter_aperture_scale: float,
    intra_aperture_scale: float,
) -> Tensor:
    text_to_image = entailment_residual(
        specific=image_feats,
        general=text_feats,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
    )
    box_text_to_box_image = entailment_residual(
        specific=box_image_feats,
        general=box_text_feats,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
    )
    box_image_to_image = entailment_residual(
        specific=image_feats,
        general=box_image_feats,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
    )
    box_text_to_text = entailment_residual(
        specific=text_feats,
        general=box_text_feats,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
    )

    return 0.5 * (text_to_image + box_text_to_box_image + box_image_to_image + box_text_to_text)


def multi_part_entailment_loss(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_feats: Tensor,
    part_text_feats: Tensor,
    part_mask: Tensor,
    kappa: Tensor,
    inter_aperture_scale: float,
    intra_aperture_scale: float,
) -> Tensor:
    part_image_flat = part_image_feats[part_mask]
    part_text_flat = part_text_feats[part_mask]
    image_for_parts = image_feats[:, None, :].expand_as(part_image_feats)[part_mask]
    text_for_parts = text_feats[:, None, :].expand_as(part_text_feats)[part_mask]

    text_to_image = entailment_residual(
        specific=image_feats,
        general=text_feats,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
    )
    part_text_to_part_image = entailment_residual(
        specific=part_image_flat,
        general=part_text_flat,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
    )
    part_image_to_image = entailment_residual(
        specific=image_for_parts,
        general=part_image_flat,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
    )
    part_text_to_text = entailment_residual(
        specific=text_for_parts,
        general=part_text_flat,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
    )

    return 0.5 * (text_to_image + part_text_to_part_image + part_image_to_image + part_text_to_text)


def packed_part_entailment_loss(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_feats: Tensor,
    part_text_feats: Tensor,
    part_owner: Tensor,
    kappa: Tensor,
    inter_aperture_scale: float,
    intra_aperture_scale: float,
) -> Tensor:
    text_to_image = entailment_residual(
        specific=image_feats,
        general=text_feats,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
    )
    if part_image_feats.numel() == 0:
        return text_to_image

    image_for_parts = image_feats[part_owner]
    text_for_parts = text_feats[part_owner]
    part_text_to_part_image = entailment_residual(
        specific=part_image_feats,
        general=part_text_feats,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
    )
    part_image_to_image = entailment_residual(
        specific=image_for_parts,
        general=part_image_feats,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
    )
    part_text_to_text = entailment_residual(
        specific=text_for_parts,
        general=part_text_feats,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
    )

    return 0.5 * (text_to_image + part_text_to_part_image + part_image_to_image + part_text_to_text)


def uncha_contrastive_losses(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    kappa: Tensor,
    global_logit_scale: Tensor,
    local_logit_scale: Tensor,
    global_local_logit_scale: Tensor,
    image_euc_feats: Tensor | None = None,
    text_euc_feats: Tensor | None = None,
    part_image_euc_flat: Tensor | None = None,
    part_text_euc_flat: Tensor | None = None,
    image_for_parts_euc: Tensor | None = None,
    text_for_parts_euc: Tensor | None = None,
    all_image_feats: Tensor | None = None,
    all_text_feats: Tensor | None = None,
    all_part_image_feats: Tensor | None = None,
    all_part_text_feats: Tensor | None = None,
    all_image_for_parts: Tensor | None = None,
    all_text_for_parts: Tensor | None = None,
    all_image_euc_feats: Tensor | None = None,
    all_text_euc_feats: Tensor | None = None,
    all_part_image_euc_feats: Tensor | None = None,
    all_part_text_euc_feats: Tensor | None = None,
    all_image_for_parts_euc: Tensor | None = None,
    all_text_for_parts_euc: Tensor | None = None,
    global_targets: Tensor | None = None,
    part_targets: Tensor | None = None,
    part_weights: Tensor | None = None,
    product_metric: str = "l1",
    loss_type: str = "ce",
    contrastive_global_weight: float = 1.0,
    contrastive_local_weight: float = 1.0,
    contrastive_global_local_weight: float = 1.0,
    beta_cal_beta: float = 0.0,
    beta_cal_variant: str = "ce",
    beta_cal_weight: float = 0.0,
    part_group_ids: Tensor | None = None,
    all_part_group_ids: Tensor | None = None,
    global_logit_bias: Tensor | None = None,
    local_logit_bias: Tensor | None = None,
    global_local_logit_bias: Tensor | None = None,
    sigmoid_negative_weight: float = 1.0,
    global_local_mode: str = "repeat",
    global_local_metric: str = "distance",
    global_local_angle_aux_weight: float = 0.0,
    global_local_angle_aux_mode: str = "contrastive",
    global_local_angle_aux_scale: float = 5.5,
    global_local_angle_aux_aperture_scale: float = 1.0,
) -> dict[str, Tensor]:
    if loss_type not in {"ce", "sigmoid", "siglip", "siglip_metric"}:
        raise ValueError(
            f"Unsupported contrastive loss {loss_type!r}; expected 'ce', 'sigmoid', 'siglip', or 'siglip_metric'"
        )
    if global_local_mode not in {"repeat", "inbatch"}:
        raise ValueError("global_local_mode must be 'repeat' or 'inbatch'")
    if global_local_metric not in {"distance", "angle"}:
        raise ValueError("global_local_metric must be 'distance' or 'angle'")
    if global_local_angle_aux_mode not in {"contrastive", "positive_hinge"}:
        raise ValueError("global_local_angle_aux_mode must be 'contrastive' or 'positive_hinge'")
    if global_local_angle_aux_weight < 0.0:
        raise ValueError("global_local_angle_aux_weight must be non-negative")
    if global_local_angle_aux_scale <= 0.0:
        raise ValueError("global_local_angle_aux_scale must be positive")
    if global_local_angle_aux_aperture_scale <= 0.0:
        raise ValueError("global_local_angle_aux_aperture_scale must be positive")
    all_image_feats = image_feats if all_image_feats is None else all_image_feats
    all_text_feats = text_feats if all_text_feats is None else all_text_feats
    all_part_image_feats = part_image_flat if all_part_image_feats is None else all_part_image_feats
    all_part_text_feats = part_text_flat if all_part_text_feats is None else all_part_text_feats
    all_image_for_parts = image_for_parts if all_image_for_parts is None else all_image_for_parts
    all_text_for_parts = text_for_parts if all_text_for_parts is None else all_text_for_parts
    if global_targets is None:
        global_targets = torch.arange(image_feats.size(0), device=image_feats.device)
    if part_targets is None:
        part_targets = torch.arange(part_image_flat.size(0), device=part_image_flat.device)

    global_scale = global_logit_scale.exp().clamp(max=100.0)
    local_scale = local_logit_scale.exp().clamp(max=100.0)
    global_local_scale = global_local_logit_scale.exp().clamp(max=100.0)

    if loss_type == "siglip":
        if image_euc_feats is None or text_euc_feats is None:
            raise ValueError("siglip contrastive requires image_euc_feats and text_euc_feats")
        if image_feats.dim() != 2 or text_feats.dim() != 2:
            raise ValueError("siglip contrastive is only supported for non-product features")
        all_image_euc_feats = (
            image_euc_feats if all_image_euc_feats is None else all_image_euc_feats
        )
        all_text_euc_feats = text_euc_feats if all_text_euc_feats is None else all_text_euc_feats
        zimg = F.normalize(image_euc_feats.float(), dim=-1)
        ztxt = F.normalize(text_euc_feats.float(), dim=-1)
        zimg_all = F.normalize(all_image_euc_feats.float(), dim=-1)
        ztxt_all = F.normalize(all_text_euc_feats.float(), dim=-1)
        image_logits = (zimg @ ztxt_all.T) * global_scale
        text_logits = (ztxt @ zimg_all.T) * global_scale
    else:
        image_logits = (
            -metric_pairwise_dist(image_feats, all_text_feats, kappa, product_metric=product_metric)
            * global_scale
        )
        text_logits = (
            -metric_pairwise_dist(text_feats, all_image_feats, kappa, product_metric=product_metric)
            * global_scale
        )

    if loss_type in {"sigmoid", "siglip", "siglip_metric"}:
        bias = (
            image_logits.new_zeros(())
            if global_logit_bias is None
            else global_logit_bias.to(image_logits.device)
        )
        image_logits = image_logits + bias
        text_logits = text_logits + bias
    global_contrastive = 0.5 * (
        _contrastive_loss(image_logits, global_targets, None, loss_type, sigmoid_negative_weight)
        + _contrastive_loss(text_logits, global_targets, None, loss_type, sigmoid_negative_weight)
    )

    if part_image_flat.numel() == 0:
        zero = image_feats.new_zeros(())
        contrastive = contrastive_global_weight * global_contrastive
        return {
            "contrastive_loss": contrastive,
            "global_contrastive_loss": global_contrastive,
            "local_contrastive_loss": zero,
            "global_local_contrastive_loss": zero,
            "global_local_angle_aux_loss": zero,
            "beta_cal_loss": zero,
        }

    if loss_type == "siglip":
        if part_image_euc_flat is None or part_text_euc_flat is None:
            raise ValueError(
                "siglip contrastive requires part_image_euc_flat and part_text_euc_flat when parts exist"
            )
        all_part_image_euc_feats = (
            part_image_euc_flat if all_part_image_euc_feats is None else all_part_image_euc_feats
        )
        all_part_text_euc_feats = (
            part_text_euc_flat if all_part_text_euc_feats is None else all_part_text_euc_feats
        )
        zpi = F.normalize(part_image_euc_flat.float(), dim=-1)
        zpt = F.normalize(part_text_euc_flat.float(), dim=-1)
        zpi_all = F.normalize(all_part_image_euc_feats.float(), dim=-1)
        zpt_all = F.normalize(all_part_text_euc_feats.float(), dim=-1)
        part_image_logits = (zpi @ zpt_all.T) * local_scale
        part_text_logits = (zpt @ zpi_all.T) * local_scale
    else:
        part_image_logits = (
            -metric_pairwise_dist(
                part_image_flat, all_part_text_feats, kappa, product_metric=product_metric
            )
            * local_scale
        )
        part_text_logits = (
            -metric_pairwise_dist(
                part_text_flat, all_part_image_feats, kappa, product_metric=product_metric
            )
            * local_scale
        )

    if loss_type in {"sigmoid", "siglip", "siglip_metric"}:
        bias = (
            part_image_logits.new_zeros(())
            if local_logit_bias is None
            else local_logit_bias.to(part_image_logits.device)
        )
        part_image_logits = part_image_logits + bias
        part_text_logits = part_text_logits + bias
    local_contrastive = 0.5 * (
        _contrastive_loss(
            part_image_logits, part_targets, part_weights, loss_type, sigmoid_negative_weight
        )
        + _contrastive_loss(
            part_text_logits, part_targets, part_weights, loss_type, sigmoid_negative_weight
        )
    )

    global_local_contrastive = image_feats.new_zeros(())
    global_local_angle_aux = image_feats.new_zeros(())
    if contrastive_global_local_weight != 0.0:
        if global_local_mode == "inbatch":
            if part_group_ids is None:
                raise ValueError(
                    "inbatch global-local contrastive requires part_group_ids to be provided"
                )
            global_local_targets = part_group_ids
            all_text_for_global_local = all_text_feats
            all_image_for_global_local = all_image_feats
            all_text_for_global_local_euc = all_text_euc_feats
            all_image_for_global_local_euc = all_image_euc_feats
        else:
            global_local_targets = part_targets
            all_text_for_global_local = all_text_for_parts
            all_image_for_global_local = all_image_for_parts
            all_text_for_global_local_euc = all_text_for_parts_euc
            all_image_for_global_local_euc = all_image_for_parts_euc

        image_uncertainty = embedding_uncertainty(part_image_flat).detach()
        text_uncertainty = embedding_uncertainty(part_text_flat).detach()
        image_temp = torch.exp(-0.5 * image_uncertainty).clamp(min=0.1, max=10.0)
        text_temp = torch.exp(-0.5 * text_uncertainty).clamp(min=0.1, max=10.0)

        if loss_type == "siglip":
            if part_image_euc_flat is None or part_text_euc_flat is None:
                raise ValueError(
                    "siglip global-local contrastive requires part_image_euc_flat/part_text_euc_flat"
                )
            if all_text_for_global_local_euc is None or all_image_for_global_local_euc is None:
                raise ValueError(
                    "siglip global-local contrastive requires all_image_euc_feats/all_text_euc_feats"
                )
            zpi = F.normalize(part_image_euc_flat.float(), dim=-1)
            zpt = F.normalize(part_text_euc_flat.float(), dim=-1)
            zimg_all = F.normalize(all_image_for_global_local_euc.float(), dim=-1)
            ztxt_all = F.normalize(all_text_for_global_local_euc.float(), dim=-1)
            part_image_to_whole_text = (zpi @ ztxt_all.T) * image_temp[:, None] * global_local_scale
            part_text_to_whole_image = (zpt @ zimg_all.T) * text_temp[:, None] * global_local_scale
        else:
            if global_local_metric == "angle":
                part_image_to_whole_text = -metric_pairwise_oxy_angle(
                    part_image_flat,
                    all_text_for_global_local,
                    kappa,
                    product_metric=product_metric,
                )
                part_text_to_whole_image = -metric_pairwise_oxy_angle(
                    part_text_flat,
                    all_image_for_global_local,
                    kappa,
                    product_metric=product_metric,
                )
            else:
                part_image_to_whole_text = -metric_pairwise_dist(
                    part_image_flat, all_text_for_global_local, kappa, product_metric=product_metric
                )
                part_text_to_whole_image = -metric_pairwise_dist(
                    part_text_flat, all_image_for_global_local, kappa, product_metric=product_metric
                )
            part_image_to_whole_text = (
                part_image_to_whole_text * image_temp[:, None] * global_local_scale
            )
            part_text_to_whole_image = (
                part_text_to_whole_image * text_temp[:, None] * global_local_scale
            )

        if loss_type in {"sigmoid", "siglip", "siglip_metric"}:
            bias = (
                part_image_to_whole_text.new_zeros(())
                if global_local_logit_bias is None
                else global_local_logit_bias.to(part_image_to_whole_text.device)
            )
            part_image_to_whole_text = part_image_to_whole_text + bias
            part_text_to_whole_image = part_text_to_whole_image + bias

        global_local_contrastive = 0.5 * (
            _contrastive_loss(
                part_image_to_whole_text,
                global_local_targets,
                part_weights,
                loss_type,
                sigmoid_negative_weight,
            )
            + _contrastive_loss(
                part_text_to_whole_image,
                global_local_targets,
                part_weights,
                loss_type,
                sigmoid_negative_weight,
            )
        )

        if global_local_angle_aux_weight > 0.0:
            if global_local_angle_aux_mode == "positive_hinge":
                positive_text = all_text_for_global_local.index_select(0, global_local_targets)
                positive_image = all_image_for_global_local.index_select(0, global_local_targets)
                global_local_angle_aux = 0.5 * (
                    weighted_entailment_residual(
                        specific=part_image_flat,
                        general=positive_text,
                        kappa=kappa,
                        aperture_scale=global_local_angle_aux_aperture_scale,
                        weights=part_weights,
                    )
                    + weighted_entailment_residual(
                        specific=part_text_flat,
                        general=positive_image,
                        kappa=kappa,
                        aperture_scale=global_local_angle_aux_aperture_scale,
                        weights=part_weights,
                    )
                )
            elif loss_type != "siglip":
                angle_scale = part_image_flat.new_tensor(float(global_local_angle_aux_scale))
                part_image_to_whole_text_angle = (
                    -metric_pairwise_oxy_angle(
                        part_image_flat,
                        all_text_for_global_local,
                        kappa,
                        product_metric=product_metric,
                    )
                    * image_temp[:, None]
                    * angle_scale
                )
                part_text_to_whole_image_angle = (
                    -metric_pairwise_oxy_angle(
                        part_text_flat,
                        all_image_for_global_local,
                        kappa,
                        product_metric=product_metric,
                    )
                    * text_temp[:, None]
                    * angle_scale
                )
                if loss_type in {"sigmoid", "siglip_metric"}:
                    bias = (
                        part_image_to_whole_text_angle.new_zeros(())
                        if global_local_logit_bias is None
                        else global_local_logit_bias.to(part_image_to_whole_text_angle.device)
                    )
                    part_image_to_whole_text_angle = part_image_to_whole_text_angle + bias
                    part_text_to_whole_image_angle = part_text_to_whole_image_angle + bias
                global_local_angle_aux = 0.5 * (
                    _contrastive_loss(
                        part_image_to_whole_text_angle,
                        global_local_targets,
                        part_weights,
                        loss_type,
                        sigmoid_negative_weight,
                    )
                    + _contrastive_loss(
                        part_text_to_whole_image_angle,
                        global_local_targets,
                        part_weights,
                        loss_type,
                        sigmoid_negative_weight,
                    )
                )

    beta_cal = image_feats.new_zeros(())
    if beta_cal_weight > 0.0 and beta_cal_beta > 0.0:
        if part_group_ids is None or all_part_group_ids is None:
            raise ValueError(
                "beta_cal requires part_group_ids and all_part_group_ids to be provided"
            )
        beta_cal = 0.5 * (
            beta_cal_loss(
                part_image_logits,
                targets=part_targets,
                group_ids=part_group_ids,
                all_group_ids=all_part_group_ids,
                beta=beta_cal_beta,
                variant=beta_cal_variant,
                weights=part_weights,
            )
            + beta_cal_loss(
                part_text_logits,
                targets=part_targets,
                group_ids=part_group_ids,
                all_group_ids=all_part_group_ids,
                beta=beta_cal_beta,
                variant=beta_cal_variant,
                weights=part_weights,
            )
        )

    contrastive = (
        contrastive_global_weight * global_contrastive
        + contrastive_local_weight * local_contrastive
        + contrastive_global_local_weight * global_local_contrastive
        + global_local_angle_aux_weight * global_local_angle_aux
        + beta_cal_weight * beta_cal
    )
    return {
        "contrastive_loss": contrastive,
        "global_contrastive_loss": global_contrastive,
        "local_contrastive_loss": local_contrastive,
        "global_local_contrastive_loss": global_local_contrastive,
        "global_local_angle_aux_loss": global_local_angle_aux,
        "beta_cal_loss": beta_cal,
    }


def _contrastive_loss(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor | None,
    loss_type: str,
    sigmoid_negative_weight: float,
) -> Tensor:
    if loss_type == "ce":
        return contrastive_ce(logits, targets, weights)
    if loss_type == "sigmoid":
        return contrastive_sigmoid(
            logits, targets, weights, negative_weight=sigmoid_negative_weight
        )
    if loss_type in {"siglip", "siglip_metric"}:
        return contrastive_siglip(logits, targets, weights, negative_weight=sigmoid_negative_weight)
    raise ValueError(f"Unsupported contrastive loss {loss_type!r}")


def uncha_entailment_losses(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    kappa: Tensor,
    inter_aperture_scale: float,
    intra_aperture_scale: float,
    piecewise_factor: float = 0.1,
    calibration_alpha: float = 10.0,
    stop_grad_calibration: bool = True,
    geometry: str = "lorentz",
    part_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    text_image = piecewise_entailment_residual(
        specific=image_feats,
        general=text_feats,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
        factor=piecewise_factor,
        geometry=geometry,
    )
    text_image_entailment = 0.5 * text_image.mean()

    if part_image_flat.numel() == 0:
        zero = image_feats.new_zeros(())
        return {
            "entailment_loss": text_image_entailment,
            "text_image_entailment_loss": text_image_entailment,
            "part_text_image_entailment_loss": zero,
            "cross_image_entailment_loss": zero,
            "cross_text_entailment_loss": zero,
            "cross_image_calibration_loss": zero,
            "cross_text_calibration_loss": zero,
        }

    part_text_image = piecewise_entailment_residual(
        specific=part_image_flat,
        general=part_text_flat,
        kappa=kappa,
        aperture_scale=inter_aperture_scale,
        factor=piecewise_factor,
        geometry=geometry,
    )
    cross_image = piecewise_entailment_residual(
        specific=image_for_parts,
        general=part_image_flat,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
        factor=piecewise_factor,
        geometry=geometry,
    )
    cross_text = piecewise_entailment_residual(
        specific=text_for_parts,
        general=part_text_flat,
        kappa=kappa,
        aperture_scale=intra_aperture_scale,
        factor=piecewise_factor,
        geometry=geometry,
    )

    part_text_image_entailment = 0.5 * weighted_mean(part_text_image, part_weights)
    cross_image_entailment, cross_image_calibration = uncertainty_calibrated_entailment_loss(
        cross_image,
        embedding_uncertainty(part_image_flat),
        alpha=calibration_alpha,
        stop_grad=stop_grad_calibration,
        weights=part_weights,
    )
    cross_text_entailment, cross_text_calibration = uncertainty_calibrated_entailment_loss(
        cross_text,
        embedding_uncertainty(part_text_flat),
        alpha=calibration_alpha,
        stop_grad=stop_grad_calibration,
        weights=part_weights,
    )

    entailment = (
        text_image_entailment
        + part_text_image_entailment
        + 0.5 * (cross_image_entailment + cross_text_entailment)
        + cross_image_calibration
        + cross_text_calibration
    )
    return {
        "entailment_loss": entailment,
        "text_image_entailment_loss": text_image_entailment,
        "part_text_image_entailment_loss": part_text_image_entailment,
        "cross_image_entailment_loss": cross_image_entailment,
        "cross_text_entailment_loss": cross_text_entailment,
        "cross_image_calibration_loss": cross_image_calibration,
        "cross_text_calibration_loss": cross_text_calibration,
    }


def uncha_argent_entailment_losses(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    kappa: Tensor,
    beta: float = 1.0,
    part_weights: Tensor | None = None,
    product_metric: str = "l1",
    aggregation: str = "uncha",
) -> dict[str, Tensor]:
    if aggregation not in {"uncha", "equal"}:
        raise ValueError("aggregation must be 'uncha' or 'equal'")
    text_image = argent_adaptive_entailment_residual(
        specific=image_feats,
        general=text_feats,
        kappa=kappa,
        adaptive_weight=False,
        beta=beta,
        product_metric=product_metric,
    )
    text_image_entailment = 0.5 * text_image.mean()

    if part_image_flat.numel() == 0:
        zero = image_feats.new_zeros(())
        norm_regularization = argent_norm_regularization_loss(image_feats, text_feats)
        return {
            "entailment_loss": text_image_entailment,
            "text_image_entailment_loss": text_image_entailment,
            "part_text_image_entailment_loss": zero,
            "cross_image_entailment_loss": zero,
            "cross_text_entailment_loss": zero,
            "cross_image_calibration_loss": zero,
            "cross_text_calibration_loss": zero,
            "norm_regularization_loss": norm_regularization,
        }

    part_text_image = argent_adaptive_entailment_residual(
        specific=part_image_flat,
        general=part_text_flat,
        kappa=kappa,
        adaptive_weight=False,
        beta=beta,
        product_metric=product_metric,
    )
    cross_image = argent_adaptive_entailment_residual(
        specific=image_for_parts,
        general=part_image_flat,
        kappa=kappa,
        adaptive_weight=True,
        beta=beta,
        product_metric=product_metric,
    )
    cross_text = argent_adaptive_entailment_residual(
        specific=text_for_parts,
        general=part_text_flat,
        kappa=kappa,
        adaptive_weight=True,
        beta=beta,
        product_metric=product_metric,
    )

    part_text_image_entailment = 0.5 * weighted_mean(part_text_image, part_weights)
    cross_image_entailment = 0.5 * weighted_mean(cross_image, part_weights)
    cross_text_entailment = 0.5 * weighted_mean(cross_text, part_weights)
    norm_regularization = argent_norm_regularization_loss(
        image_feats, text_feats, part_image_flat, part_text_flat
    )
    if aggregation == "equal":
        entailment = (
            text_image_entailment
            + part_text_image_entailment
            + cross_image_entailment
            + cross_text_entailment
        )
    else:
        entailment = (
            text_image_entailment
            + part_text_image_entailment
            + 0.5 * (cross_image_entailment + cross_text_entailment)
        )
    diagnostics = argent_entailment_diagnostics(
        image_feats=image_feats,
        text_feats=text_feats,
        part_image_flat=part_image_flat,
        part_text_flat=part_text_flat,
        image_for_parts=image_for_parts,
        text_for_parts=text_for_parts,
        kappa=kappa,
        product_metric=product_metric,
    )

    return {
        "entailment_loss": entailment,
        "text_image_entailment_loss": text_image_entailment,
        "part_text_image_entailment_loss": part_text_image_entailment,
        "cross_image_entailment_loss": cross_image_entailment,
        "cross_text_entailment_loss": cross_text_entailment,
        "cross_image_calibration_loss": image_feats.new_zeros(()),
        "cross_text_calibration_loss": image_feats.new_zeros(()),
        "norm_regularization_loss": norm_regularization,
        **diagnostics,
    }


def hierarchical_beta_argent_entailment_losses(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    beta_query_image_feats: Tensor,
    beta_query_text_feats: Tensor,
    beta_query_owner: Tensor,
    beta_query_parent: Tensor,
    beta_query_weight: Tensor,
    kappa: Tensor,
    beta_query_source_part: Tensor | None = None,
    beta: float = 1.0,
    part_weights: Tensor | None = None,
    product_metric: str = "l1",
    aggregation: str = "uncha",
) -> dict[str, Tensor]:
    base = uncha_argent_entailment_losses(
        image_feats=image_feats,
        text_feats=text_feats,
        part_image_flat=part_image_flat,
        part_text_flat=part_text_flat,
        image_for_parts=image_for_parts,
        text_for_parts=text_for_parts,
        kappa=kappa,
        beta=beta,
        part_weights=part_weights,
        product_metric=product_metric,
        aggregation=aggregation,
    )
    if beta_query_image_feats.numel() == 0:
        return {
            **base,
            "hier_beta_query_text_entailment_loss": image_feats.new_zeros(()),
            "hier_beta_visual_entailment_loss": image_feats.new_zeros(()),
            "hier_beta_text_entailment_loss": image_feats.new_zeros(()),
            "hier_beta_sourcepart_visual_entailment_loss": image_feats.new_zeros(()),
            "hier_beta_sourcepart_text_entailment_loss": image_feats.new_zeros(()),
            "hier_beta_query_count": beta_query_owner.new_tensor(0),
            "hier_beta_sourcepart_query_count": beta_query_owner.new_tensor(0),
        }

    query_owner = beta_query_owner.to(device=image_feats.device, dtype=torch.long)
    query_weights = beta_query_weight.to(device=image_feats.device, dtype=torch.float32).clamp_min(
        0.0
    )
    if query_weights.numel() != beta_query_image_feats.size(0):
        raise ValueError("beta_query_weight must have one value per beta query")
    query_weights = query_weights / query_weights.mean().clamp_min(
        torch.finfo(query_weights.dtype).eps
    )

    query_text = argent_adaptive_entailment_residual(
        specific=beta_query_image_feats,
        general=beta_query_text_feats,
        kappa=kappa,
        adaptive_weight=False,
        beta=beta,
        product_metric=product_metric,
    )
    visual_hierarchy = argent_adaptive_entailment_residual(
        specific=image_feats.index_select(0, query_owner),
        general=beta_query_image_feats,
        kappa=kappa,
        adaptive_weight=True,
        beta=beta,
        product_metric=product_metric,
    )
    query_text_entailment = 0.5 * weighted_mean(query_text, query_weights)
    visual_entailment = 0.5 * weighted_mean(visual_hierarchy, query_weights)

    parent = beta_query_parent.to(device=image_feats.device, dtype=torch.long)
    parent_mask = (parent >= 0) & (parent < beta_query_text_feats.size(0)) & (query_weights > 0.0)
    if bool(parent_mask.any()):
        child_text = beta_query_text_feats[parent_mask]
        parent_text = beta_query_text_feats[parent[parent_mask]]
        text_hierarchy = argent_adaptive_entailment_residual(
            specific=parent_text,
            general=child_text,
            kappa=kappa,
            adaptive_weight=True,
            beta=beta,
            product_metric=product_metric,
        )
        text_entailment = 0.5 * weighted_mean(text_hierarchy, query_weights[parent_mask])
    else:
        text_entailment = image_feats.new_zeros(())

    sourcepart_visual_entailment = image_feats.new_zeros(())
    sourcepart_text_entailment = image_feats.new_zeros(())
    sourcepart_query_count = beta_query_owner.new_tensor(0)
    if beta_query_source_part is not None and part_image_flat.numel() > 0:
        source_part = beta_query_source_part.to(device=image_feats.device, dtype=torch.long)
        if source_part.numel() != beta_query_image_feats.size(0):
            raise ValueError("beta_query_source_part must have one value per beta query")
        source_mask = (
            (source_part >= 0) & (source_part < part_image_flat.size(0)) & (query_weights > 0.0)
        )
        if bool(source_mask.any()):
            source_indices = source_part[source_mask]
            sourcepart_visual = argent_adaptive_entailment_residual(
                specific=part_image_flat.index_select(0, source_indices),
                general=beta_query_image_feats[source_mask],
                kappa=kappa,
                adaptive_weight=True,
                beta=beta,
                product_metric=product_metric,
            )
            sourcepart_text = argent_adaptive_entailment_residual(
                specific=part_text_flat.index_select(0, source_indices),
                general=beta_query_text_feats[source_mask],
                kappa=kappa,
                adaptive_weight=True,
                beta=beta,
                product_metric=product_metric,
            )
            source_weights = query_weights[source_mask]
            sourcepart_visual_entailment = 0.5 * weighted_mean(sourcepart_visual, source_weights)
            sourcepart_text_entailment = 0.5 * weighted_mean(sourcepart_text, source_weights)
            sourcepart_query_count = beta_query_owner.new_tensor(int(source_mask.sum().item()))

    norm_regularization = argent_norm_regularization_loss(
        image_feats,
        text_feats,
        part_image_flat,
        part_text_flat,
        beta_query_image_feats,
        beta_query_text_feats,
    )
    sourcepart_entailment = 0.5 * (sourcepart_visual_entailment + sourcepart_text_entailment)
    query_entailment = (
        query_text_entailment + 0.5 * (visual_entailment + text_entailment) + sourcepart_entailment
    )
    return {
        **base,
        "entailment_loss": base["entailment_loss"] + query_entailment,
        "norm_regularization_loss": norm_regularization,
        "hier_beta_query_text_entailment_loss": query_text_entailment,
        "hier_beta_visual_entailment_loss": visual_entailment,
        "hier_beta_text_entailment_loss": text_entailment,
        "hier_beta_sourcepart_visual_entailment_loss": sourcepart_visual_entailment,
        "hier_beta_sourcepart_text_entailment_loss": sourcepart_text_entailment,
        "hier_beta_query_count": beta_query_owner.new_tensor(beta_query_owner.numel()),
        "hier_beta_sourcepart_query_count": sourcepart_query_count,
    }


def argent_entailment_diagnostics(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    kappa: Tensor,
    product_metric: str = "l1",
) -> dict[str, Tensor]:
    zero = image_feats.new_zeros(())

    def angle_mean(specific: Tensor, general: Tensor) -> Tensor:
        if specific.numel() == 0:
            return zero
        angles = factor_oxy_angle(specific=specific, general=general, kappa=kappa)
        if angles.dim() == 2:
            angles = angles.mean(dim=-1)
        return angles.detach().mean()

    def pent_mean(specific: Tensor, general: Tensor) -> Tensor:
        if specific.numel() == 0:
            return zero
        angles = factor_oxy_angle(specific=specific, general=general, kappa=kappa)
        if angles.dim() == 2:
            angles = angles.mean(dim=-1)
        scores = torch.clamp(1.0 - (2.0 * angles / math.pi), min=0.0, max=1.0)
        return scores.detach().mean()

    def distance_mean(specific: Tensor, general: Tensor) -> Tensor:
        if specific.numel() == 0:
            return zero
        return lorentz_dist(specific, general, kappa, product_metric=product_metric).detach().mean()

    def adaptive_weight_mean(specific: Tensor, general: Tensor) -> Tensor:
        if specific.numel() == 0:
            return zero
        weights = 1.0 - torch.exp(
            -lorentz_dist(specific, general, kappa, product_metric=product_metric)
        )
        return weights.detach().mean()

    def space_norm_mean(embedding: Tensor) -> Tensor:
        if embedding.numel() == 0:
            return zero
        return torch.linalg.norm(_space_components(embedding).float(), dim=-1).detach().mean()

    return {
        "argent_text_image_angle_mean": angle_mean(image_feats, text_feats),
        "argent_text_image_pent_mean": pent_mean(image_feats, text_feats),
        "argent_part_text_image_angle_mean": angle_mean(part_image_flat, part_text_flat),
        "argent_part_text_image_pent_mean": pent_mean(part_image_flat, part_text_flat),
        "argent_cross_image_angle_mean": angle_mean(image_for_parts, part_image_flat),
        "argent_cross_image_pent_mean": pent_mean(image_for_parts, part_image_flat),
        "argent_cross_image_distance_mean": distance_mean(image_for_parts, part_image_flat),
        "argent_cross_image_adaptive_weight_mean": adaptive_weight_mean(
            image_for_parts, part_image_flat
        ),
        "argent_cross_text_angle_mean": angle_mean(text_for_parts, part_text_flat),
        "argent_cross_text_pent_mean": pent_mean(text_for_parts, part_text_flat),
        "argent_cross_text_distance_mean": distance_mean(text_for_parts, part_text_flat),
        "argent_cross_text_adaptive_weight_mean": adaptive_weight_mean(
            text_for_parts, part_text_flat
        ),
        "argent_image_space_norm_mean": space_norm_mean(image_feats),
        "argent_text_space_norm_mean": space_norm_mean(text_feats),
        "argent_part_image_space_norm_mean": space_norm_mean(part_image_flat),
        "argent_part_text_space_norm_mean": space_norm_mean(part_text_flat),
    }


def part_quality_weights(
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    part_owner: Tensor,
    batch_size: int,
    kappa: Tensor,
    mode: str,
    topk: int = 5,
    temperature: float = 4.0,
    product_metric: str = "l1",
) -> tuple[Tensor | None, Tensor, Tensor]:
    if mode not in {"none", "soft", "topk"}:
        raise ValueError(
            f"Unsupported part quality mode {mode!r}; expected 'none', 'soft', or 'topk'"
        )
    if mode == "none" or part_image_flat.numel() == 0:
        empty = part_image_flat.new_zeros((part_image_flat.size(0),))
        return None, empty, empty

    with torch.no_grad():
        image_parent = torch.exp(
            -lorentz_dist(part_image_flat, image_for_parts, kappa, product_metric=product_metric)
        )
        text_parent = torch.exp(
            -lorentz_dist(part_text_flat, text_for_parts, kappa, product_metric=product_metric)
        )
        image_text = torch.exp(
            -lorentz_dist(part_image_flat, part_text_flat, kappa, product_metric=product_metric)
        )
        scores = torch.stack([image_parent, text_parent, image_text]).mean(dim=0).clamp_min(0.0)

        if mode == "soft":
            weights = _owner_softmax_weights(scores, part_owner, batch_size, temperature)
        else:
            weights = _owner_topk_weights(scores, part_owner, batch_size, topk)
        weights = weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
    return weights, scores, (weights > 0.0).to(dtype=scores.dtype)


def _owner_softmax_weights(
    scores: Tensor, part_owner: Tensor, batch_size: int, temperature: float
) -> Tensor:
    weights = torch.zeros_like(scores)
    for owner in range(batch_size):
        mask = part_owner == owner
        if not bool(mask.any()):
            continue
        owner_scores = scores[mask]
        owner_weights = torch.softmax(owner_scores * temperature, dim=0) * owner_scores.numel()
        weights[mask] = owner_weights
    return weights


def _owner_topk_weights(scores: Tensor, part_owner: Tensor, batch_size: int, topk: int) -> Tensor:
    if topk <= 0:
        raise ValueError("topk must be positive for top-k part quality weighting")
    weights = torch.zeros_like(scores)
    for owner in range(batch_size):
        indices = torch.nonzero(part_owner == owner, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        keep = min(topk, indices.numel())
        selected = indices[scores[indices].topk(k=keep).indices]
        weights[selected] = 1.0
    return weights


def argent_adaptive_entailment_residual(
    specific: Tensor,
    general: Tensor,
    kappa: Tensor,
    adaptive_weight: bool,
    beta: float = 1.0,
    product_metric: str = "l1",
) -> Tensor:
    angles = factor_oxy_angle(specific=specific, general=general, kappa=kappa)
    if angles.dim() == 2:
        angles = angles.mean(dim=-1)
    if adaptive_weight:
        weights = 1.0 - torch.exp(
            -lorentz_dist(
                specific=specific, general=general, kappa=kappa, product_metric=product_metric
            )
        )
        angles = angles * weights
    return F.huber_loss(angles, torch.zeros_like(angles), delta=beta, reduction="none")


def lorentz_dist(
    specific: Tensor, general: Tensor, kappa: Tensor, product_metric: str = "l1"
) -> Tensor:
    return paired_dist(specific, general, kappa, product_metric=product_metric)


def argent_norm_regularization_loss(*embeddings: Tensor, eps: float = 1e-6) -> Tensor:
    losses = []
    for embedding in embeddings:
        if embedding.numel() == 0:
            continue
        space = _space_components(embedding)
        space_norm = torch.linalg.norm(space.float(), dim=-1).clamp_min(eps)
        losses.append((space_norm.square() - torch.log(space_norm)).mean())
    if not losses:
        raise ValueError(
            "argent_norm_regularization_loss requires at least one non-empty embedding tensor"
        )
    return torch.stack(losses).mean()


def piecewise_entailment_residual(
    specific: Tensor,
    general: Tensor,
    kappa: Tensor,
    aperture_scale: float,
    factor: float = 0.1,
    geometry: str = "lorentz",
) -> Tensor:
    if geometry == "lorentz":
        angles = factor_oxy_angle(specific=specific, general=general, kappa=kappa)
        apertures = factor_half_aperture(general=general, kappa=kappa)
    elif geometry == "euclidean":
        angles = euclidean_angle(specific=specific, general=general)
        apertures = euclidean_half_aperture(general=general, aperture_scale=aperture_scale)
        aperture_scale = 1.0
    else:
        raise ValueError(
            f"Unsupported entailment geometry {geometry!r}; expected 'lorentz' or 'euclidean'"
        )
    residual = angles - aperture_scale * apertures
    loss = torch.where(residual > 0.0, residual + factor * angles, factor * angles)
    return loss.mean(dim=-1) if loss.dim() == 2 else loss


def euclidean_angle(specific: Tensor, general: Tensor, eps: float = 1e-6) -> Tensor:
    specific_space = _space_components(specific).float()
    general_space = _space_components(general).float()
    numerator = (specific_space * general_space).sum(dim=-1)
    denominator = torch.linalg.norm(specific_space, dim=-1) * torch.linalg.norm(
        general_space, dim=-1
    )
    dtype_eps = torch.finfo(specific_space.dtype).eps
    angle_eps = max(eps, 16.0 * dtype_eps)
    cosine = (numerator / denominator.clamp_min(angle_eps)).clamp(
        min=-1.0 + angle_eps, max=1.0 - angle_eps
    )
    return torch.acos(cosine)


def euclidean_half_aperture(general: Tensor, aperture_scale: float, eps: float = 1e-8) -> Tensor:
    general_norm = torch.linalg.norm(_space_components(general).float(), dim=-1).clamp_min(eps)
    return torch.atan(
        torch.as_tensor(aperture_scale, device=general.device, dtype=general.dtype) / general_norm
    )


def aggregate_part_consistency_loss(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    part_owner: Tensor,
    part_weights: Tensor | None = None,
) -> Tensor:
    if part_image_flat.numel() == 0:
        return image_feats.new_zeros(())

    batch_size = image_feats.size(0)
    image_space = _space_components(image_feats).reshape(batch_size, -1).float()
    text_space = _space_components(text_feats).reshape(batch_size, -1).float()
    part_image_space = (
        _space_components(part_image_flat).reshape(part_image_flat.size(0), -1).float()
    )
    part_text_space = _space_components(part_text_flat).reshape(part_text_flat.size(0), -1).float()
    if part_weights is None:
        counts = torch.bincount(part_owner, minlength=batch_size).to(
            device=image_feats.device, dtype=image_space.dtype
        )
        denom = counts
        valid = counts > 0
        weights = part_image_space.new_ones((part_image_space.size(0),))
    else:
        weights = part_weights.to(device=image_feats.device, dtype=image_space.dtype).flatten()
        if weights.numel() != part_owner.numel():
            raise ValueError(
                "part_weights must have the same number of elements as part_owner when provided"
            )
        denom = torch.zeros(batch_size, device=image_feats.device, dtype=image_space.dtype)
        denom.index_add_(0, part_owner, weights)
        valid = denom > 0

    image_agg = image_space.new_zeros(image_space.shape)
    text_agg = text_space.new_zeros(text_space.shape)
    image_agg.index_add_(0, part_owner, part_image_space * weights[:, None])
    text_agg.index_add_(0, part_owner, part_text_space * weights[:, None])
    image_agg = image_agg[valid] / denom[valid, None].clamp_min(1.0)
    text_agg = text_agg[valid] / denom[valid, None].clamp_min(1.0)

    image_space = image_space[valid]
    text_space = text_space[valid]
    return 0.25 * (
        cosine_residual(image_agg, image_space)
        + cosine_residual(text_agg, text_space)
        + cosine_residual(image_agg, text_space)
        + cosine_residual(text_agg, image_space)
    )


def cosine_residual(x: Tensor, y: Tensor) -> Tensor:
    return (1.0 - F.cosine_similarity(x, y, dim=-1)).mean()


def uncertainty_calibrated_entailment_loss(
    entail_residual: Tensor,
    log_uncertainty: Tensor,
    alpha: float = 10.0,
    stop_grad: bool = True,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    mean_loss = 0.5 * entail_residual
    uncertainty = torch.exp(log_uncertainty).clamp(min=1e-6, max=1e6)
    residual = entail_residual.detach() if stop_grad else entail_residual
    scaled_entail = residual / (uncertainty + 1e-6)
    calibration_term = 0.5 * scaled_entail + 0.5 * log_uncertainty
    prob = torch.softmax(log_uncertainty.flatten(), dim=0)
    entropy = -(prob * torch.log(prob + 1e-8)).sum()
    calibration_loss = alpha * (calibration_term + entropy)
    return weighted_mean(mean_loss, weights), weighted_mean(calibration_loss, weights)


def embedding_uncertainty(x: Tensor) -> Tensor:
    space = _space_components(x)
    norm = torch.linalg.norm(space.float(), dim=-1)
    if norm.dim() > 1:
        norm = norm.mean(dim=-1)
    return F.softplus(-norm)


def _space_components(x: Tensor) -> Tensor:
    return x[..., 1:] if x.shape[-1] > 1 else x


def _flatten_valid_parts(
    part_image_feats: Tensor, part_text_feats: Tensor, part_mask: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    part_targets = targets[:, None].expand_as(part_mask)[part_mask]
    return part_image_feats[part_mask], part_text_feats[part_mask], part_targets
