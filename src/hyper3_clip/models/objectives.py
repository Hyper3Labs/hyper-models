from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from hyper3_clip.models.lorentz import log_map0, metric_pairwise_dist
from hyper3_clip.models.losses import (
    aggregate_part_consistency_loss,
    contrastive_ce,
    gramian_volume_loss,
    hierarchical_beta_argent_entailment_losses,
    packed_part_contrastive_loss,
    packed_part_entailment_loss,
    part_quality_weights,
    radius_order_hinge,
    uncha_argent_entailment_losses,
    uncha_contrastive_losses,
    uncha_entailment_losses,
)
from hyper3_clip.training.distributed import (
    gather_variable_many_with_grad,
    gather_variable_no_grad,
    get_rank,
)


class HyCoCLIPObjective(nn.Module):
    def __init__(
        self,
        entail_weight: float,
        inter_aperture_scale: float,
        intra_aperture_scale: float,
        product_metric: str = "l1",
    ) -> None:
        super().__init__()
        self.entail_weight = entail_weight
        self.inter_aperture_scale = inter_aperture_scale
        self.intra_aperture_scale = intra_aperture_scale
        self.product_metric = product_metric

    def forward(self, embeddings: Mapping[str, Tensor], logit_scale: Tensor) -> dict[str, Tensor]:
        part_owner = embeddings["part_owner"].long()
        part_count = part_owner.new_tensor(part_owner.numel())
        contrastive = packed_part_contrastive_loss(
            image_feats=embeddings["image_feats"],
            text_feats=embeddings["text_feats"],
            part_image_feats=embeddings["part_image_feats"],
            part_text_feats=embeddings["part_text_feats"],
            part_owner=part_owner,
            kappa=embeddings["kappa"],
            logit_scale=logit_scale,
            all_image_feats=embeddings.get("all_image_feats"),
            all_text_feats=embeddings.get("all_text_feats"),
            targets=embeddings.get("targets"),
        )
        entailment = packed_part_entailment_loss(
            image_feats=embeddings["image_feats"],
            text_feats=embeddings["text_feats"],
            part_image_feats=embeddings["part_image_feats"],
            part_text_feats=embeddings["part_text_feats"],
            part_owner=part_owner,
            kappa=embeddings["kappa"],
            inter_aperture_scale=self.inter_aperture_scale,
            intra_aperture_scale=self.intra_aperture_scale,
        )
        total = contrastive + self.entail_weight * entailment
        return {
            "loss": total,
            "contrastive_loss": contrastive,
            "entailment_loss": entailment,
            "part_count": part_count,
        }


class UNCHAObjective(nn.Module):
    def __init__(
        self,
        entail_weight: float,
        inter_aperture_scale: float,
        intra_aperture_scale: float,
        piecewise_factor: float = 0.1,
        calibration_alpha: float = 10.0,
        stop_grad_calibration: bool = True,
        entailment_geometry: str = "lorentz",
        aggregate_weight: float = 0.0,
        entailment_loss: str = "piecewise",
        argent_beta: float = 1.0,
        argent_norm_weight: float = 0.0,
        argent_aux_weight: float = 0.5,
        argent_aggregation: str = "uncha",
        part_weight_power: float = 0.0,
        product_metric: str = "l1",
        contrastive_loss: str = "ce",
        sigmoid_negative_weight: float = 1.0,
        part_quality_mode: str = "none",
        part_quality_topk: int = 5,
        part_quality_temperature: float = 4.0,
        contrastive_global_weight: float = 1.0,
        contrastive_local_weight: float = 1.0,
        contrastive_global_local_weight: float = 1.0,
        beta_cal_beta: float = 0.0,
        beta_cal_variant: str = "ce",
        beta_cal_weight: float = 0.0,
        himo_component_weight: float = 0.0,
        global_local_mode: str = "repeat",
        global_local_metric: str = "distance",
        global_local_angle_aux_weight: float = 0.0,
        global_local_angle_aux_mode: str = "contrastive",
        global_local_angle_aux_scale: float = 5.5,
        global_local_angle_aux_aperture_scale: float = 1.0,
        radius_order_weight: float = 0.0,
        radius_order_margin: float = 0.0,
        gramian_align_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if entailment_loss not in {
            "piecewise",
            "argent",
            "piecewise_argent",
            "hier_beta_argent",
            "hier_beta_sourcepart_argent",
        }:
            raise ValueError(
                f"Unsupported UNCHA entailment loss {entailment_loss!r}; "
                "expected 'piecewise', 'argent', 'piecewise_argent', 'hier_beta_argent', "
                "or 'hier_beta_sourcepart_argent'"
            )
        if contrastive_loss not in {"ce", "sigmoid", "siglip", "siglip_metric"}:
            raise ValueError(
                "contrastive_loss must be 'ce', 'sigmoid', 'siglip', or 'siglip_metric'"
            )
        if beta_cal_variant not in {"ce", "bce"}:
            raise ValueError("beta_cal_variant must be 'ce' or 'bce'")
        if argent_aggregation not in {"uncha", "equal"}:
            raise ValueError("argent_aggregation must be 'uncha' or 'equal'")
        if part_quality_mode not in {"none", "soft", "topk"}:
            raise ValueError("part_quality_mode must be 'none', 'soft', or 'topk'")
        if global_local_mode not in {"repeat", "inbatch"}:
            raise ValueError("global_local_mode must be 'repeat' or 'inbatch'")
        if global_local_metric not in {"distance", "angle"}:
            raise ValueError("global_local_metric must be 'distance' or 'angle'")
        if global_local_angle_aux_mode not in {"contrastive", "positive_hinge"}:
            raise ValueError(
                "global_local_angle_aux_mode must be 'contrastive' or 'positive_hinge'"
            )
        if global_local_angle_aux_weight < 0.0:
            raise ValueError("global_local_angle_aux_weight must be non-negative")
        if global_local_angle_aux_scale <= 0.0:
            raise ValueError("global_local_angle_aux_scale must be positive")
        if global_local_angle_aux_aperture_scale <= 0.0:
            raise ValueError("global_local_angle_aux_aperture_scale must be positive")
        if part_quality_topk <= 0:
            raise ValueError("part_quality_topk must be positive")
        self.entail_weight = entail_weight
        self.inter_aperture_scale = inter_aperture_scale
        self.intra_aperture_scale = intra_aperture_scale
        self.piecewise_factor = piecewise_factor
        self.calibration_alpha = calibration_alpha
        self.stop_grad_calibration = stop_grad_calibration
        self.entailment_geometry = entailment_geometry
        self.aggregate_weight = aggregate_weight
        self.entailment_loss = entailment_loss
        self.argent_beta = argent_beta
        self.argent_norm_weight = argent_norm_weight
        self.argent_aux_weight = argent_aux_weight
        self.argent_aggregation = argent_aggregation
        self.part_weight_power = part_weight_power
        self.product_metric = product_metric
        self.contrastive_loss = contrastive_loss
        self.sigmoid_negative_weight = sigmoid_negative_weight
        self.part_quality_mode = part_quality_mode
        self.part_quality_topk = part_quality_topk
        self.part_quality_temperature = part_quality_temperature
        self.contrastive_global_weight = float(contrastive_global_weight)
        self.contrastive_local_weight = float(contrastive_local_weight)
        self.contrastive_global_local_weight = float(contrastive_global_local_weight)
        self.beta_cal_beta = float(beta_cal_beta)
        self.beta_cal_variant = beta_cal_variant
        self.beta_cal_weight = float(beta_cal_weight)
        self.himo_component_weight = float(himo_component_weight)
        self.global_local_mode = global_local_mode
        self.global_local_metric = global_local_metric
        self.global_local_angle_aux_weight = float(global_local_angle_aux_weight)
        self.global_local_angle_aux_mode = global_local_angle_aux_mode
        self.global_local_angle_aux_scale = float(global_local_angle_aux_scale)
        self.global_local_angle_aux_aperture_scale = float(global_local_angle_aux_aperture_scale)
        self.radius_order_weight = float(radius_order_weight)
        self.radius_order_margin = float(radius_order_margin)
        self.gramian_align_weight = float(gramian_align_weight)

    def forward(
        self, embeddings: Mapping[str, Tensor], logit_scales: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        part_owner = embeddings["part_owner"].long()
        part_count = part_owner.new_tensor(part_owner.numel())
        part_image_flat = embeddings["part_image_feats"]
        part_text_flat = embeddings["part_text_feats"]
        image_feats = embeddings["image_feats"]
        text_feats = embeddings["text_feats"]

        if part_owner.numel() == 0:
            image_for_parts = image_feats.new_zeros((0, image_feats.size(-1)))
            text_for_parts = text_feats.new_zeros((0, text_feats.size(-1)))
        else:
            image_for_parts = image_feats[part_owner]
            text_for_parts = text_feats[part_owner]
        count_part_weights = _part_weights(part_owner, image_feats.size(0), self.part_weight_power)
        quality_part_weights, quality_scores, quality_keep = part_quality_weights(
            image_for_parts=image_for_parts,
            text_for_parts=text_for_parts,
            part_image_flat=part_image_flat,
            part_text_flat=part_text_flat,
            part_owner=part_owner,
            batch_size=image_feats.size(0),
            kappa=embeddings["kappa"],
            mode=self.part_quality_mode,
            topk=self.part_quality_topk,
            temperature=self.part_quality_temperature,
            product_metric=self.product_metric,
        )
        part_weights = _combine_part_weights(count_part_weights, quality_part_weights)

        needs_repeated_global_local = (
            self.global_local_mode == "repeat" and self.contrastive_global_local_weight != 0.0
        )
        part_feature_tensors = [part_image_flat, part_text_flat]
        if needs_repeated_global_local:
            part_feature_tensors.extend([image_for_parts, text_for_parts])
        gathered_part_features, part_counts = gather_variable_many_with_grad(part_feature_tensors)
        all_part_image_feats = gathered_part_features[0]
        all_part_text_feats = gathered_part_features[1]
        all_image_for_parts = gathered_part_features[2] if needs_repeated_global_local else None
        all_text_for_parts = gathered_part_features[3] if needs_repeated_global_local else None
        image_euc_feats = embeddings.get("image_euc_feats")
        text_euc_feats = embeddings.get("text_euc_feats")
        part_image_euc_flat = embeddings.get("part_image_euc_feats")
        part_text_euc_flat = embeddings.get("part_text_euc_feats")
        image_for_parts_euc = None
        text_for_parts_euc = None
        all_part_image_euc_feats = None
        all_part_text_euc_feats = None
        all_image_for_parts_euc = None
        all_text_for_parts_euc = None
        if (
            image_euc_feats is not None
            and text_euc_feats is not None
            and part_owner.numel() > 0
            and needs_repeated_global_local
        ):
            image_for_parts_euc = image_euc_feats[part_owner]
            text_for_parts_euc = text_euc_feats[part_owner]
        if part_image_euc_flat is not None and part_text_euc_flat is not None:
            euc_feature_tensors = [part_image_euc_flat, part_text_euc_flat]
            if image_for_parts_euc is not None and text_for_parts_euc is not None:
                euc_feature_tensors.extend([image_for_parts_euc, text_for_parts_euc])
            gathered_euc_features, _ = gather_variable_many_with_grad(euc_feature_tensors)
            all_part_image_euc_feats = gathered_euc_features[0]
            all_part_text_euc_feats = gathered_euc_features[1]
            if image_for_parts_euc is not None and text_for_parts_euc is not None:
                all_image_for_parts_euc = gathered_euc_features[2]
                all_text_for_parts_euc = gathered_euc_features[3]
        if "targets" not in embeddings:
            raise ValueError("UNCHAObjective requires 'targets' to compute group-aware losses")
        global_targets = embeddings["targets"]
        part_group_ids = (
            global_targets[part_owner] if part_owner.numel() > 0 else part_owner.new_zeros((0,))
        )
        all_part_group_ids = None
        if self.beta_cal_weight > 0.0 and self.beta_cal_beta > 0.0:
            all_part_group_ids, _ = gather_variable_no_grad(part_group_ids)
        part_offset = (
            part_counts[: get_rank()].sum()
            if part_counts.numel() > 1
            else part_counts.new_zeros(())
        )
        part_targets = (
            torch.arange(part_image_flat.size(0), device=part_image_flat.device) + part_offset
        )

        contrastive = uncha_contrastive_losses(
            image_feats=image_feats,
            text_feats=text_feats,
            part_image_flat=part_image_flat,
            part_text_flat=part_text_flat,
            image_for_parts=image_for_parts,
            text_for_parts=text_for_parts,
            image_euc_feats=image_euc_feats,
            text_euc_feats=text_euc_feats,
            part_image_euc_flat=part_image_euc_flat,
            part_text_euc_flat=part_text_euc_flat,
            image_for_parts_euc=image_for_parts_euc,
            text_for_parts_euc=text_for_parts_euc,
            kappa=embeddings["kappa"],
            global_logit_scale=logit_scales["global"],
            local_logit_scale=logit_scales["local"],
            global_local_logit_scale=logit_scales["global_local"],
            all_image_feats=embeddings.get("all_image_feats"),
            all_text_feats=embeddings.get("all_text_feats"),
            all_part_image_feats=all_part_image_feats,
            all_part_text_feats=all_part_text_feats,
            all_image_for_parts=all_image_for_parts,
            all_text_for_parts=all_text_for_parts,
            all_image_euc_feats=embeddings.get("all_image_euc_feats"),
            all_text_euc_feats=embeddings.get("all_text_euc_feats"),
            all_part_image_euc_feats=all_part_image_euc_feats,
            all_part_text_euc_feats=all_part_text_euc_feats,
            all_image_for_parts_euc=all_image_for_parts_euc,
            all_text_for_parts_euc=all_text_for_parts_euc,
            global_targets=global_targets,
            part_targets=part_targets,
            part_weights=part_weights,
            product_metric=self.product_metric,
            loss_type=self.contrastive_loss,
            contrastive_global_weight=self.contrastive_global_weight,
            contrastive_local_weight=self.contrastive_local_weight,
            contrastive_global_local_weight=self.contrastive_global_local_weight,
            beta_cal_beta=self.beta_cal_beta,
            beta_cal_variant=self.beta_cal_variant,
            beta_cal_weight=self.beta_cal_weight,
            part_group_ids=part_group_ids,
            all_part_group_ids=all_part_group_ids,
            global_logit_bias=logit_scales.get("global_bias"),
            local_logit_bias=logit_scales.get("local_bias"),
            global_local_logit_bias=logit_scales.get("global_local_bias"),
            sigmoid_negative_weight=self.sigmoid_negative_weight,
            global_local_mode=self.global_local_mode,
            global_local_metric=self.global_local_metric,
            global_local_angle_aux_weight=self.global_local_angle_aux_weight,
            global_local_angle_aux_mode=self.global_local_angle_aux_mode,
            global_local_angle_aux_scale=self.global_local_angle_aux_scale,
            global_local_angle_aux_aperture_scale=self.global_local_angle_aux_aperture_scale,
        )
        himo_component_loss = image_feats.new_zeros(())
        if self.himo_component_weight > 0.0 and embeddings.get("himo_text_feats") is not None:
            himo_text_feats = embeddings["himo_text_feats"]
            all_himo_text_feats = embeddings.get("all_himo_text_feats")
            if all_himo_text_feats is None:
                raise ValueError(
                    "himo_text_feats requires all_himo_text_feats for distributed contrastive loss"
                )
            scale = logit_scales["global"].exp().clamp(max=100.0)
            logits_i_t = (
                -metric_pairwise_dist(
                    image_feats,
                    all_himo_text_feats,
                    embeddings["kappa"],
                    product_metric=self.product_metric,
                )
                * scale
            )
            logits_t_i = (
                -metric_pairwise_dist(
                    himo_text_feats,
                    embeddings["all_image_feats"],
                    embeddings["kappa"],
                    product_metric=self.product_metric,
                )
                * scale
            )
            himo_component_loss = 0.5 * (
                contrastive_ce(logits_i_t, global_targets)
                + contrastive_ce(logits_t_i, global_targets)
            )
        if self.entailment_loss == "argent":
            entailment = uncha_argent_entailment_losses(
                image_feats=image_feats,
                text_feats=text_feats,
                part_image_flat=part_image_flat,
                part_text_flat=part_text_flat,
                image_for_parts=image_for_parts,
                text_for_parts=text_for_parts,
                kappa=embeddings["kappa"],
                beta=self.argent_beta,
                part_weights=part_weights,
                product_metric=self.product_metric,
                aggregation=self.argent_aggregation,
            )
        elif self.entailment_loss in {"hier_beta_argent", "hier_beta_sourcepart_argent"}:
            required = (
                "beta_query_image_feats",
                "beta_query_text_feats",
                "beta_query_owner",
                "beta_query_parent",
                "beta_query_weight",
            )
            if self.entailment_loss == "hier_beta_sourcepart_argent":
                required = (*required, "beta_query_source_part")
            missing = [key for key in required if embeddings.get(key) is None]
            if missing:
                raise ValueError(
                    f"{self.entailment_loss} requires beta query embeddings: missing {missing}"
                )
            entailment = hierarchical_beta_argent_entailment_losses(
                image_feats=image_feats,
                text_feats=text_feats,
                part_image_flat=part_image_flat,
                part_text_flat=part_text_flat,
                image_for_parts=image_for_parts,
                text_for_parts=text_for_parts,
                beta_query_image_feats=embeddings["beta_query_image_feats"],
                beta_query_text_feats=embeddings["beta_query_text_feats"],
                beta_query_owner=embeddings["beta_query_owner"],
                beta_query_parent=embeddings["beta_query_parent"],
                beta_query_weight=embeddings["beta_query_weight"],
                beta_query_source_part=embeddings.get("beta_query_source_part")
                if self.entailment_loss == "hier_beta_sourcepart_argent"
                else None,
                kappa=embeddings["kappa"],
                beta=self.argent_beta,
                part_weights=part_weights,
                product_metric=self.product_metric,
                aggregation=self.argent_aggregation,
            )
        else:
            piecewise_entailment = uncha_entailment_losses(
                image_feats=image_feats,
                text_feats=text_feats,
                part_image_flat=part_image_flat,
                part_text_flat=part_text_flat,
                image_for_parts=image_for_parts,
                text_for_parts=text_for_parts,
                kappa=embeddings["kappa"],
                inter_aperture_scale=self.inter_aperture_scale,
                intra_aperture_scale=self.intra_aperture_scale,
                piecewise_factor=self.piecewise_factor,
                calibration_alpha=self.calibration_alpha,
                stop_grad_calibration=self.stop_grad_calibration,
                geometry=self.entailment_geometry,
                part_weights=part_weights,
            )
            if self.entailment_loss == "piecewise_argent":
                argent_entailment = uncha_argent_entailment_losses(
                    image_feats=image_feats,
                    text_feats=text_feats,
                    part_image_flat=part_image_flat,
                    part_text_flat=part_text_flat,
                    image_for_parts=image_for_parts,
                    text_for_parts=text_for_parts,
                    kappa=embeddings["kappa"],
                    beta=self.argent_beta,
                    part_weights=part_weights,
                    product_metric=self.product_metric,
                    aggregation=self.argent_aggregation,
                )
                entailment = {
                    **piecewise_entailment,
                    "entailment_loss": piecewise_entailment["entailment_loss"]
                    + self.argent_aux_weight * argent_entailment["entailment_loss"],
                    "piecewise_entailment_loss": piecewise_entailment["entailment_loss"],
                    "argent_entailment_loss": argent_entailment["entailment_loss"],
                    "norm_regularization_loss": argent_entailment["norm_regularization_loss"],
                }
            else:
                entailment = piecewise_entailment
        aggregate = aggregate_part_consistency_loss(
            image_feats=image_feats,
            text_feats=text_feats,
            part_image_flat=part_image_flat,
            part_text_flat=part_text_flat,
            part_owner=part_owner,
            part_weights=part_weights,
        )
        radius_order = image_feats.new_zeros(())
        if self.radius_order_weight > 0.0:
            radius_order = (
                radius_order_hinge(
                    image_feats, text_feats, embeddings["kappa"], self.radius_order_margin
                )
                + radius_order_hinge(
                    part_image_flat,
                    part_text_flat,
                    embeddings["kappa"],
                    self.radius_order_margin,
                    part_weights,
                )
                + radius_order_hinge(
                    image_for_parts,
                    part_image_flat,
                    embeddings["kappa"],
                    self.radius_order_margin,
                    part_weights,
                )
                + radius_order_hinge(
                    text_for_parts,
                    part_text_flat,
                    embeddings["kappa"],
                    self.radius_order_margin,
                    part_weights,
                )
            )
        gramian_align = image_feats.new_zeros(())
        if self.gramian_align_weight > 0.0 and part_owner.numel() > 0:

            def _tangent_flat(x: Tensor) -> Tensor:
                tangent = log_map0(x, embeddings["kappa"])
                return tangent.reshape(tangent.size(0), -1) if tangent.dim() == 3 else tangent

            gramian_vectors = torch.stack(
                [
                    _tangent_flat(image_for_parts),
                    _tangent_flat(text_for_parts),
                    _tangent_flat(part_image_flat),
                    _tangent_flat(part_text_flat),
                ],
                dim=1,
            )
            gramian_align = gramian_volume_loss(gramian_vectors, part_weights)
        entail_weight_scale = embeddings.get("entail_weight_scale", image_feats.new_ones(()))
        total = (
            contrastive["contrastive_loss"]
            + self.himo_component_weight * himo_component_loss
            + self.entail_weight * entail_weight_scale * entailment["entailment_loss"]
            + self.aggregate_weight * aggregate
            + self.radius_order_weight * radius_order
            + self.gramian_align_weight * gramian_align
            + self.argent_norm_weight
            * entailment.get(
                "norm_regularization_loss",
                image_feats.new_zeros(()),
            )
        )
        return {
            "loss": total,
            **contrastive,
            "himo_component_contrastive_loss": himo_component_loss,
            **entailment,
            "aggregate_consistency_loss": aggregate,
            "radius_order_loss": radius_order,
            "gramian_align_loss": gramian_align,
            "part_count": part_count,
            "entail_weight_scale": entail_weight_scale.detach(),
            "part_quality_mean": (
                image_feats.new_zeros(())
                if quality_scores.numel() == 0
                else quality_scores.mean().detach()
            ),
            "part_quality_keep_fraction": (
                image_feats.new_zeros(())
                if quality_keep.numel() == 0
                else quality_keep.mean().detach()
            ),
        }


def build_objective(
    objective: str,
    entail_weight: float,
    inter_aperture_scale: float,
    intra_aperture_scale: float,
    uncha_piecewise_factor: float = 0.1,
    uncha_calibration_alpha: float = 10.0,
    uncha_stop_grad_calibration: bool = True,
    uncha_entailment_geometry: str = "lorentz",
    uncha_aggregate_weight: float = 0.0,
    uncha_entailment_loss: str = "piecewise",
    uncha_argent_beta: float = 1.0,
    uncha_argent_norm_weight: float = 0.0,
    uncha_argent_aux_weight: float = 0.5,
    uncha_argent_aggregation: str = "uncha",
    uncha_part_weight_power: float = 0.0,
    uncha_contrastive_loss: str = "ce",
    uncha_sigmoid_negative_weight: float = 1.0,
    uncha_part_quality_mode: str = "none",
    uncha_part_quality_topk: int = 5,
    uncha_part_quality_temperature: float = 4.0,
    uncha_contrastive_global_weight: float = 1.0,
    uncha_contrastive_local_weight: float = 1.0,
    uncha_contrastive_global_local_weight: float = 1.0,
    uncha_beta_cal_beta: float = 0.0,
    uncha_beta_cal_variant: str = "ce",
    uncha_beta_cal_weight: float = 0.0,
    uncha_himo_component_weight: float = 0.0,
    uncha_global_local_mode: str = "repeat",
    uncha_global_local_metric: str = "distance",
    uncha_global_local_angle_aux_weight: float = 0.0,
    uncha_global_local_angle_aux_mode: str = "contrastive",
    uncha_global_local_angle_aux_scale: float = 5.5,
    uncha_global_local_angle_aux_aperture_scale: float = 1.0,
    uncha_radius_order_weight: float = 0.0,
    uncha_radius_order_margin: float = 0.0,
    uncha_gramian_align_weight: float = 0.0,
    product_metric: str = "l1",
) -> nn.Module:
    if objective == "hycoclip":
        return HyCoCLIPObjective(
            entail_weight=entail_weight,
            inter_aperture_scale=inter_aperture_scale,
            intra_aperture_scale=intra_aperture_scale,
            product_metric=product_metric,
        )
    if objective == "uncha":
        return UNCHAObjective(
            entail_weight=entail_weight,
            inter_aperture_scale=inter_aperture_scale,
            intra_aperture_scale=intra_aperture_scale,
            piecewise_factor=uncha_piecewise_factor,
            calibration_alpha=uncha_calibration_alpha,
            stop_grad_calibration=uncha_stop_grad_calibration,
            entailment_geometry=uncha_entailment_geometry,
            aggregate_weight=uncha_aggregate_weight,
            entailment_loss=uncha_entailment_loss,
            argent_beta=uncha_argent_beta,
            argent_norm_weight=uncha_argent_norm_weight,
            argent_aux_weight=uncha_argent_aux_weight,
            argent_aggregation=uncha_argent_aggregation,
            part_weight_power=uncha_part_weight_power,
            product_metric=product_metric,
            contrastive_loss=uncha_contrastive_loss,
            sigmoid_negative_weight=uncha_sigmoid_negative_weight,
            part_quality_mode=uncha_part_quality_mode,
            part_quality_topk=uncha_part_quality_topk,
            part_quality_temperature=uncha_part_quality_temperature,
            contrastive_global_weight=uncha_contrastive_global_weight,
            contrastive_local_weight=uncha_contrastive_local_weight,
            contrastive_global_local_weight=uncha_contrastive_global_local_weight,
            beta_cal_beta=uncha_beta_cal_beta,
            beta_cal_variant=uncha_beta_cal_variant,
            beta_cal_weight=uncha_beta_cal_weight,
            himo_component_weight=uncha_himo_component_weight,
            global_local_mode=uncha_global_local_mode,
            global_local_metric=uncha_global_local_metric,
            global_local_angle_aux_weight=uncha_global_local_angle_aux_weight,
            global_local_angle_aux_mode=uncha_global_local_angle_aux_mode,
            global_local_angle_aux_scale=uncha_global_local_angle_aux_scale,
            global_local_angle_aux_aperture_scale=uncha_global_local_angle_aux_aperture_scale,
            radius_order_weight=uncha_radius_order_weight,
            radius_order_margin=uncha_radius_order_margin,
            gramian_align_weight=uncha_gramian_align_weight,
        )
    raise ValueError(f"Unsupported objective {objective!r}; expected 'hycoclip' or 'uncha'")


def _part_weights(part_owner: Tensor, batch_size: int, power: float) -> Tensor | None:
    if power <= 0.0 or part_owner.numel() == 0:
        return None
    counts = torch.bincount(part_owner, minlength=batch_size).to(
        dtype=torch.float32, device=part_owner.device
    )
    weights = counts[part_owner].clamp_min(1.0).pow(-power)
    return weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)


def _combine_part_weights(
    count_weights: Tensor | None, quality_weights: Tensor | None
) -> Tensor | None:
    if count_weights is None:
        return quality_weights
    if quality_weights is None:
        return count_weights
    weights = count_weights * quality_weights
    return weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
