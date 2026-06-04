from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hyper3_clip.models.encoders import TextEncoder, VisionEncoder
from hyper3_clip.models.experimental import ExperimentalObjectiveMixin
from hyper3_clip.models.himo import hide_reconstruct_embeddings
from hyper3_clip.models.lorentz import exp_map0, metric_similarity
from hyper3_clip.models.objectives import build_objective
from hyper3_clip.training.distributed import (
    gather_with_grad,
    get_rank,
    get_world_size,
    local_target_indices,
)


class Hyper3CLIP(ExperimentalObjectiveMixin, nn.Module):
    def __init__(
        self,
        vision_backbone: str,
        text_model_name: str,
        embed_dim: int,
        curv_init: float,
        learn_curv: bool,
        entail_weight: float,
        inter_aperture_scale: float,
        intra_aperture_scale: float,
        objective: str = "hycoclip",
        uncha_piecewise_factor: float = 0.1,
        uncha_calibration_alpha: float = 10.0,
        uncha_stop_grad_calibration: bool = True,
        vision_pretrained: bool = True,
        text_pretrained: bool = True,
        text_pooling: str = "auto",
        freeze_vision_encoder: bool = False,
        freeze_text_encoder: bool = False,
        normalize_encoder_features: bool = False,
        projection_hidden_dim: int | None = None,
        uncha_entailment_geometry: str = "lorentz",
        uncha_aggregate_weight: float = 0.0,
        uncha_entailment_loss: str = "piecewise",
        uncha_argent_beta: float = 1.0,
        uncha_argent_norm_weight: float = 0.0,
        uncha_argent_aux_weight: float = 0.5,
        uncha_argent_aggregation: str = "uncha",
        uncha_part_weight_power: float = 0.0,
        uncha_contrastive_loss: str = "ce",
        uncha_sigmoid_bias_init: float = -10.0,
        uncha_sigmoid_negative_weight: float = 1.0,
        uncha_part_quality_mode: str = "none",
        uncha_part_quality_topk: int = 5,
        uncha_part_quality_temperature: float = 4.0,
        uncha_entailment_warmup_steps: int = 0,
        uncha_contrastive_global_weight: float = 1.0,
        uncha_contrastive_local_weight: float = 1.0,
        uncha_contrastive_global_local_weight: float = 1.0,
        uncha_global_local_mode: str = "repeat",
        uncha_global_local_metric: str = "distance",
        uncha_global_local_angle_aux_weight: float = 0.0,
        uncha_global_local_angle_aux_mode: str = "contrastive",
        uncha_global_local_angle_aux_scale: float = 5.5,
        uncha_global_local_angle_aux_aperture_scale: float = 1.0,
        uncha_beta_cal_beta: float = 0.0,
        uncha_beta_cal_variant: str = "ce",
        uncha_beta_cal_weight: float = 0.0,
        uncha_himo_component_weight: float = 0.0,
        uncha_himo_variance_threshold: float = 0.9,
        uncha_himo_detach_pca: bool = True,
        uncha_radius_order_weight: float = 0.0,
        uncha_radius_order_margin: float = 0.0,
        uncha_gramian_align_weight: float = 0.0,
        phyclip_subspace_dim: int | None = None,
        phyclip_product_metric: str = "l1",
        proclip_weight: float = 0.0,
        proclip_component_dim: int | None = None,
        proclip_retrieval: bool = False,
        proclip_geometry: str = "product",
        proclip_dedicated_hyperbolic: bool = False,
        proclip_projection_hidden_dim: int | None = None,
        beta_clip_weight: float = 0.0,
        beta_clip_global_weight: float = 0.0,
        beta_clip_beta: float = 0.5,
        beta_clip_variant: str = "ce",
        beta_clip_similarity: str = "metric",
        beta_clip_num_heads: int = 8,
        beta_clip_mlp_ratio: float = 4.0,
        beta_clip_drop_cls_token: bool = True,
        tren_weight: float = 0.0,
        tren_visual_distill_weight: float = 1.0,
        tren_text_distill_weight: float = 1.0,
        tren_region_text_weight: float = 1.0,
        tren_num_region_tokens: int = 3,
        tren_num_decoder_layers: int = 2,
        tren_num_attention_heads: int = 8,
        tren_prompt_grid_size: int = 7,
        tren_dropout: float = 0.1,
        fuse_whole_part_encoder_forwards: bool = False,
        fuse_beta_query_encoder_forwards: bool = False,
        group_beta_query_pooling: bool = False,
        objective_autocast_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if objective not in {"hycoclip", "uncha", "proclip"}:
            raise ValueError(
                f"Unsupported objective {objective!r}; expected 'hycoclip', 'uncha', or 'proclip'"
            )
        if phyclip_product_metric not in {"l1", "l2"}:
            raise ValueError("phyclip_product_metric must be 'l1' or 'l2'")
        self._validate_experimental_options(
            proclip_geometry=proclip_geometry,
            proclip_projection_hidden_dim=proclip_projection_hidden_dim,
            proclip_component_dim=proclip_component_dim,
            beta_clip_weight=beta_clip_weight,
            beta_clip_global_weight=beta_clip_global_weight,
            beta_clip_beta=beta_clip_beta,
            beta_clip_variant=beta_clip_variant,
            beta_clip_similarity=beta_clip_similarity,
            beta_clip_num_heads=beta_clip_num_heads,
            beta_clip_mlp_ratio=beta_clip_mlp_ratio,
            tren_weight=tren_weight,
            tren_visual_distill_weight=tren_visual_distill_weight,
            tren_text_distill_weight=tren_text_distill_weight,
            tren_region_text_weight=tren_region_text_weight,
            tren_num_region_tokens=tren_num_region_tokens,
            tren_num_decoder_layers=tren_num_decoder_layers,
            tren_num_attention_heads=tren_num_attention_heads,
            tren_prompt_grid_size=tren_prompt_grid_size,
            tren_dropout=tren_dropout,
        )
        if objective_autocast_dtype not in {
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
        }:
            raise ValueError(
                "objective_autocast_dtype must be one of 'float32', 'float16', or 'bfloat16'"
            )
        if uncha_contrastive_loss not in {"ce", "sigmoid", "siglip", "siglip_metric"}:
            raise ValueError(
                "uncha_contrastive_loss must be 'ce', 'sigmoid', 'siglip', or 'siglip_metric'"
            )
        if uncha_global_local_metric not in {"distance", "angle"}:
            raise ValueError("uncha_global_local_metric must be 'distance' or 'angle'")
        if uncha_global_local_angle_aux_mode not in {"contrastive", "positive_hinge"}:
            raise ValueError(
                "uncha_global_local_angle_aux_mode must be 'contrastive' or 'positive_hinge'"
            )
        if uncha_global_local_angle_aux_weight < 0.0:
            raise ValueError("uncha_global_local_angle_aux_weight must be non-negative")
        if uncha_global_local_angle_aux_scale <= 0.0:
            raise ValueError("uncha_global_local_angle_aux_scale must be positive")
        if uncha_global_local_angle_aux_aperture_scale <= 0.0:
            raise ValueError("uncha_global_local_angle_aux_aperture_scale must be positive")
        if uncha_entailment_warmup_steps < 0:
            raise ValueError("uncha_entailment_warmup_steps must be non-negative")
        self.objective_name = objective
        self.uncha_contrastive_loss = uncha_contrastive_loss
        self.uncha_entailment_loss = uncha_entailment_loss
        self.uncha_entailment_warmup_steps = uncha_entailment_warmup_steps
        self.uncha_himo_component_weight = float(uncha_himo_component_weight)
        self.uncha_himo_variance_threshold = float(uncha_himo_variance_threshold)
        self.uncha_himo_detach_pca = bool(uncha_himo_detach_pca)
        self.proclip_weight = float(proclip_weight)
        self.proclip_retrieval = bool(proclip_retrieval)
        self.proclip_geometry = proclip_geometry
        self.proclip_dedicated_hyperbolic = bool(proclip_dedicated_hyperbolic)
        self.beta_clip_weight = float(beta_clip_weight)
        self.beta_clip_global_weight = float(beta_clip_global_weight)
        self.beta_clip_beta = float(beta_clip_beta)
        self.beta_clip_variant = beta_clip_variant
        self.beta_clip_similarity = beta_clip_similarity
        self.beta_clip_drop_cls_token = bool(beta_clip_drop_cls_token)
        self.tren_weight = float(tren_weight)
        self.tren_visual_distill_weight = float(tren_visual_distill_weight)
        self.tren_text_distill_weight = float(tren_text_distill_weight)
        self.tren_region_text_weight = float(tren_region_text_weight)
        self.fuse_whole_part_encoder_forwards = bool(fuse_whole_part_encoder_forwards)
        self.fuse_beta_query_encoder_forwards = bool(fuse_beta_query_encoder_forwards)
        self.group_beta_query_pooling = bool(group_beta_query_pooling)
        self.objective_autocast_dtype = objective_autocast_dtype
        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        self.normalize_encoder_features = bool(normalize_encoder_features)
        self.phyclip_subspace_dim = phyclip_subspace_dim
        self.phyclip_product_metric = phyclip_product_metric
        self.proclip_component_dim = proclip_component_dim
        if projection_hidden_dim is not None and projection_hidden_dim <= 0:
            raise ValueError("projection_hidden_dim must be positive when set")
        if self.proclip_enabled and phyclip_subspace_dim is not None:
            raise ValueError(
                "ProCLIP mixed-curvature proxy cannot be combined with PHyCLIP Lorentz factors"
            )
        if phyclip_subspace_dim is not None:
            if phyclip_subspace_dim <= 0:
                raise ValueError("phyclip_subspace_dim must be positive when set")
            if embed_dim % phyclip_subspace_dim != 0:
                raise ValueError("embed_dim must be divisible by phyclip_subspace_dim")
            self.phyclip_num_factors = embed_dim // phyclip_subspace_dim
        else:
            self.phyclip_num_factors = 0
        self.vision_encoder = VisionEncoder(vision_backbone, pretrained=vision_pretrained)
        self.text_encoder = TextEncoder(
            text_model_name, pretrained=text_pretrained, pooling=text_pooling
        )
        self.tokenizer = self.text_encoder.tokenizer
        self.embed_dim = embed_dim
        if self.freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
            self.vision_encoder.eval()
        if self.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

        self.image_proj = _projection_head(
            self.vision_encoder.output_dim, embed_dim, projection_hidden_dim
        )
        self.text_proj = _projection_head(
            self.text_encoder.output_dim, embed_dim, projection_hidden_dim
        )
        self._init_experimental_modules(
            beta_clip_num_heads=beta_clip_num_heads,
            beta_clip_mlp_ratio=beta_clip_mlp_ratio,
            tren_num_region_tokens=tren_num_region_tokens,
            tren_num_decoder_layers=tren_num_decoder_layers,
            tren_num_attention_heads=tren_num_attention_heads,
            tren_prompt_grid_size=tren_prompt_grid_size,
            tren_dropout=tren_dropout,
            projection_hidden_dim=projection_hidden_dim,
            proclip_projection_hidden_dim=proclip_projection_hidden_dim,
            projection_head=_projection_head,
        )

        if objective == "hycoclip":
            self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        elif objective == "uncha":
            self.global_logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
            self.local_logit_scale = nn.Parameter(torch.tensor(1 / 0.05).log())
            self.global_local_logit_scale = nn.Parameter(torch.tensor(1 / 0.06).log())
            if uncha_contrastive_loss in {"sigmoid", "siglip", "siglip_metric"}:
                self.global_logit_bias = nn.Parameter(torch.tensor(float(uncha_sigmoid_bias_init)))
                self.local_logit_bias = nn.Parameter(torch.tensor(float(uncha_sigmoid_bias_init)))
                self.global_local_logit_bias = nn.Parameter(
                    torch.tensor(float(uncha_sigmoid_bias_init))
                )
        alpha_dim = phyclip_subspace_dim or embed_dim
        alpha_shape = (self.phyclip_num_factors,) if self.phyclip_enabled else ()
        self.visual_alpha = nn.Parameter(torch.full(alpha_shape, alpha_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.full(alpha_shape, alpha_dim**-0.5).log())

        curv_shape = (self.phyclip_num_factors,) if self.phyclip_enabled else ()
        log_curv = torch.full(curv_shape, curv_init).log()
        self.log_curv = nn.Parameter(log_curv, requires_grad=learn_curv)
        self.curv_min = curv_init / 10.0
        self.curv_max = curv_init * 10.0
        self.objective = None
        if objective != "proclip":
            self.objective = build_objective(
                objective=objective,
                entail_weight=entail_weight,
                inter_aperture_scale=inter_aperture_scale,
                intra_aperture_scale=intra_aperture_scale,
                uncha_piecewise_factor=uncha_piecewise_factor,
                uncha_calibration_alpha=uncha_calibration_alpha,
                uncha_stop_grad_calibration=uncha_stop_grad_calibration,
                uncha_entailment_geometry=uncha_entailment_geometry,
                uncha_aggregate_weight=uncha_aggregate_weight,
                uncha_entailment_loss=uncha_entailment_loss,
                uncha_argent_beta=uncha_argent_beta,
                uncha_argent_norm_weight=uncha_argent_norm_weight,
                uncha_argent_aux_weight=uncha_argent_aux_weight,
                uncha_argent_aggregation=uncha_argent_aggregation,
                uncha_part_weight_power=uncha_part_weight_power,
                uncha_contrastive_loss=uncha_contrastive_loss,
                uncha_sigmoid_negative_weight=uncha_sigmoid_negative_weight,
                uncha_part_quality_mode=uncha_part_quality_mode,
                uncha_part_quality_topk=uncha_part_quality_topk,
                uncha_part_quality_temperature=uncha_part_quality_temperature,
                uncha_contrastive_global_weight=uncha_contrastive_global_weight,
                uncha_contrastive_local_weight=uncha_contrastive_local_weight,
                uncha_contrastive_global_local_weight=uncha_contrastive_global_local_weight,
                uncha_global_local_mode=uncha_global_local_mode,
                uncha_global_local_metric=uncha_global_local_metric,
                uncha_global_local_angle_aux_weight=uncha_global_local_angle_aux_weight,
                uncha_global_local_angle_aux_mode=uncha_global_local_angle_aux_mode,
                uncha_global_local_angle_aux_scale=uncha_global_local_angle_aux_scale,
                uncha_global_local_angle_aux_aperture_scale=uncha_global_local_angle_aux_aperture_scale,
                uncha_beta_cal_beta=uncha_beta_cal_beta,
                uncha_beta_cal_variant=uncha_beta_cal_variant,
                uncha_beta_cal_weight=uncha_beta_cal_weight,
                uncha_himo_component_weight=uncha_himo_component_weight,
                uncha_radius_order_weight=uncha_radius_order_weight,
                uncha_radius_order_margin=uncha_radius_order_margin,
                uncha_gramian_align_weight=uncha_gramian_align_weight,
                product_metric=phyclip_product_metric,
            )

    def train(self, mode: bool = True) -> Hyper3CLIP:
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_encoder.eval()
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    @property
    def phyclip_enabled(self) -> bool:
        return self.phyclip_subspace_dim is not None

    def _kappa(self) -> torch.Tensor:
        return self.log_curv.exp().clamp(min=self.curv_min, max=self.curv_max)

    def encode_image(self, image: torch.Tensor, project: bool = True) -> torch.Tensor:
        feats = self.image_proj(self.encode_image_base(image))
        if not project:
            return feats
        return self.project_image_features(feats)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, project: bool = True
    ) -> torch.Tensor:
        feats = self.text_proj(self.encode_text_base(input_ids, attention_mask))
        if not project:
            return feats
        return self.project_text_features(feats)

    def encode_image_base(self, image: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.training and not self.freeze_vision_encoder):
            feats = self.vision_encoder(image)
        feats = feats.detach() if self.freeze_vision_encoder else feats
        return F.normalize(feats.float(), dim=-1) if self.normalize_encoder_features else feats

    def encode_image_base_with_tokens(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(self.training and not self.freeze_vision_encoder):
            feats, tokens = self.vision_encoder.forward_with_tokens(image)
        if self.freeze_vision_encoder:
            feats = feats.detach()
            tokens = tokens.detach()
        if self.normalize_encoder_features:
            feats = F.normalize(feats.float(), dim=-1)
        return feats, tokens

    def encode_text_base(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.set_grad_enabled(self.training and not self.freeze_text_encoder):
            feats = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        feats = feats.detach() if self.freeze_text_encoder else feats
        return F.normalize(feats.float(), dim=-1) if self.normalize_encoder_features else feats

    def project_image_features(self, feats: torch.Tensor) -> torch.Tensor:
        if self.phyclip_enabled:
            return self._project_product_features(feats, self.visual_alpha)
        return exp_map0(feats.float() * self.visual_alpha.exp().float(), self._kappa().float())

    def project_text_features(self, feats: torch.Tensor) -> torch.Tensor:
        if self.phyclip_enabled:
            return self._project_product_features(feats, self.textual_alpha)
        return exp_map0(feats.float() * self.textual_alpha.exp().float(), self._kappa().float())

    def similarity_scores(
        self, image_feats: torch.Tensor, text_feats: torch.Tensor
    ) -> torch.Tensor:
        return metric_similarity(
            image_feats, text_feats, self._kappa(), product_metric=self.phyclip_product_metric
        )

    def encode_retrieval_image(self, image: torch.Tensor) -> torch.Tensor:
        base = self.encode_image_base(image)
        tangent = self.image_proj(base)
        if self.proclip_retrieval:
            return self._project_proclip_image_base(base, self.project_image_features(tangent))
        return self.project_image_features(tangent)

    def encode_retrieval_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        base = self.encode_text_base(input_ids, attention_mask)
        tangent = self.text_proj(base)
        if self.proclip_retrieval:
            return self._project_proclip_text_base(base, self.project_text_features(tangent))
        return self.project_text_features(tangent)

    def retrieval_similarity_scores(
        self, image_feats: torch.Tensor, text_feats: torch.Tensor
    ) -> torch.Tensor:
        if self.proclip_retrieval:
            return self._proclip_similarity_scores(image_feats, text_feats)
        return self.similarity_scores(image_feats, text_feats)

    @property
    def retrieval_requires_chunking(self) -> bool:
        return self.phyclip_enabled or self.proclip_retrieval

    def _objective_autocast(self, device_type: str):
        dtype = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }[self.objective_autocast_dtype]
        enabled = device_type != "cpu" and dtype is not torch.float32
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)

    def forward(
        self,
        image: torch.Tensor,
        part_images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        part_text_input_ids: torch.Tensor,
        part_text_attention_mask: torch.Tensor,
        part_owner: torch.Tensor,
        step: int | None = None,
        beta_query_input_ids: torch.Tensor | None = None,
        beta_query_attention_mask: torch.Tensor | None = None,
        beta_query_owner: torch.Tensor | None = None,
        beta_query_type: torch.Tensor | None = None,
        beta_query_parent: torch.Tensor | None = None,
        beta_query_weight: torch.Tensor | None = None,
        beta_query_source_part: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            self._clamp_logit_scales()
            self.visual_alpha.clamp_(max=0.0)
            self.textual_alpha.clamp_(max=0.0)
        kappa = self._kappa()

        feature_dim = self.embed_dim
        beta_image_tokens = None
        beta_query_base = None
        part_image_base = part_images.new_zeros((0, self.vision_encoder.output_dim))
        part_text_base = part_images.new_zeros((0, self.text_encoder.output_dim))
        hier_beta_enabled = self.objective_name == "uncha" and self.uncha_entailment_loss in {
            "hier_beta_argent",
            "hier_beta_sourcepart_argent",
        }
        if (
            hier_beta_enabled
            and self.fuse_beta_query_encoder_forwards
            and not self.tren_enabled
            and beta_query_input_ids is not None
            and beta_query_attention_mask is not None
            and part_images.shape[0] > 0
        ):
            (
                image_base,
                text_base,
                image_euc,
                text_euc,
                image_feats,
                text_feats,
                part_image_feats,
                part_text_feats,
                part_image_euc,
                part_text_euc,
                part_image_base,
                part_text_base,
                beta_image_tokens,
                beta_query_base,
            ) = self._encode_hier_beta_whole_parts_and_queries(
                image=image,
                part_images=part_images,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                part_text_input_ids=part_text_input_ids,
                part_text_attention_mask=part_text_attention_mask,
                beta_query_input_ids=beta_query_input_ids,
                beta_query_attention_mask=beta_query_attention_mask,
            )
        elif self.beta_query_pooling_enabled or self.tren_enabled:
            image_base, beta_image_tokens = self.encode_image_base_with_tokens(image)
            text_base = self.encode_text_base(text_input_ids, text_attention_mask)
            image_euc = self.image_proj(image_base)
            text_euc = self.text_proj(text_base)
            image_feats = self.project_image_features(image_euc)
            text_feats = self.project_text_features(text_euc)
            (
                part_image_feats,
                part_text_feats,
                part_image_euc,
                part_text_euc,
                part_image_base,
                part_text_base,
            ) = self._encode_parts_with_base(
                part_images=part_images,
                part_text_input_ids=part_text_input_ids,
                part_text_attention_mask=part_text_attention_mask,
                feature_dim=feature_dim,
            )
        elif (
            self.fuse_whole_part_encoder_forwards
            and self.objective_name != "proclip"
            and part_images.shape[0] > 0
        ):
            (
                image_base,
                text_base,
                image_euc,
                text_euc,
                image_feats,
                text_feats,
                part_image_feats,
                part_text_feats,
                part_image_euc,
                part_text_euc,
                part_image_base,
                part_text_base,
            ) = self._encode_whole_and_parts(
                image=image,
                part_images=part_images,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                part_text_input_ids=part_text_input_ids,
                part_text_attention_mask=part_text_attention_mask,
            )
        else:
            image_base = self.encode_image_base(image)
            text_base = self.encode_text_base(text_input_ids, text_attention_mask)
            image_euc = self.image_proj(image_base)
            text_euc = self.text_proj(text_base)
            image_feats = self.project_image_features(image_euc)
            text_feats = self.project_text_features(text_euc)
            (
                part_image_feats,
                part_text_feats,
                part_image_euc,
                part_text_euc,
                part_image_base,
                part_text_base,
            ) = self._encode_parts_with_base(
                part_images=part_images,
                part_text_input_ids=part_text_input_ids,
                part_text_attention_mask=part_text_attention_mask,
                feature_dim=feature_dim,
            )
        targets = local_target_indices(image_feats.size(0), image_feats.device)

        if self.objective_name == "proclip":
            proclip_image_feats = self._project_proclip_image_base(image_base, image_feats)
            proclip_text_feats = self._project_proclip_text_base(text_base, text_feats)
            proclip_loss = self._proclip_contrastive_loss(
                image_feats=proclip_image_feats,
                text_feats=proclip_text_feats,
                all_image_feats=gather_with_grad(proclip_image_feats),
                all_text_feats=gather_with_grad(proclip_text_feats),
                targets=targets,
            )
            zero = proclip_loss.new_zeros(())
            return {
                "loss": proclip_loss,
                "contrastive_loss": proclip_loss,
                "entailment_loss": zero,
                "part_count": part_owner.new_tensor(0),
                "proclip_contrastive_loss": proclip_loss,
                **self._detached_kappa_logs(kappa),
                **self._detached_logit_scales(),
            }

        himo_text_feats = None
        all_himo_text_feats = None
        if self.objective_name == "uncha" and self.uncha_himo_component_weight > 0.0:
            all_text_euc = gather_with_grad(text_euc)
            all_component_euc = hide_reconstruct_embeddings(
                all_text_euc,
                variance_threshold=self.uncha_himo_variance_threshold,
                detach_pca=self.uncha_himo_detach_pca,
            )
            if get_world_size() > 1:
                start = text_euc.size(0) * get_rank()
                end = start + text_euc.size(0)
                component_euc = all_component_euc[start:end]
            else:
                component_euc = all_component_euc
            himo_text_feats = self.project_text_features(component_euc)
            all_himo_text_feats = gather_with_grad(himo_text_feats)
        all_image_feats = gather_with_grad(image_feats)
        all_text_feats = gather_with_grad(text_feats)
        all_image_euc = None
        all_text_euc = None
        if self.objective_name == "uncha" and self.uncha_contrastive_loss == "siglip":
            all_image_euc = gather_with_grad(image_euc)
            all_text_euc = gather_with_grad(text_euc)
        part_owner = part_owner.to(device=image_feats.device, dtype=torch.long)
        beta_query_embeddings = {}
        if self.objective_name == "uncha" and self.uncha_entailment_loss in {
            "hier_beta_argent",
            "hier_beta_sourcepart_argent",
        }:
            if beta_image_tokens is None:
                raise RuntimeError(f"{self.uncha_entailment_loss} requires image patch tokens")
            with torch.autocast(device_type=image.device.type, enabled=False):
                beta_query_embeddings = self._beta_query_entailment_embeddings(
                    image_tokens=beta_image_tokens.float(),
                    beta_query_input_ids=beta_query_input_ids,
                    beta_query_attention_mask=beta_query_attention_mask,
                    beta_query_owner=beta_query_owner,
                    beta_query_parent=beta_query_parent,
                    beta_query_weight=beta_query_weight,
                    beta_query_source_part=beta_query_source_part,
                    kappa=kappa.float(),
                    query_base=beta_query_base,
                )

        with self._objective_autocast(image.device.type):
            if self.objective is None:
                raise RuntimeError("Non-ProCLIP forward requires an objective module")
            losses = self.objective(
                {
                    "image_feats": image_feats,
                    "text_feats": text_feats,
                    "part_image_feats": part_image_feats,
                    "part_text_feats": part_text_feats,
                    "part_owner": part_owner,
                    "all_image_feats": all_image_feats,
                    "all_text_feats": all_text_feats,
                    **(
                        {
                            "image_euc_feats": image_euc,
                            "text_euc_feats": text_euc,
                            "part_image_euc_feats": part_image_euc,
                            "part_text_euc_feats": part_text_euc,
                            "all_image_euc_feats": all_image_euc,
                            "all_text_euc_feats": all_text_euc,
                        }
                        if all_image_euc is not None and all_text_euc is not None
                        else {}
                    ),
                    "targets": targets,
                    "kappa": kappa,
                    "entail_weight_scale": self._entail_weight_scale(step, image_feats.device),
                    **beta_query_embeddings,
                    **(
                        {
                            "himo_text_feats": himo_text_feats,
                            "all_himo_text_feats": all_himo_text_feats,
                        }
                        if himo_text_feats is not None
                        else {}
                    ),
                },
                self._objective_logit_scales(),
            )

        if self.beta_clip_global_weight > 0.0:
            with torch.autocast(device_type=image.device.type, enabled=False):
                beta_clip_global_loss = self._beta_clip_global_contrastive_loss(
                    image_euc=image_euc,
                    text_euc=text_euc,
                    targets=targets,
                )
            losses = {
                **losses,
                "loss": losses["loss"] + self.beta_clip_global_weight * beta_clip_global_loss,
                "beta_clip_global_loss": beta_clip_global_loss,
            }

        if self.beta_clip_enabled:
            if beta_image_tokens is None:
                raise RuntimeError("beta-CLIP auxiliary requires image patch tokens")
            with torch.autocast(device_type=image.device.type, enabled=False):
                beta_clip_loss = self._beta_clip_auxiliary_loss(
                    image_tokens=beta_image_tokens.float(),
                    beta_query_input_ids=beta_query_input_ids,
                    beta_query_attention_mask=beta_query_attention_mask,
                    beta_query_owner=beta_query_owner,
                    global_targets=targets,
                    kappa=kappa.float(),
                )
            losses = {
                **losses,
                "loss": losses["loss"] + self.beta_clip_weight * beta_clip_loss,
                "beta_clip_loss": beta_clip_loss,
            }

        if self.tren_enabled:
            if beta_image_tokens is None:
                raise RuntimeError("T-REN auxiliary requires image patch tokens")
            with torch.autocast(device_type=image.device.type, enabled=False):
                tren_losses = self._tren_auxiliary_losses(
                    image_tokens=beta_image_tokens.float(),
                    part_owner=part_owner,
                    part_image_base=part_image_base.float(),
                    part_text_base=part_text_base.float(),
                )
            losses = {
                **losses,
                "loss": losses["loss"] + self.tren_weight * tren_losses["tren_loss"],
                **tren_losses,
            }

        if self.proclip_enabled and self.proclip_weight > 0.0:
            proclip_image_feats = self._project_proclip_image_base(image_base, image_feats)
            proclip_text_feats = self._project_proclip_text_base(text_base, text_feats)
            proclip_loss = self._proclip_contrastive_loss(
                image_feats=proclip_image_feats,
                text_feats=proclip_text_feats,
                all_image_feats=gather_with_grad(proclip_image_feats),
                all_text_feats=gather_with_grad(proclip_text_feats),
                targets=targets,
            )
            losses = {
                **losses,
                "loss": losses["loss"] + self.proclip_weight * proclip_loss,
                "proclip_contrastive_loss": proclip_loss,
            }

        return {**losses, **self._detached_kappa_logs(kappa), **self._detached_logit_scales()}

    def _encode_parts(
        self,
        part_images: torch.Tensor,
        part_text_input_ids: torch.Tensor,
        part_text_attention_mask: torch.Tensor,
        feature_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if part_images.shape[0] == 0:
            empty = part_images.new_zeros((0, feature_dim))
            return empty, empty, empty, empty

        part_image_euc = self.image_proj(self.encode_image_base(part_images))
        part_text_euc = self.text_proj(
            self.encode_text_base(part_text_input_ids, part_text_attention_mask)
        )
        part_image_feats = self.project_image_features(part_image_euc)
        part_text_feats = self.project_text_features(part_text_euc)
        return part_image_feats, part_text_feats, part_image_euc, part_text_euc

    def _encode_parts_with_base(
        self,
        part_images: torch.Tensor,
        part_text_input_ids: torch.Tensor,
        part_text_attention_mask: torch.Tensor,
        feature_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if part_images.shape[0] == 0:
            empty = part_images.new_zeros((0, feature_dim))
            empty_image_base = part_images.new_zeros((0, self.vision_encoder.output_dim))
            empty_text_base = part_images.new_zeros((0, self.text_encoder.output_dim))
            return empty, empty, empty, empty, empty_image_base, empty_text_base

        part_image_base = self.encode_image_base(part_images)
        part_text_base = self.encode_text_base(part_text_input_ids, part_text_attention_mask)
        part_image_euc = self.image_proj(part_image_base)
        part_text_euc = self.text_proj(part_text_base)
        part_image_feats = self.project_image_features(part_image_euc)
        part_text_feats = self.project_text_features(part_text_euc)
        return (
            part_image_feats,
            part_text_feats,
            part_image_euc,
            part_text_euc,
            part_image_base,
            part_text_base,
        )

    def _encode_whole_and_parts(
        self,
        image: torch.Tensor,
        part_images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        part_text_input_ids: torch.Tensor,
        part_text_attention_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = image.shape[0]
        part_count = part_images.shape[0]
        image_base_all = self.encode_image_base(torch.cat([image, part_images], dim=0))
        image_euc_all = self.image_proj(image_base_all)
        image_feats_all = self.project_image_features(image_euc_all)

        text_ids, text_mask = self._concat_text_batches(
            text_input_ids,
            text_attention_mask,
            part_text_input_ids,
            part_text_attention_mask,
        )
        text_base_all = self.encode_text_base(text_ids, text_mask)
        text_euc_all = self.text_proj(text_base_all)
        text_feats_all = self.project_text_features(text_euc_all)

        image_base, part_image_base = image_base_all.split([batch_size, part_count], dim=0)
        text_base, part_text_base = text_base_all.split([batch_size, part_count], dim=0)
        image_euc, part_image_euc = image_euc_all.split([batch_size, part_count], dim=0)
        text_euc, part_text_euc = text_euc_all.split([batch_size, part_count], dim=0)
        image_feats, part_image_feats = image_feats_all.split([batch_size, part_count], dim=0)
        text_feats, part_text_feats = text_feats_all.split([batch_size, part_count], dim=0)
        return (
            image_base,
            text_base,
            image_euc,
            text_euc,
            image_feats,
            text_feats,
            part_image_feats,
            part_text_feats,
            part_image_euc,
            part_text_euc,
            part_image_base,
            part_text_base,
        )

    def _encode_hier_beta_whole_parts_and_queries(
        self,
        image: torch.Tensor,
        part_images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        part_text_input_ids: torch.Tensor,
        part_text_attention_mask: torch.Tensor,
        beta_query_input_ids: torch.Tensor,
        beta_query_attention_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = image.shape[0]
        part_count = part_images.shape[0]
        query_count = beta_query_input_ids.shape[0]

        image_base_all, image_tokens_all = self.encode_image_base_with_tokens(
            torch.cat([image, part_images], dim=0)
        )
        image_euc_all = self.image_proj(image_base_all)
        image_feats_all = self.project_image_features(image_euc_all)
        image_base, part_image_base = image_base_all.split([batch_size, part_count], dim=0)
        image_euc, part_image_euc = image_euc_all.split([batch_size, part_count], dim=0)
        image_feats, part_image_feats = image_feats_all.split([batch_size, part_count], dim=0)
        beta_image_tokens = image_tokens_all[:batch_size]

        text_ids, text_mask = self._concat_text_batch_list(
            (text_input_ids, text_attention_mask),
            (part_text_input_ids, part_text_attention_mask),
            (beta_query_input_ids, beta_query_attention_mask),
        )
        text_base_all = self.encode_text_base(text_ids, text_mask)
        text_euc_all = self.text_proj(text_base_all)
        text_feats_all = self.project_text_features(text_euc_all)
        text_base, part_text_base, beta_query_base = text_base_all.split(
            [batch_size, part_count, query_count], dim=0
        )
        text_euc, part_text_euc, _ = text_euc_all.split(
            [batch_size, part_count, query_count], dim=0
        )
        text_feats, part_text_feats, _ = text_feats_all.split(
            [batch_size, part_count, query_count], dim=0
        )

        return (
            image_base,
            text_base,
            image_euc,
            text_euc,
            image_feats,
            text_feats,
            part_image_feats,
            part_text_feats,
            part_image_euc,
            part_text_euc,
            part_image_base,
            part_text_base,
            beta_image_tokens,
            beta_query_base,
        )

    def _concat_text_batches(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        part_text_input_ids: torch.Tensor,
        part_text_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._concat_text_batch_list(
            (text_input_ids, text_attention_mask),
            (part_text_input_ids, part_text_attention_mask),
        )

    def _concat_text_batch_list(
        self,
        *batches: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_length = max(input_ids.shape[1] for input_ids, _ in batches)
        pad_token_id = self.text_encoder.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        return (
            torch.cat(
                [
                    _pad_sequence_dim(input_ids, target_length, pad_token_id)
                    for input_ids, _ in batches
                ],
                dim=0,
            ),
            torch.cat(
                [
                    _pad_sequence_dim(attention_mask, target_length, 0)
                    for _, attention_mask in batches
                ],
                dim=0,
            ),
        )

    def _clamp_logit_scales(self) -> None:
        if self.objective_name == "proclip":
            self.proclip_logit_scale.clamp_(max=4.6052)
            self._clamp_experimental_logit_scales()
            return
        if self.objective_name == "hycoclip":
            self.logit_scale.clamp_(max=4.6052)
            self._clamp_experimental_logit_scales()
            return
        self.global_logit_scale.clamp_(max=4.6052)
        self.local_logit_scale.clamp_(max=4.6052)
        self.global_local_logit_scale.clamp_(max=4.6052)
        self._clamp_experimental_logit_scales()

    def _objective_logit_scales(self) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.objective_name == "hycoclip":
            return self.logit_scale
        if self.objective_name == "proclip":
            return self.proclip_logit_scale
        return {
            "global": self.global_logit_scale,
            "local": self.local_logit_scale,
            "global_local": self.global_local_logit_scale,
            **(
                {
                    "global_bias": self.global_logit_bias,
                    "local_bias": self.local_logit_bias,
                    "global_local_bias": self.global_local_logit_bias,
                }
                if self.uncha_contrastive_loss in {"sigmoid", "siglip", "siglip_metric"}
                else {}
            ),
        }

    def _detached_logit_scales(self) -> dict[str, torch.Tensor]:
        if self.objective_name == "proclip":
            return self._detached_experimental_logit_scales()
        if self.objective_name == "hycoclip":
            logs = {"logit_scale": self.logit_scale.exp().detach()}
            logs.update(self._detached_experimental_logit_scales())
            return logs
        logs = {
            "global_logit_scale": self.global_logit_scale.exp().detach(),
            "local_logit_scale": self.local_logit_scale.exp().detach(),
            "global_local_logit_scale": self.global_local_logit_scale.exp().detach(),
        }
        if self.uncha_contrastive_loss in {"sigmoid", "siglip", "siglip_metric"}:
            logs.update(
                {
                    "global_logit_bias": self.global_logit_bias.detach(),
                    "local_logit_bias": self.local_logit_bias.detach(),
                    "global_local_logit_bias": self.global_local_logit_bias.detach(),
                }
            )
        logs.update(self._detached_experimental_logit_scales())
        return logs

    def _project_product_features(self, feats: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        product_feats = feats.float().reshape(
            feats.size(0), self.phyclip_num_factors, self.phyclip_subspace_dim
        )
        product_feats = product_feats * alpha.exp().float().view(1, -1, 1)
        return exp_map0(product_feats, self._kappa().float().view(1, -1, 1))

    def _detached_kappa_logs(self, kappa: torch.Tensor) -> dict[str, torch.Tensor]:
        detached = kappa.detach()
        if detached.numel() == 1:
            return {"kappa": detached.reshape(())}
        return {
            "kappa": detached.mean(),
            "kappa_min": detached.min(),
            "kappa_max": detached.max(),
        }

    def _entail_weight_scale(self, step: int | None, device: torch.device) -> torch.Tensor:
        if self.uncha_entailment_warmup_steps <= 0 or step is None:
            return torch.ones((), device=device)
        scale = min(1.0, float(step + 1) / float(self.uncha_entailment_warmup_steps))
        return torch.tensor(scale, device=device)


def _projection_head(input_dim: int, output_dim: int, hidden_dim: int | None) -> nn.Module:
    if hidden_dim is None:
        return nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _pad_sequence_dim(tensor: torch.Tensor, target_length: int, value: int) -> torch.Tensor:
    pad = target_length - tensor.shape[1]
    if pad <= 0:
        return tensor
    return F.pad(tensor, (0, pad), value=value)
