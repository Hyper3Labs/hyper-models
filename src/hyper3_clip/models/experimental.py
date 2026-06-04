from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from hyper3_clip.models.lorentz import exp_map0, metric_pairwise_dist
from hyper3_clip.models.losses import beta_cal_loss
from hyper3_clip.models.tren import TRENRegionEncoder
from hyper3_clip.training.distributed import gather_variable_with_grad, gather_with_grad, get_rank

ProjectionHeadFactory = Callable[[int, int, int | None], nn.Module]


class ExperimentalObjectiveMixin:
    @staticmethod
    def _validate_experimental_options(
        *,
        proclip_geometry: str,
        proclip_projection_hidden_dim: int | None,
        proclip_component_dim: int | None,
        beta_clip_weight: float,
        beta_clip_global_weight: float,
        beta_clip_beta: float,
        beta_clip_variant: str,
        beta_clip_similarity: str,
        beta_clip_num_heads: int,
        beta_clip_mlp_ratio: float,
        tren_weight: float,
        tren_visual_distill_weight: float,
        tren_text_distill_weight: float,
        tren_region_text_weight: float,
        tren_num_region_tokens: int,
        tren_num_decoder_layers: int,
        tren_num_attention_heads: int,
        tren_prompt_grid_size: int,
        tren_dropout: float,
    ) -> None:
        if proclip_geometry not in {"product", "hyperbolic", "euclidean", "spherical", "clip"}:
            raise ValueError(
                "proclip_geometry must be 'product', 'hyperbolic', 'euclidean', 'spherical', or 'clip'"
            )
        if proclip_projection_hidden_dim is not None and proclip_projection_hidden_dim <= 0:
            raise ValueError("proclip_projection_hidden_dim must be positive when set")
        if proclip_component_dim is not None and proclip_component_dim <= 0:
            raise ValueError("proclip_component_dim must be positive when set")
        if beta_clip_variant not in {"ce", "bce"}:
            raise ValueError("beta_clip_variant must be 'ce' or 'bce'")
        if beta_clip_similarity not in {"metric", "dot"}:
            raise ValueError("beta_clip_similarity must be 'metric' or 'dot'")
        if beta_clip_weight < 0.0:
            raise ValueError("beta_clip_weight must be non-negative")
        if beta_clip_global_weight < 0.0:
            raise ValueError("beta_clip_global_weight must be non-negative")
        if beta_clip_beta < 0.0:
            raise ValueError("beta_clip_beta must be non-negative")
        if beta_clip_num_heads <= 0:
            raise ValueError("beta_clip_num_heads must be positive")
        if beta_clip_mlp_ratio <= 0.0:
            raise ValueError("beta_clip_mlp_ratio must be positive")
        if tren_weight < 0.0:
            raise ValueError("tren_weight must be non-negative")
        if (
            tren_visual_distill_weight < 0.0
            or tren_text_distill_weight < 0.0
            or tren_region_text_weight < 0.0
        ):
            raise ValueError("T-REN loss weights must be non-negative")
        if tren_num_region_tokens <= 0:
            raise ValueError("tren_num_region_tokens must be positive")
        if tren_num_decoder_layers <= 0:
            raise ValueError("tren_num_decoder_layers must be positive")
        if tren_num_attention_heads <= 0:
            raise ValueError("tren_num_attention_heads must be positive")
        if tren_prompt_grid_size <= 0:
            raise ValueError("tren_prompt_grid_size must be positive")
        if tren_dropout < 0.0:
            raise ValueError("tren_dropout must be non-negative")

    def _init_experimental_modules(
        self,
        *,
        beta_clip_num_heads: int,
        beta_clip_mlp_ratio: float,
        tren_num_region_tokens: int,
        tren_num_decoder_layers: int,
        tren_num_attention_heads: int,
        tren_prompt_grid_size: int,
        tren_dropout: float,
        projection_hidden_dim: int | None,
        proclip_projection_hidden_dim: int | None,
        projection_head: ProjectionHeadFactory,
    ) -> None:
        if self.beta_query_pooling_enabled:
            if self.vision_encoder.output_dim % beta_clip_num_heads != 0:
                raise ValueError(
                    "vision encoder output_dim must be divisible by beta_clip_num_heads"
                )
            beta_clip_hidden_dim = max(
                1, int(round(self.vision_encoder.output_dim * beta_clip_mlp_ratio))
            )
            self.beta_clip_text_query_proj = nn.Linear(
                self.text_encoder.output_dim, self.vision_encoder.output_dim
            )
            self.beta_clip_cross_attention = nn.MultiheadAttention(
                self.vision_encoder.output_dim,
                beta_clip_num_heads,
                batch_first=True,
            )
            self.beta_clip_mlp_norm = nn.LayerNorm(self.vision_encoder.output_dim)
            self.beta_clip_pool_mlp = nn.Sequential(
                nn.Linear(self.vision_encoder.output_dim, beta_clip_hidden_dim),
                nn.GELU(),
                nn.Linear(beta_clip_hidden_dim, self.vision_encoder.output_dim),
            )
        if self.beta_clip_enabled:
            self.beta_clip_logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        if self.tren_enabled:
            self.tren_region_encoder = TRENRegionEncoder(
                vision_dim=self.vision_encoder.output_dim,
                text_dim=self.text_encoder.output_dim,
                num_region_tokens=tren_num_region_tokens,
                num_decoder_layers=tren_num_decoder_layers,
                num_attention_heads=tren_num_attention_heads,
                prompt_grid_size=tren_prompt_grid_size,
                dropout=tren_dropout,
            )
            self.tren_logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        if self.proclip_enabled:
            component_dim = self._proclip_component_dim
            spherical_dim = self._proclip_spherical_ambient_dim
            proclip_hidden_dim = proclip_projection_hidden_dim
            if proclip_hidden_dim is None:
                proclip_hidden_dim = projection_hidden_dim
            if self.proclip_dedicated_hyperbolic:
                self.proclip_image_hyperbolic_proj = projection_head(
                    self.vision_encoder.output_dim, self.embed_dim, proclip_hidden_dim
                )
                self.proclip_text_hyperbolic_proj = projection_head(
                    self.text_encoder.output_dim, self.embed_dim, proclip_hidden_dim
                )
            self.proclip_image_euclidean_proj = projection_head(
                self.vision_encoder.output_dim, component_dim, proclip_hidden_dim
            )
            self.proclip_text_euclidean_proj = projection_head(
                self.text_encoder.output_dim, component_dim, proclip_hidden_dim
            )
            self.proclip_image_spherical_proj = projection_head(
                self.vision_encoder.output_dim, spherical_dim, proclip_hidden_dim
            )
            self.proclip_text_spherical_proj = projection_head(
                self.text_encoder.output_dim, spherical_dim, proclip_hidden_dim
            )
            self.proclip_logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
            self.proclip_log_weights = nn.Parameter(torch.zeros(3))

    @property
    def proclip_enabled(self) -> bool:
        return (
            self.objective_name == "proclip"
            or self.proclip_component_dim is not None
            or self.proclip_weight > 0.0
            or self.proclip_retrieval
        )

    @property
    def beta_clip_enabled(self) -> bool:
        return self.beta_clip_weight > 0.0

    @property
    def beta_query_pooling_enabled(self) -> bool:
        return self.beta_clip_enabled or (
            self.objective_name == "uncha"
            and self.uncha_entailment_loss in {"hier_beta_argent", "hier_beta_sourcepart_argent"}
        )

    @property
    def tren_enabled(self) -> bool:
        return self.tren_weight > 0.0

    @property
    def _proclip_component_dim(self) -> int:
        return int(self.proclip_component_dim or self.embed_dim)

    @property
    def _proclip_spherical_ambient_dim(self) -> int:
        return self._proclip_component_dim + 1

    def _clamp_experimental_logit_scales(self) -> None:
        if self.proclip_enabled:
            self.proclip_logit_scale.clamp_(max=4.6052)
        if self.beta_clip_enabled:
            self.beta_clip_logit_scale.clamp_(max=4.6052)
        if self.tren_enabled:
            self.tren_logit_scale.clamp_(max=4.6052)

    def _detached_experimental_logit_scales(self) -> dict[str, torch.Tensor]:
        logs = {}
        if self.proclip_enabled:
            logs.update(self._detached_proclip_logs())
        if self.beta_clip_enabled:
            logs["beta_clip_logit_scale"] = self.beta_clip_logit_scale.exp().detach()
        if self.tren_enabled:
            logs["tren_logit_scale"] = self.tren_logit_scale.exp().detach()
        return logs

    def _beta_clip_global_contrastive_loss(
        self,
        *,
        image_euc: torch.Tensor,
        text_euc: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        image_feats = F.normalize(image_euc.float(), dim=-1)
        text_feats = F.normalize(text_euc.float(), dim=-1)
        all_image_feats = gather_with_grad(image_feats)
        all_text_feats = gather_with_grad(text_feats)
        if self.objective_name == "hycoclip":
            scale = self.logit_scale.exp().clamp(max=100.0)
        elif self.objective_name == "proclip":
            scale = self.proclip_logit_scale.exp().clamp(max=100.0)
        else:
            scale = self.global_logit_scale.exp().clamp(max=100.0)
        logits_i_t = image_feats @ all_text_feats.T * scale
        logits_t_i = text_feats @ all_image_feats.T * scale
        return 0.5 * (F.cross_entropy(logits_i_t, targets) + F.cross_entropy(logits_t_i, targets))

    def _beta_query_entailment_embeddings(
        self,
        *,
        image_tokens: torch.Tensor,
        beta_query_input_ids: torch.Tensor | None,
        beta_query_attention_mask: torch.Tensor | None,
        beta_query_owner: torch.Tensor | None,
        beta_query_parent: torch.Tensor | None,
        beta_query_weight: torch.Tensor | None,
        beta_query_source_part: torch.Tensor | None,
        kappa: torch.Tensor,
        query_base: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            beta_query_input_ids is None
            or beta_query_attention_mask is None
            or beta_query_owner is None
        ):
            raise ValueError(
                f"{self.uncha_entailment_loss} requires beta query tensors from the collator"
            )
        if beta_query_parent is None or beta_query_weight is None:
            raise ValueError(
                f"{self.uncha_entailment_loss} requires beta query hierarchy metadata from the collator"
            )
        if (
            self.uncha_entailment_loss == "hier_beta_sourcepart_argent"
            and beta_query_source_part is None
        ):
            raise ValueError(
                "hier_beta_sourcepart_argent requires beta_query_source_part from the collator"
            )
        if beta_query_input_ids.shape[0] == 0:
            source_part = (
                beta_query_source_part.to(device=image_tokens.device, dtype=torch.long)
                if beta_query_source_part is not None
                else beta_query_owner.new_zeros((0,), device=image_tokens.device, dtype=torch.long)
            )
            return {
                "beta_query_image_feats": image_tokens.new_zeros((0, self.embed_dim)),
                "beta_query_text_feats": image_tokens.new_zeros((0, self.embed_dim)),
                "beta_query_owner": beta_query_owner.to(
                    device=image_tokens.device, dtype=torch.long
                ),
                "beta_query_parent": beta_query_parent.to(
                    device=image_tokens.device, dtype=torch.long
                ),
                "beta_query_weight": beta_query_weight.to(
                    device=image_tokens.device, dtype=torch.float32
                ),
                "beta_query_source_part": source_part,
            }

        query_owner = beta_query_owner.to(device=image_tokens.device, dtype=torch.long)
        if query_base is None:
            query_base = self.encode_text_base(beta_query_input_ids, beta_query_attention_mask)
        conditioned_image_base = self._beta_clip_text_conditioned_pool(
            image_tokens, query_base, query_owner
        )
        query_image_euc = self.image_proj(conditioned_image_base)
        query_text_euc = self.text_proj(query_base)
        return {
            "beta_query_image_feats": self.project_image_features(query_image_euc),
            "beta_query_text_feats": self.project_text_features(query_text_euc),
            "beta_query_owner": query_owner,
            "beta_query_parent": beta_query_parent.to(device=image_tokens.device, dtype=torch.long),
            "beta_query_weight": beta_query_weight.to(
                device=image_tokens.device, dtype=torch.float32
            ),
            **(
                {
                    "beta_query_source_part": beta_query_source_part.to(
                        device=image_tokens.device, dtype=torch.long
                    )
                }
                if beta_query_source_part is not None
                else {}
            ),
        }

    def _beta_clip_auxiliary_loss(
        self,
        *,
        image_tokens: torch.Tensor,
        beta_query_input_ids: torch.Tensor | None,
        beta_query_attention_mask: torch.Tensor | None,
        beta_query_owner: torch.Tensor | None,
        global_targets: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        if (
            beta_query_input_ids is None
            or beta_query_attention_mask is None
            or beta_query_owner is None
        ):
            raise ValueError("beta-CLIP auxiliary requires beta query tensors from the collator")
        if beta_query_input_ids.shape[0] == 0:
            return image_tokens.new_zeros(())

        beta_query_owner = beta_query_owner.to(device=image_tokens.device, dtype=torch.long)
        query_base = self.encode_text_base(beta_query_input_ids, beta_query_attention_mask)
        conditioned_image_base = self._beta_clip_text_conditioned_pool(
            image_tokens, query_base, beta_query_owner
        )
        query_image_euc = self.image_proj(conditioned_image_base)
        query_text_euc = self.text_proj(query_base)

        if self.beta_clip_similarity == "dot":
            query_image_feats = F.normalize(query_image_euc.float(), dim=-1)
            query_text_feats = F.normalize(query_text_euc.float(), dim=-1)
        else:
            query_image_feats = self.project_image_features(query_image_euc)
            query_text_feats = self.project_text_features(query_text_euc)

        all_query_image_feats, query_counts = gather_variable_with_grad(query_image_feats)
        all_query_text_feats, _ = gather_variable_with_grad(query_text_feats)
        query_offset = (
            query_counts[: get_rank()].sum()
            if query_counts.numel() > 1
            else query_counts.new_zeros(())
        )
        query_targets = (
            torch.arange(query_image_feats.size(0), device=query_image_feats.device) + query_offset
        )
        query_group_ids = global_targets.index_select(0, beta_query_owner)
        all_query_group_ids, _ = gather_variable_with_grad(query_group_ids)

        scale = self.beta_clip_logit_scale.exp().clamp(max=100.0)
        if self.beta_clip_similarity == "dot":
            logits_i_t = query_image_feats @ all_query_text_feats.T * scale
            logits_t_i = query_text_feats @ all_query_image_feats.T * scale
        else:
            logits_i_t = (
                -metric_pairwise_dist(
                    query_image_feats,
                    all_query_text_feats,
                    kappa,
                    product_metric=self.phyclip_product_metric,
                )
                * scale
            )
            logits_t_i = (
                -metric_pairwise_dist(
                    query_text_feats,
                    all_query_image_feats,
                    kappa,
                    product_metric=self.phyclip_product_metric,
                )
                * scale
            )
        return 0.5 * (
            beta_cal_loss(
                logits_i_t,
                targets=query_targets,
                group_ids=query_group_ids,
                all_group_ids=all_query_group_ids,
                beta=self.beta_clip_beta,
                variant=self.beta_clip_variant,
            )
            + beta_cal_loss(
                logits_t_i,
                targets=query_targets,
                group_ids=query_group_ids,
                all_group_ids=all_query_group_ids,
                beta=self.beta_clip_beta,
                variant=self.beta_clip_variant,
            )
        )

    def _beta_clip_text_conditioned_pool(
        self,
        image_tokens: torch.Tensor,
        query_base: torch.Tensor,
        query_owner: torch.Tensor,
    ) -> torch.Tensor:
        if image_tokens.ndim != 3:
            raise ValueError("beta-CLIP image tokens must have shape [batch, tokens, dim]")
        if getattr(self, "group_beta_query_pooling", False):
            return self._beta_clip_text_conditioned_pool_grouped(
                image_tokens, query_base, query_owner
            )
        if self.beta_clip_drop_cls_token and image_tokens.size(1) > 1:
            image_tokens = image_tokens[:, 1:, :]
        selected_tokens = image_tokens.index_select(0, query_owner).to(dtype=query_base.dtype)
        query = self.beta_clip_text_query_proj(query_base).unsqueeze(1)
        attended, _ = self.beta_clip_cross_attention(
            query, selected_tokens, selected_tokens, need_weights=False
        )
        pooled = attended.squeeze(1)
        return pooled + self.beta_clip_pool_mlp(self.beta_clip_mlp_norm(pooled))

    def _beta_clip_text_conditioned_pool_grouped(
        self,
        image_tokens: torch.Tensor,
        query_base: torch.Tensor,
        query_owner: torch.Tensor,
    ) -> torch.Tensor:
        if query_owner.numel() == 0:
            return query_base.new_zeros((0, self.vision_encoder.output_dim))
        if query_owner.min().item() < 0 or query_owner.max().item() >= image_tokens.size(0):
            raise IndexError("beta_query_owner contains an out-of-range image index")

        tokens = (
            image_tokens[:, 1:, :]
            if self.beta_clip_drop_cls_token and image_tokens.size(1) > 1
            else image_tokens
        )
        tokens = tokens.to(dtype=query_base.dtype)
        query_projected = self.beta_clip_text_query_proj(query_base)
        counts = torch.bincount(query_owner, minlength=image_tokens.size(0))
        max_queries = int(counts.max().item())

        order = torch.argsort(query_owner)
        sorted_owner = query_owner.index_select(0, order)
        owner_offsets = torch.zeros_like(counts)
        owner_offsets[1:] = counts.cumsum(0)[:-1]
        sorted_positions = torch.arange(
            query_owner.numel(), device=query_owner.device
        ) - owner_offsets.index_select(0, sorted_owner)
        positions = torch.empty_like(sorted_positions)
        positions[order] = sorted_positions

        packed_query = query_projected.new_zeros(
            (image_tokens.size(0), max_queries, query_projected.size(-1))
        )
        packed_query[query_owner, positions] = query_projected
        attended, _ = self.beta_clip_cross_attention(
            packed_query, tokens, tokens, need_weights=False
        )
        pooled = attended[query_owner, positions]
        return pooled + self.beta_clip_pool_mlp(self.beta_clip_mlp_norm(pooled))

    def _tren_auxiliary_losses(
        self,
        *,
        image_tokens: torch.Tensor,
        part_owner: torch.Tensor,
        part_image_base: torch.Tensor,
        part_text_base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = image_tokens.new_zeros(())
        if part_owner.numel() == 0:
            return {
                "tren_loss": zero,
                "tren_visual_distill_loss": zero,
                "tren_text_distill_loss": zero,
                "tren_region_text_contrastive_loss": zero,
                "tren_assignment_count": part_owner.new_tensor(0),
            }

        tren_outputs = self.tren_region_encoder(image_tokens)
        visual_tokens = tren_outputs["visual_tokens"].flatten(1, 2)
        text_tokens = tren_outputs["text_aligned_tokens"].flatten(1, 2)

        matched_visual: list[torch.Tensor] = []
        matched_text: list[torch.Tensor] = []
        target_visual: list[torch.Tensor] = []
        target_text: list[torch.Tensor] = []
        for owner in range(image_tokens.size(0)):
            region_mask = part_owner == owner
            if not bool(region_mask.any()):
                continue
            owner_target_visual = part_image_base[region_mask].detach()
            owner_target_text = part_text_base[region_mask].detach()
            owner_visual_tokens = visual_tokens[owner]
            owner_text_tokens = text_tokens[owner]
            pred_indices, target_indices = _greedy_region_assignment(
                owner_visual_tokens, owner_target_visual
            )
            if pred_indices.numel() == 0:
                continue
            matched_visual.append(owner_visual_tokens.index_select(0, pred_indices))
            matched_text.append(owner_text_tokens.index_select(0, pred_indices))
            target_visual.append(owner_target_visual.index_select(0, target_indices))
            target_text.append(owner_target_text.index_select(0, target_indices))

        if not matched_visual:
            return {
                "tren_loss": zero,
                "tren_visual_distill_loss": zero,
                "tren_text_distill_loss": zero,
                "tren_region_text_contrastive_loss": zero,
                "tren_assignment_count": part_owner.new_tensor(0),
            }

        matched_visual_tensor = torch.cat(matched_visual, dim=0)
        matched_text_tensor = torch.cat(matched_text, dim=0)
        target_visual_tensor = torch.cat(target_visual, dim=0)
        target_text_tensor = torch.cat(target_text, dim=0)
        visual_distill = (
            1.0 - F.cosine_similarity(matched_visual_tensor, target_visual_tensor, dim=-1).mean()
        )
        text_distill = (
            1.0 - F.cosine_similarity(matched_text_tensor, target_text_tensor, dim=-1).mean()
        )
        region_text = _symmetric_dot_contrastive(
            matched_text_tensor,
            target_text_tensor,
            scale=self.tren_logit_scale.exp().clamp(max=100.0),
        )
        total = (
            self.tren_visual_distill_weight * visual_distill
            + self.tren_text_distill_weight * text_distill
            + self.tren_region_text_weight * region_text
        )
        return {
            "tren_loss": total,
            "tren_visual_distill_loss": visual_distill,
            "tren_text_distill_loss": text_distill,
            "tren_region_text_contrastive_loss": region_text,
            "tren_assignment_count": part_owner.new_tensor(matched_visual_tensor.size(0)),
        }

    def _project_proclip_image_base(
        self, base_feats: torch.Tensor, hyperbolic: torch.Tensor
    ) -> torch.Tensor:
        if self.proclip_geometry == "clip":
            return F.normalize(base_feats.float(), dim=-1)
        if self.proclip_dedicated_hyperbolic:
            hyperbolic = exp_map0(
                self.proclip_image_hyperbolic_proj(base_feats.float()), self._kappa().float()
            )
        return self._pack_proclip_features(
            hyperbolic=hyperbolic,
            euclidean=self.proclip_image_euclidean_proj(base_feats.float()),
            spherical=self.proclip_image_spherical_proj(base_feats.float()),
        )

    def _project_proclip_text_base(
        self, base_feats: torch.Tensor, hyperbolic: torch.Tensor
    ) -> torch.Tensor:
        if self.proclip_geometry == "clip":
            return F.normalize(base_feats.float(), dim=-1)
        if self.proclip_dedicated_hyperbolic:
            hyperbolic = exp_map0(
                self.proclip_text_hyperbolic_proj(base_feats.float()), self._kappa().float()
            )
        return self._pack_proclip_features(
            hyperbolic=hyperbolic,
            euclidean=self.proclip_text_euclidean_proj(base_feats.float()),
            spherical=self.proclip_text_spherical_proj(base_feats.float()),
        )

    def _pack_proclip_features(
        self, hyperbolic: torch.Tensor, euclidean: torch.Tensor, spherical: torch.Tensor
    ) -> torch.Tensor:
        spherical = F.normalize(spherical.float(), dim=-1)
        if self.proclip_geometry == "hyperbolic":
            return hyperbolic.float()
        if self.proclip_geometry == "euclidean":
            return euclidean.float()
        if self.proclip_geometry == "spherical":
            return spherical
        return torch.cat([hyperbolic.float(), euclidean.float(), spherical], dim=-1)

    def _split_proclip_features(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hyperbolic_dim = self.embed_dim + 1
        component_dim = self._proclip_component_dim
        spherical_dim = self._proclip_spherical_ambient_dim
        hyperbolic = feats[:, :hyperbolic_dim]
        euclidean = feats[:, hyperbolic_dim : hyperbolic_dim + component_dim]
        spherical = feats[
            :, hyperbolic_dim + component_dim : hyperbolic_dim + component_dim + spherical_dim
        ]
        return hyperbolic, euclidean, spherical

    def _proclip_similarity_scores(
        self, image_feats: torch.Tensor, text_feats: torch.Tensor
    ) -> torch.Tensor:
        if self.proclip_geometry == "clip":
            return image_feats.float() @ text_feats.float().T
        if self.proclip_geometry == "hyperbolic":
            return -metric_pairwise_dist(image_feats, text_feats, self._kappa()).square()
        if self.proclip_geometry == "euclidean":
            return -torch.cdist(image_feats.float(), text_feats.float(), p=2).square()
        if self.proclip_geometry == "spherical":
            dot = (image_feats.float() @ text_feats.float().T).clamp(
                min=-1.0 + 1e-6, max=1.0 - 1e-6
            )
            return -torch.acos(dot).square()
        image_hyp, image_euc, image_sph = self._split_proclip_features(image_feats)
        text_hyp, text_euc, text_sph = self._split_proclip_features(text_feats)
        weights = self.proclip_log_weights.exp().to(device=image_feats.device, dtype=torch.float32)
        hyperbolic_dist2 = metric_pairwise_dist(image_hyp, text_hyp, self._kappa()).square()
        euclidean_dist2 = torch.cdist(image_euc.float(), text_euc.float(), p=2).square()
        spherical_dot = (image_sph.float() @ text_sph.float().T).clamp(
            min=-1.0 + 1e-6, max=1.0 - 1e-6
        )
        spherical_dist2 = torch.acos(spherical_dot).square()
        return -(
            weights[0] * hyperbolic_dist2
            + weights[1] * euclidean_dist2
            + weights[2] * spherical_dist2
        )

    def _proclip_contrastive_loss(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        all_image_feats: torch.Tensor,
        all_text_feats: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.proclip_logit_scale.exp().clamp(max=100.0)
        logits_i_t = self._proclip_similarity_scores(image_feats, all_text_feats) * scale
        logits_t_i = self._proclip_similarity_scores(text_feats, all_image_feats) * scale
        return 0.5 * (F.cross_entropy(logits_i_t, targets) + F.cross_entropy(logits_t_i, targets))

    def _detached_proclip_logs(self) -> dict[str, torch.Tensor]:
        weights = self.proclip_log_weights.exp().detach()
        return {
            "proclip_logit_scale": self.proclip_logit_scale.exp().detach(),
            "proclip_hyperbolic_weight": weights[0],
            "proclip_euclidean_weight": weights[1],
            "proclip_spherical_weight": weights[2],
        }


def _greedy_region_assignment(
    pred_tokens: torch.Tensor, target_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_tokens.numel() == 0 or target_tokens.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=pred_tokens.device)
        return empty, empty
    similarities = (
        F.normalize(pred_tokens.float(), dim=-1) @ F.normalize(target_tokens.float(), dim=-1).T
    )
    pair_scores = similarities.flatten()
    order = torch.argsort(pair_scores, descending=True)
    used_pred = torch.zeros(pred_tokens.size(0), dtype=torch.bool, device=pred_tokens.device)
    used_target = torch.zeros(target_tokens.size(0), dtype=torch.bool, device=pred_tokens.device)
    pred_indices: list[torch.Tensor] = []
    target_indices: list[torch.Tensor] = []
    for flat_index in order:
        pred_index = torch.div(flat_index, target_tokens.size(0), rounding_mode="floor")
        target_index = flat_index % target_tokens.size(0)
        if used_pred[pred_index] or used_target[target_index]:
            continue
        used_pred[pred_index] = True
        used_target[target_index] = True
        pred_indices.append(pred_index)
        target_indices.append(target_index)
        if len(target_indices) == target_tokens.size(0):
            break
    if not pred_indices:
        empty = torch.zeros((0,), dtype=torch.long, device=pred_tokens.device)
        return empty, empty
    return torch.stack(pred_indices), torch.stack(target_indices)


def _symmetric_dot_contrastive(
    region_tokens: torch.Tensor, text_tokens: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if region_tokens.size(0) == 1:
        return region_tokens.new_zeros(())
    region_tokens = F.normalize(region_tokens.float(), dim=-1)
    text_tokens = F.normalize(text_tokens.float(), dim=-1)
    logits = region_tokens @ text_tokens.T * scale
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
