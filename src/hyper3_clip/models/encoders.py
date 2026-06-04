from __future__ import annotations

import timm
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    SiglipTextConfig,
    SiglipTextModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)


class VisionEncoder(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.kind = "timm"
        if backbone_name.startswith("hf_clip_projected:"):
            self.kind = "hf_clip_projected"
            model_name = backbone_name.removeprefix("hf_clip_projected:")
            self.backbone = (
                CLIPVisionModelWithProjection.from_pretrained(model_name)
                if pretrained
                else CLIPVisionModelWithProjection(CLIPVisionConfig.from_pretrained(model_name))
            )
            self.output_dim = self.backbone.config.projection_dim
        elif backbone_name.startswith("hf_clip:"):
            self.kind = "hf_vision"
            model_name = backbone_name.removeprefix("hf_clip:")
            self.backbone = (
                CLIPVisionModel.from_pretrained(model_name)
                if pretrained
                else CLIPVisionModel(CLIPVisionConfig.from_pretrained(model_name))
            )
            self.output_dim = self.backbone.config.hidden_size
        elif backbone_name.startswith("hf_siglip:"):
            self.kind = "hf_vision"
            model_name = backbone_name.removeprefix("hf_siglip:")
            self.backbone = (
                SiglipVisionModel.from_pretrained(model_name)
                if pretrained
                else SiglipVisionModel(SiglipVisionConfig.from_pretrained(model_name))
            )
            self.output_dim = self.backbone.config.hidden_size
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            self.output_dim = self.backbone.num_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.kind == "hf_clip_projected":
            return self.backbone(pixel_values=image).image_embeds
        if self.kind == "hf_vision":
            out = self.backbone(pixel_values=image)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0]
        return self.backbone(image)

    def forward_with_tokens(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "hf_clip_projected":
            out = self.backbone(pixel_values=image)
            tokens = getattr(out, "last_hidden_state", None)
            if tokens is None and hasattr(out, "vision_model_output"):
                tokens = out.vision_model_output.last_hidden_state
            if tokens is None:
                raise RuntimeError("Projected CLIP vision output did not include patch tokens")
            return out.image_embeds, tokens
        if self.kind == "hf_vision":
            out = self.backbone(pixel_values=image)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
                pooled = out.last_hidden_state[:, 0]
            return pooled, out.last_hidden_state

        if not hasattr(self.backbone, "forward_features"):
            pooled = self.backbone(image)
            return pooled, pooled[:, None, :]
        features = self.backbone.forward_features(image)
        if hasattr(self.backbone, "forward_head"):
            pooled = self.backbone.forward_head(features, pre_logits=False)
        else:
            pooled = self.backbone(image)
        return pooled, _tokens_from_features(features)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, pooling: str = "auto") -> None:
        super().__init__()
        if pooling not in {"auto", "pooler", "cls", "mean"}:
            raise ValueError(
                f"Unsupported text pooling {pooling!r}; expected auto, pooler, cls, or mean"
            )
        self.kind = "hf_text"
        self.pooling = pooling
        tokenizer_name = model_name.removeprefix("hf_clip_projected:")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model_name_lower = model_name.lower()
        if model_name.startswith("hf_clip_projected:"):
            self.kind = "hf_clip_projected"
            projected_model_name = model_name.removeprefix("hf_clip_projected:")
            if pretrained:
                self.backbone = CLIPTextModelWithProjection.from_pretrained(projected_model_name)
            else:
                self.backbone = CLIPTextModelWithProjection(
                    CLIPTextConfig.from_pretrained(projected_model_name)
                )
            self.output_dim = self.backbone.config.projection_dim
        elif "siglip" in model_name_lower:
            if pretrained:
                self.backbone = SiglipTextModel.from_pretrained(model_name)
            else:
                self.backbone = SiglipTextModel(SiglipTextConfig.from_pretrained(model_name))
            self.output_dim = self.backbone.config.hidden_size
        elif "clip" in model_name_lower:
            if pretrained:
                self.backbone = CLIPTextModel.from_pretrained(model_name)
            else:
                self.backbone = CLIPTextModel(CLIPTextConfig.from_pretrained(model_name))
            self.output_dim = self.backbone.config.hidden_size
        else:
            if pretrained:
                self.backbone = AutoModel.from_pretrained(model_name)
            else:
                self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
            hidden_size = getattr(self.backbone.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(f"Unsupported text model config for {model_name}")
            self.output_dim = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.kind == "hf_clip_projected":
            return out.text_embeds
        if self.pooling == "mean":
            mask = attention_mask.to(dtype=out.last_hidden_state.dtype).unsqueeze(-1)
            summed = (out.last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return summed / denom
        if (
            self.pooling in {"auto", "pooler"}
            and hasattr(out, "pooler_output")
            and out.pooler_output is not None
        ):
            return out.pooler_output
        return out.last_hidden_state[:, 0]


def _tokens_from_features(features: torch.Tensor | dict | tuple | list) -> torch.Tensor:
    if isinstance(features, dict):
        for key in ("x", "last_hidden_state", "features"):
            if key in features:
                features = features[key]
                break
        else:
            features = next(iter(features.values()))
    if isinstance(features, tuple | list):
        features = features[0]
    if not torch.is_tensor(features):
        raise TypeError(f"Expected tensor features, got {type(features)!r}")
    if features.ndim == 4:
        return features.flatten(2).transpose(1, 2)
    if features.ndim == 3:
        return features
    if features.ndim == 2:
        return features[:, None, :]
    raise ValueError(f"Unsupported feature tensor shape {tuple(features.shape)}")
