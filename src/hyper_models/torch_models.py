"""Torch-backed catalog model implementations.

These are internal loader targets used by selected hyper-models catalog entries.
They do not create a separate public provider surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hyper_models.preprocessing import ImageConfig, preprocess_images

__all__ = ["Hyper3ClipTorchModel", "UNCHATorchModel"]


class _UNCHAImageEncoder:
    """Minimal UNCHA image encoder graph for inference-only embeddings."""

    def __init__(self, torch: Any, timm: Any, nn: Any, *, variant: str) -> None:
        self._torch = torch
        self._timm = timm
        self._nn = nn
        self._variant = variant

        self._register_custom_variants()
        self.visual = self._build_visual()
        self.visual_proj = nn.Linear(self.visual.width, 512, bias=False)

        self.visual_alpha = nn.Parameter(torch.tensor(512**-0.5).log())
        self.curv = nn.Parameter(torch.tensor(1.0).log())

        self.pixel_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def _register_custom_variants(self) -> None:
        if self._timm.models.is_model("vit_small_mocov3_patch16_224"):
            return

        @self._timm.models.register_model
        def vit_small_mocov3_patch16_224(**kwargs: Any):
            return self._timm.models.vision_transformer._create_vision_transformer(
                "vit_small_patch16_224",
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=12,
                **kwargs,
            )

    def _build_visual(self) -> Any:
        arch = {
            "vit_s": "vit_small_mocov3_patch16_224",
            "vit_b": "vit_base_patch16_224",
        }[self._variant]

        visual = self._timm.create_model(
            arch,
            num_classes=0,
            global_pool="token",
            class_token=True,
            norm_layer=self._nn.LayerNorm,
            pretrained=False,
        )

        # Match upstream UNCHA image encoder API expected by projection layers.
        visual.width = int(getattr(visual, "embed_dim", getattr(visual, "num_features", 0)))
        if visual.width <= 0:
            raise RuntimeError("Unable to infer visual width for UNCHA image encoder")
        return visual

    def _exp_map0(self, tangent: Any, curvature: Any) -> Any:
        torch = self._torch
        sqrt_curvature = torch.sqrt(curvature)
        norm = torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.sinh(sqrt_curvature * norm) / (sqrt_curvature * norm)
        return tangent * scale

    def _lift_hyperboloid(self, space: Any, curvature: Any) -> Any:
        torch = self._torch
        time = torch.sqrt((1.0 / curvature) + torch.sum(space * space, dim=-1, keepdim=True))
        return torch.cat([time, space], dim=-1)

    def encode(self, images_bchw: Any) -> Any:
        torch = self._torch

        images = (images_bchw - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        visual_scale = torch.exp(torch.clamp(self.visual_alpha, max=0.0))
        tangent = image_feats * visual_scale

        curvature = torch.exp(self.curv).clamp(min=1e-8)
        space = self._exp_map0(tangent, curvature)
        return self._lift_hyperboloid(space, curvature)


class UNCHATorchModel:
    """UNCHA catalog entry runtime using torch+timm for image inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        geometry: str,
        dim: int,
        variant: str,
        image_config: ImageConfig | None = None,
        device: str | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self.geometry = geometry
        self.dim = dim
        self._variant = variant
        self._image_config = image_config or ImageConfig()
        self._device_override = device

        self._torch = None
        self._device = None
        self._encoder = None

    def _import_ml_stack(self) -> tuple[Any, Any, Any]:
        try:
            import timm
            import torch
            from torch import nn
        except ImportError as e:
            raise ImportError(
                "This catalog entry requires the optional 'ml' dependencies. "
                "Install with: uv sync --extra ml or uv pip install 'hyper-models[ml]'"
            ) from e

        return torch, timm, nn

    def _resolve_device(self, torch: Any) -> Any:
        if self._device_override:
            return torch.device(self._device_override)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_state_dict(self) -> dict[str, Any]:
        assert self._torch is not None

        checkpoint_obj = self._torch.load(
            self._checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
            model_state = checkpoint_obj["model"]
            if isinstance(model_state, dict):
                return model_state

        if isinstance(checkpoint_obj, dict):
            return checkpoint_obj

        raise TypeError(
            "Unexpected UNCHA checkpoint format; expected dict or {'model': state_dict}"
        )

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return

        torch, timm, nn = self._import_ml_stack()
        self._torch = torch
        self._device = self._resolve_device(torch)

        encoder = _UNCHAImageEncoder(torch, timm, nn, variant=self._variant)
        state_dict = self._load_state_dict()

        own_state: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("model."):
                key = key[len("model.") :]

            if key.startswith("visual.") or key.startswith("visual_proj."):
                own_state[key] = value
            elif key in {"visual_alpha", "curv"}:
                own_state[key] = value

        module = torch.nn.Module()
        module.visual = encoder.visual
        module.visual_proj = encoder.visual_proj
        module.visual_alpha = encoder.visual_alpha
        module.curv = encoder.curv

        load_result = module.load_state_dict(own_state, strict=False)
        if len(own_state) == 0:
            raise RuntimeError("UNCHA checkpoint did not contain visual encoder weights")
        if "visual_proj.weight" in load_result.missing_keys:
            raise RuntimeError("UNCHA checkpoint missing required key: visual_proj.weight")

        encoder.visual = module.visual.to(self._device)
        encoder.visual_proj = module.visual_proj.to(self._device)
        encoder.visual_alpha = torch.nn.Parameter(module.visual_alpha.to(self._device))
        encoder.curv = torch.nn.Parameter(module.curv.to(self._device))
        encoder.pixel_mean = encoder.pixel_mean.to(self._device)
        encoder.pixel_std = encoder.pixel_std.to(self._device)

        encoder.visual.eval()
        encoder.visual_proj.eval()

        self._encoder = encoder

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode preprocessed inputs (B, C, H, W) to embeddings (B, D)."""
        self._ensure_encoder()

        assert self._encoder is not None
        assert self._torch is not None
        assert self._device is not None

        images = self._torch.from_numpy(inputs).to(device=self._device, dtype=self._torch.float32)
        with self._torch.inference_mode():
            emb = self._encoder.encode(images)

        return np.asarray(emb.detach().cpu().numpy(), dtype=np.float32)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode PIL images to embeddings (B, D)."""
        return self.encode(preprocess_images(images, self._image_config))


_CHECKPOINT_TEXT_PREFIX = "text_encoder.backbone.text_model."
_MODEL_TEXT_PREFIX = "text_encoder.backbone."


def _align_text_tower_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Match published Hyper3-CLIP text weights to the model's parameter names.

    The checkpoint stores the text tower under the full Hugging Face
    ``CLIPTextModel`` path, while :class:`Hyper3CLIP` binds its backbone one
    level deeper, at ``CLIPTextModel.text_model``. Without this rename the text
    tower silently loads no pretrained weights and every text embedding comes
    from a randomly initialised encoder.
    """

    return {
        (
            _MODEL_TEXT_PREFIX + key[len(_CHECKPOINT_TEXT_PREFIX) :]
            if key.startswith(_CHECKPOINT_TEXT_PREFIX)
            else key
        ): value
        for key, value in state.items()
    }


class Hyper3ClipTorchModel:
    """Hyper3-CLIP catalog entry runtime using torch for image inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        geometry: str,
        dim: int,
        image_config: ImageConfig | None = None,
        device: str | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._config_path = checkpoint_path.with_name("config.yaml")
        self.geometry = geometry
        self.dim = dim
        self._image_config = image_config or ImageConfig()
        self._device_override = device

        self._torch = None
        self._device = None
        self._model = None

    def _import_ml_stack(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            import yaml
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "This catalog entry requires the optional 'ml' dependencies. "
                "Install with: uv sync --extra ml or uv pip install 'hyper-models[ml]'"
            ) from e

        return torch, yaml, load_file

    def _resolve_device(self, torch: Any) -> Any:
        if self._device_override:
            return torch.device(self._device_override)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        torch, yaml, load_file = self._import_ml_stack()
        from hyper3_clip import Hyper3CLIP

        if not self._config_path.exists():
            raise FileNotFoundError(f"Hyper3-CLIP config not found: {self._config_path}")
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Hyper3-CLIP checkpoint not found: {self._checkpoint_path}")

        self._torch = torch
        self._device = self._resolve_device(torch)

        config = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        model_config = dict(config["model"])
        model_config["vision_pretrained"] = False
        model_config["text_pretrained"] = False
        model = Hyper3CLIP(**model_config)
        state = _align_text_tower_keys(load_file(self._checkpoint_path, device="cpu"))
        load_result = model.load_state_dict(state, strict=False)
        if load_result.missing_keys:
            missing = ", ".join(load_result.missing_keys[:8])
            raise RuntimeError(f"Hyper3-CLIP checkpoint missing required keys: {missing}")
        model.to(self._device)
        model.eval()
        self._model = model

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode preprocessed inputs (B, C, H, W) to embeddings (B, D)."""
        self._ensure_model()

        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        images = self._torch.from_numpy(inputs).to(device=self._device, dtype=self._torch.float32)
        with self._torch.inference_mode():
            emb = self._model.encode_image(images)

        return np.asarray(emb.detach().cpu().numpy(), dtype=np.float32)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode PIL images to Hyper3-CLIP embeddings (B, D)."""
        return self.encode(preprocess_images(images, self._image_config))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode text queries into the same hyperboloid as the images (B, D)."""
        self._ensure_model()

        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        encoded = self._model.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        with self._torch.inference_mode():
            emb = self._model.encode_text(input_ids, attention_mask)

        return np.asarray(emb.detach().cpu().numpy(), dtype=np.float32)
