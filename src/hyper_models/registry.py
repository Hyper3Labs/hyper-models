"""Catalog registry for hyper-models entries and their loader metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from hyper_models.preprocessing import ImageConfig

__all__ = ["ModelInfo", "list_models", "get_model_info"]


@dataclass
class ModelInfo:
    """Metadata for a registered catalog model.

    `loader` describes how the catalog entry is instantiated internally.
    This is intentionally separate from the public catalog surface so callers
    can always use ``hyper_models.load(name)`` regardless of runtime backend.
    """

    name: str
    geometry: str  # 'hyperboloid', 'poincare', 'sphere', 'euclidean'
    dim: int
    hub_id: str
    hub_path: str
    license: str
    hub_patterns: tuple[str, ...] | None = None
    loader: str = "onnx"  # e.g. 'onnx', 'uncha-image-torch'
    variant: str | None = None  # optional loader hint (e.g., 'vit_s', 'vit_b')
    optional_dependencies: tuple[str, ...] = ()
    description: str = ""
    input_name: str = "image"
    output_name: str | None = None
    modalities: tuple[str, ...] = ("image",)  # what the loaded model can encode
    image_config: ImageConfig = field(default_factory=ImageConfig)


_MODELS: dict[str, ModelInfo] = {
    "hycoclip-vit-s": ModelInfo(
        name="hycoclip-vit-s",
        geometry="hyperboloid",
        dim=513,
        hub_id="mnm-matin/hyperbolic-clip",
        hub_path="hycoclip-vit-s/model.onnx",
        license="CC-BY-NC",
        description="HyCoCLIP ViT-Small (512D hyperboloid)",
        output_name="embedding_hyperboloid",
    ),
    "hycoclip-vit-b": ModelInfo(
        name="hycoclip-vit-b",
        geometry="hyperboloid",
        dim=513,
        hub_id="mnm-matin/hyperbolic-clip",
        hub_path="hycoclip-vit-b/model.onnx",
        license="CC-BY-NC",
        description="HyCoCLIP ViT-Base (512D hyperboloid)",
        output_name="embedding_hyperboloid",
    ),
    "meru-vit-s": ModelInfo(
        name="meru-vit-s",
        geometry="hyperboloid",
        dim=513,
        hub_id="mnm-matin/hyperbolic-clip",
        hub_path="meru-vit-s/model.onnx",
        license="CC-BY-NC",
        description="MERU ViT-Small (512D hyperboloid)",
        output_name="embedding_hyperboloid",
    ),
    "meru-vit-b": ModelInfo(
        name="meru-vit-b",
        geometry="hyperboloid",
        dim=513,
        hub_id="mnm-matin/hyperbolic-clip",
        hub_path="meru-vit-b/model.onnx",
        license="CC-BY-NC",
        description="MERU ViT-Base (512D hyperboloid)",
        output_name="embedding_hyperboloid",
    ),
    "uncha-vit-s": ModelInfo(
        name="uncha-vit-s",
        geometry="hyperboloid",
        dim=513,
        hub_id="hayeonkim/uncha",
        hub_path="uncha_vit_s.pth",
        license="Unknown",
        loader="uncha-image-torch",
        variant="vit_s",
        optional_dependencies=("ml",),
        description="UNCHA ViT-S/16 checkpoint (HF .pth, torch inference)",
    ),
    "uncha-vit-b": ModelInfo(
        name="uncha-vit-b",
        geometry="hyperboloid",
        dim=513,
        hub_id="hayeonkim/uncha",
        hub_path="uncha_vit_b.pth",
        license="Unknown",
        loader="uncha-image-torch",
        variant="vit_b",
        optional_dependencies=("ml",),
        description="UNCHA ViT-B/16 checkpoint (HF .pth, torch inference)",
    ),
    "hyper3-clip-v1": ModelInfo(
        name="hyper3-clip-v1",
        geometry="hyperboloid",
        dim=513,
        hub_id="hyper3labs/hyper3-clip-v1",
        hub_path="model.safetensors",
        hub_patterns=("config.yaml", "model.safetensors"),
        license="Unknown",
        loader="hyper3-clip-torch",
        optional_dependencies=("ml",),
        modalities=("image", "text"),
        description="Hyper3-CLIP v1 ViT-B image+text encoder (HF safetensors, torch inference)",
        image_config=ImageConfig(
            size=224,
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ),
    "megadescriptor": ModelInfo(
        name="megadescriptor",
        geometry="sphere",
        dim=1024,
        hub_id="BVRA/MegaDescriptor-L-384",
        hub_path="",
        license="MIT",
        loader="timm-image",
        description="MegaDescriptor-L-384 (via timm, use provider='timm-image')",
        image_config=ImageConfig(size=384),
    ),
}

_ALIASES = {
    # Compatibility for callers that selected the model before its v1 rename.
    "hyper3-clip-v0.5": "hyper3-clip-v1",
}


def list_models(geometry: str | None = None) -> list[str]:
    """List available model names, optionally filtered by geometry."""
    if geometry is None:
        return list(_MODELS.keys())
    return [name for name, info in _MODELS.items() if info.geometry == geometry]


def get_model_info(name: str) -> ModelInfo:
    """Get metadata for a model. Raises KeyError if not found."""
    name = _ALIASES.get(name, name)
    if name not in _MODELS:
        raise KeyError(f"Model '{name}' not found. Available: {', '.join(_MODELS.keys())}")
    return _MODELS[name]
