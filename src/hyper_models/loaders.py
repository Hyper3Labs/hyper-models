"""Internal loader implementations for hyper-models catalog entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from hyper_models.models import ONNXModel
from hyper_models.registry import ModelInfo

__all__ = ["list_loaders", "load_model"]

LoaderFn = Callable[[ModelInfo, Path], Any]


def _load_onnx_model(info: ModelInfo, artifact_path: Path) -> ONNXModel:
    return ONNXModel(
        path=artifact_path,
        geometry=info.geometry,
        dim=info.dim,
        input_name=info.input_name,
        output_name=info.output_name,
        image_config=info.image_config,
    )


def _load_uncha_image_torch_model(info: ModelInfo, artifact_path: Path) -> Any:
    if info.variant is None:
        raise ValueError(f"UNCHA model '{info.name}' is missing a registry variant")

    from hyper_models.torch_models import UNCHATorchModel

    return UNCHATorchModel(
        checkpoint_path=artifact_path,
        geometry=info.geometry,
        dim=info.dim,
        variant=info.variant,
        image_config=info.image_config,
    )


def _load_hyper3_clip_torch_model(info: ModelInfo, artifact_path: Path) -> Any:
    from hyper_models.torch_models import Hyper3ClipTorchModel

    return Hyper3ClipTorchModel(
        checkpoint_path=artifact_path,
        geometry=info.geometry,
        dim=info.dim,
        image_config=info.image_config,
    )


def _load_timm_image_route(info: ModelInfo, artifact_path: Path) -> Any:
    raise ValueError(
        f"Model '{info.name}' is a timm model. "
        f"Use HyperView provider='timm-image' with model='hf-hub:{info.hub_id}' instead.\n"
        f"  dataset.compute_embeddings(model='hf-hub:{info.hub_id}', provider='timm-image')"
    )


_LOADERS: dict[str, LoaderFn] = {
    "hyper3-clip-torch": _load_hyper3_clip_torch_model,
    "onnx": _load_onnx_model,
    "uncha-image-torch": _load_uncha_image_torch_model,
    "timm-image": _load_timm_image_route,
}


def list_loaders() -> list[str]:
    """List supported internal loader kinds for catalog entries."""
    return sorted(_LOADERS)


def load_model(info: ModelInfo, artifact_path: Path) -> Any:
    """Instantiate a catalog entry using the loader declared in ``ModelInfo``."""
    try:
        loader = _LOADERS[info.loader]
    except KeyError:
        available = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unsupported loader '{info.loader}' for model '{info.name}'. Available: {available}"
        ) from None

    return loader(info, artifact_path)
