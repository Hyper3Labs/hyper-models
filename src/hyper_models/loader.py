"""Public catalog loading entrypoint for hyper-models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from hyper_models.loaders import load_model
from hyper_models.registry import get_model_info

__all__ = ["load"]


def load(name: str, *, local_path: str | Path | None = None) -> Any:
    """Load a model by name.

    Args:
        name: Catalog model name (e.g., 'hycoclip-vit-s').
        local_path: Optional local artifact path (skips Hub download).

    Returns:
        Model instance ready for inference.

    Example:
        >>> model = hyper_models.load("hycoclip-vit-s")
        >>> embeddings = model.encode_images([Image.open("photo.jpg")])
    """
    info = get_model_info(name)

    # Route-only entries (e.g. timm-image) raise immediately without downloading.
    if info.loader == "timm-image":
        return load_model(info, Path())

    if local_path is None:
        allow_patterns = list(info.hub_patterns or (f"{info.hub_path}*",))
        local_dir = snapshot_download(
            info.hub_id,
            allow_patterns=allow_patterns,
            token=os.environ.get("HF_TOKEN"),
        )
        artifact_path = Path(local_dir) / info.hub_path
    else:
        artifact_path = Path(local_path)

    return load_model(info, artifact_path)
