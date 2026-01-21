"""Convenience constructors for HyperView `ModelSpec` objects.

This module intentionally lives outside the `hyperview` package:
- HyperView stays focused on visualization + runtime
- hyper-models can grow independently and later become a separate PyPI package

Usage (from repo root):

    import hyperview as hv
    from hyper_models import model_zoo as mz

    ds = hv.Dataset("my_dataset", persist=False)
    ds.add_images_dir("/path/to/images")

    spec = mz.hycoclip_onnx(onnx_path="/path/to/hycoclip_vit_s_image_encoder.onnx")
    ds.compute_embeddings(spec)

Note:
- ONNX exports that use external data require keeping `.onnx` and `.onnx.data`
  together in the same folder.
"""

from __future__ import annotations

# ruff: noqa: I001

from hyperview.embeddings.providers import ModelSpec


HYCOCLIP_TORCH_VARIANTS: tuple[str, ...] = (
    "hycoclip_vit_s",
    "hycoclip_vit_b",
    "hycoclip_vit_l",
    "meru_vit_s",
    "meru_vit_b",
)


def hycoclip_torch(
    *,
    model_id: str = "hycoclip_vit_s",
    checkpoint: str,
    config_path: str | None = None,
) -> ModelSpec:
    """HyCoCLIP/MERU via torch+hycoclip (produces hyperboloid embeddings)."""
    if model_id not in HYCOCLIP_TORCH_VARIANTS:
        raise ValueError(f"Unknown HyCoCLIP torch variant: {model_id}. Known: {HYCOCLIP_TORCH_VARIANTS}")

    return ModelSpec(
        provider="hycoclip",
        model_id=model_id,
        checkpoint=checkpoint,
        config_path=config_path,
        output_geometry="hyperboloid",
    )


def hycoclip_onnx(*, model_id: str = "hycoclip_vit_s", onnx_path: str) -> ModelSpec:
    """HyCoCLIP image encoder via ONNX Runtime (produces hyperboloid embeddings).

    Note: we reuse `ModelSpec.checkpoint` to store the ONNX path.
    """
    return ModelSpec(
        provider="hycoclip_onnx",
        model_id=model_id,
        checkpoint=onnx_path,
        output_geometry="hyperboloid",
    )
