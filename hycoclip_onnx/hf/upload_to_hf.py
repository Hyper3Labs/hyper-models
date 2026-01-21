"""Upload exported ONNX artifacts to the Hugging Face Hub.

Uploads the ONNX model with a proper model card (README.md) containing
YAML metadata for discoverability on Hugging Face.

Example:

    cd hyper_models/hycoclip_onnx
    uv run python hf/upload_to_hf.py \
        --repo-id HackerRoomAI/hycoclip-vit-s-onnx \
        --onnx ./outputs/hycoclip_vit_s_image_encoder.onnx \
        --onnx-data ./outputs/hycoclip_vit_s_image_encoder.onnx.data \
        --variant vit_s

Requirements:
- huggingface_hub (install inside the hycoclip_onnx environment: uv add huggingface-hub)
- You must be logged in: huggingface-cli login

What gets uploaded to Hugging Face:
- README.md (model card with YAML metadata)
- onnx/model.onnx (the ONNX model)
- onnx/model.onnx.data (external weights, if present)

Important:
- HyCoCLIP/MERU is CC-BY-NC (non-commercial). Respect upstream licensing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Model card template with YAML frontmatter for Hugging Face
# ---------------------------------------------------------------------------
MODEL_CARD_TEMPLATE = '''---
library_name: onnx
pipeline_tag: feature-extraction
license: cc-by-nc-4.0
base_model: PalAvik/hycoclip
tags:
  - onnx
  - vision
  - clip
  - hyperbolic
  - image-embedding
  - hyperboloid
  - non-euclidean
  - lorentz
  - meru
language:
  - en
---

# HyCoCLIP {variant_display} Image Encoder (ONNX)

This is the **ONNX export** of the [HyCoCLIP/MERU](https://github.com/PalAvik/hycoclip) image encoder for **hyperbolic image embeddings**.

## Model Description

| Property | Value |
|----------|-------|
| **Model type** | ONNX (Vision Transformer image encoder) |
| **Variant** | {variant_display} |
| **Geometry** | Hyperbolic (Lorentz/Hyperboloid model) |
| **Input** | `(batch, 3, 224, 224)` float32 images normalized to `[0, 1]` |
| **Output** | Hyperboloid coordinates `(t, x₁...xₙ)` where `t = √(1/c + ‖x‖²)` |
| **Embedding dim** | 513 (1 time component + 512 spatial) |
| **Curvature** | Learned (exported as secondary output) |
| **License** | CC-BY-NC-4.0 (Non-commercial use only) |

## Intended Uses

- Hyperbolic image embeddings for [HyperView](https://github.com/HackerRoomAI/HyperView) visualization
- Hierarchical image similarity and retrieval
- Non-Euclidean representation learning research

## Usage with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download

# Download model
onnx_path = hf_hub_download(repo_id="{repo_id}", filename="onnx/model.onnx")

# Load model
session = ort.InferenceSession(onnx_path)

# Prepare input: (batch, 3, 224, 224) float32 in [0, 1]
# Apply standard ImageNet preprocessing (resize, center crop, normalize)
image = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Run inference
embedding_hyperboloid, curvature = session.run(None, {{"image": image}})

print(f"Embedding shape: {{embedding_hyperboloid.shape}}")  # (1, 513)
print(f"Curvature: {{curvature[0]:.4f}}")
```

## Usage with HyperView

```python
import hyperview as hv
from hyper_models import model_zoo as mz

# Load dataset
ds = hv.Dataset("my_images")
ds.add_images_dir("/path/to/images")

# Use the ONNX model for embeddings
spec = mz.hycoclip_onnx(
    model_id="hycoclip_{variant}",
    hf_repo="{repo_id}"
)
ds.compute_embeddings(spec)

# Visualize in hyperbolic space
hv.show(ds)
```

## Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `embedding_hyperboloid` | `(batch, 513)` | Hyperboloid coordinates `(t, x₁...x₅₁₂)` |
| `curvature` | `(1,)` | Learned curvature parameter `c` |

### Converting to Poincaré Ball (for visualization)

```python
# Hyperboloid (t, x) → Poincaré ball
t = embedding_hyperboloid[:, 0:1]  # time component
x = embedding_hyperboloid[:, 1:]   # spatial components
poincare = x / (t + 1)             # stereographic projection
```

## Technical Details

- **Architecture**: Vision Transformer (ViT) with MoCo-v3 initialization
- **Training**: Contrastive learning with hyperbolic entailment loss
- **Curvature**: Learned during training, exported as model output
- **ONNX opset**: 18
- **Dynamic batching**: Supported (batch dimension is dynamic)

## Limitations

- **Non-commercial use only** (CC-BY-NC license)
- Optimized for natural images; may not generalize to other domains
- Requires standard ImageNet preprocessing (resize to 224×224, normalize)

## Citation

```bibtex
@inproceedings{{desai2023hyperbolic,
  title={{Hyperbolic Image-Text Representations}},
  author={{Desai, Karan and Nickel, Maximilian and Rajpurohit, Tanmay and Johnson, Justin and Vedantam, Ramakrishna}},
  booktitle={{ICML}},
  year={{2023}}
}}

@article{{hycoclip2024,
  title={{HyCoCLIP: Hyperbolic Contrastive Language-Image Pre-training}},
  author={{Avik Pal and others}},
  year={{2024}}
}}
```

## Acknowledgments

- Based on [PalAvik/hycoclip](https://github.com/PalAvik/hycoclip)
- Original MERU paper: [facebookresearch/meru](https://github.com/facebookresearch/meru)
- ONNX export by [HyperView](https://github.com/HackerRoomAI/HyperView) team
'''


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload ONNX artifacts with model card to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
What gets uploaded:
  - README.md         Model card with YAML metadata for discoverability
  - onnx/model.onnx   The ONNX model file
  - onnx/model.onnx.data  External weights (if --onnx-data provided)

The model card includes proper metadata (library_name: onnx, pipeline_tag,
license, tags) so the model appears correctly in Hugging Face search/filters.
        """,
    )
    p.add_argument("--repo-id", required=True, help="Target HF repo, e.g. HackerRoomAI/hycoclip-vit-s-onnx")
    p.add_argument("--onnx", required=True, help="Path to .onnx file")
    p.add_argument("--onnx-data", default=None, help="Path to .onnx.data file (external weights)")
    p.add_argument(
        "--variant",
        choices=["vit_s", "vit_b", "vit_l"],
        default="vit_s",
        help="Model variant for documentation (default: vit_s)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    return p.parse_args()


def _variant_display(variant: str) -> str:
    """Convert variant to display name."""
    return {"vit_s": "ViT-S", "vit_b": "ViT-B", "vit_l": "ViT-L"}.get(variant, variant.upper())


def main() -> int:
    args = _parse_args()

    # Validate paths
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.exists():
        raise SystemExit(f"ONNX file not found: {onnx_path}")
    if onnx_path.suffix.lower() != ".onnx":
        raise SystemExit(f"Expected .onnx file, got: {onnx_path}")

    data_path: Path | None = None
    if args.onnx_data:
        data_path = Path(args.onnx_data).expanduser().resolve()
        if not data_path.exists():
            raise SystemExit(f"ONNX data file not found: {data_path}")

    # Generate model card
    variant_display = _variant_display(args.variant)
    readme_content = MODEL_CARD_TEMPLATE.format(
        variant=args.variant,
        variant_display=variant_display,
        repo_id=args.repo_id,
    )

    # Show what will be uploaded
    print("=" * 60)
    print("Hugging Face Upload Summary")
    print("=" * 60)
    print(f"Repository:  https://huggingface.co/{args.repo_id}")
    print(f"Private:     {args.private}")
    print(f"Variant:     {variant_display}")
    print()
    print("Files to upload:")
    print(f"  - README.md (model card, {len(readme_content)} bytes)")
    print(f"  - onnx/model.onnx ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")
    if data_path:
        print(f"  - onnx/model.onnx.data ({data_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would upload the above files. Model card preview:\n")
        print(readme_content[:1500] + "\n...[truncated]...")
        return 0

    # Import huggingface_hub
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install with: uv add huggingface-hub\n"
            "Then login with: huggingface-cli login"
        ) from exc

    api = HfApi()

    # Create repo if needed
    print("\nCreating repository (if needed)...")
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    # Upload README.md (model card)
    print("Uploading README.md (model card)...")
    api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Add model card with ONNX metadata",
    )

    # Upload ONNX file(s) to onnx/ subdirectory
    print("Uploading ONNX model...")
    api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=str(onnx_path),
        path_in_repo="onnx/model.onnx",
        commit_message="Add ONNX model",
    )

    if data_path is not None:
        print("Uploading ONNX external weights...")
        api.upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=str(data_path),
            path_in_repo="onnx/model.onnx.data",
            commit_message="Add ONNX external weights",
        )

    print()
    print("✅ Successfully uploaded to Hugging Face!")
    print(f"   https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
