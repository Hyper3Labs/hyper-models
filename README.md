# hyper-models

<p align="center">
  <strong>A model zoo for non-Euclidean embedding models</strong>
  <br>
  <em>Hyperbolic · Spherical · Product Manifolds</em>
</p>

<p align="center">
  <a href="https://huggingface.co/mnm-matin/hyperbolic-clip">
    <img src="https://img.shields.io/badge/🤗_Models-hyperbolic--clip-orange" alt="Hugging Face">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT">
  </a>
</p>

---

## Why?

- **Standardized access** to non-Euclidean embedding models
- **One catalog surface**: model names map to internal loaders such as ONNX or optional torch-backed runtimes
- **Simple API** — `load()` and `encode_images()`

## Installation

```bash
uv pip install hyper-models
```

This base install is the simple path: it stays **torch-free** and is enough for
ONNX-backed catalog entries such as HyCoCLIP and MERU.

For torch-backed checkpoints (for example UNCHA and Hyper3-CLIP):

```bash
uv pip install "hyper-models[ml]"
```

## Usage

```python
import hyper_models
from PIL import Image

# List available models
hyper_models.list_models()
# ['hycoclip-vit-s', 'hycoclip-vit-b', 'meru-vit-s', 'meru-vit-b', 'uncha-vit-s', 'uncha-vit-b', 'hyper3-clip-v1']

# Inspect supported internal loader kinds
hyper_models.list_loaders()
# ['hyper3-clip-torch', 'onnx', 'uncha-image-torch']

# Load model (auto-downloads from Hugging Face Hub)
model = hyper_models.load("hycoclip-vit-s")
model.geometry  # 'hyperboloid'
model.dim  # 513

# Encode PIL images
images = [Image.open("image.jpg")]
embeddings = model.encode_images(images)  # (1, 513) ndarray

# Get model info
info = hyper_models.get_model_info("hycoclip-vit-s")
info.hub_id  # 'mnm-matin/hyperbolic-clip'
info.loader  # 'onnx'
info.license  # 'CC-BY-NC'

# Low-level: preprocess images yourself
batch = hyper_models.preprocess_images(images)  # (B, 3, 224, 224)
embeddings = model.encode(batch)
```

### Architecture

`hyper-models` is intended to be a timm-like catalog for non-Euclidean models.

- The public abstraction is the catalog entry name, for example `hycoclip-vit-s`.
- Each entry declares metadata such as geometry, dimensionality, artifact path,
  and an internal loader kind.
- Internal loaders may differ by model family:
  - `onnx` for exported, torch-free runtimes
  - `uncha-image-torch` for raw checkpoints that need a PyTorch image runtime
  - `hyper3-clip-torch` for Hyper3-CLIP safetensors checkpoints

This keeps callers on one stable API:

```python
model = hyper_models.load("hycoclip-vit-s")
model = hyper_models.load("uncha-vit-b")
model = hyper_models.load("hyper3-clip-v1")
```

Callers do not need to know which internal loader is used, except for optional
dependency installation when choosing entries that need `hyper-models[ml]`.

For `hyper3-clip-v1`, `encode_images(images)` and `encode_texts(texts)` return
513-coordinate Lorentz embeddings in the same space. The loader downloads the
model's runtime configuration, weights, and tokenizer together. Complete the
model's Hugging Face access form and run `hf auth login` before the first download.

### HyperView integration

HyperView auto-detects `hyper-models` names and routes them to the `hyper-models` provider.

```python
import hyperview as hv

dataset = hv.Dataset.from_huggingface(
    name="demo",
    hf_dataset="uoft-cs/cifar10",
    split="train",
    image_key="img",
)

# Uses provider='hyper-models' automatically.
space_key = dataset.compute_embeddings(model="uncha-vit-b")
layout_key = dataset.compute_visualization(space_key=space_key, layout="poincare")
```

HyperView's simple path remains torch-free. If you use the default ONNX-backed
`hyper-models` entries or the default `embed-anything` provider, HyperView does
not need PyTorch. PyTorch is only needed when you explicitly select a
torch-backed catalog entry such as `uncha-vit-s`, `uncha-vit-b`, or
`hyper3-clip-v1`.

## Models

### Hyperbolic

| Model | Available | Paper | Code |
|-------|:---------:|-------|------|
| `hycoclip-vit-s` | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/mnm-matin/hyperbolic-clip/tree/main/hycoclip-vit-s) | [ICLR 2025](https://arxiv.org/abs/2410.06912) | [PalAvik/hycoclip](https://github.com/PalAvik/hycoclip) |
| `hycoclip-vit-b` | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/mnm-matin/hyperbolic-clip/tree/main/hycoclip-vit-b) | [ICLR 2025](https://arxiv.org/abs/2410.06912) | [PalAvik/hycoclip](https://github.com/PalAvik/hycoclip) |
| `meru-vit-s` | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/mnm-matin/hyperbolic-clip/tree/main/meru-vit-s) | [ICML 2023](https://arxiv.org/abs/2304.09172) | [facebookresearch/meru](https://github.com/facebookresearch/meru) |
| `meru-vit-b` | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/mnm-matin/hyperbolic-clip/tree/main/meru-vit-b) | [ICML 2023](https://arxiv.org/abs/2304.09172) | [facebookresearch/meru](https://github.com/facebookresearch/meru) |
| `uncha-vit-s` | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/hayeonkim/uncha/blob/main/uncha_vit_s.pth) | [CVPR 2026](https://arxiv.org/abs/2603.22042) | [jeeit17/UNCHA](https://github.com/jeeit17/UNCHA) |
| `uncha-vit-b` | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/hayeonkim/uncha/blob/main/uncha_vit_b.pth) | [CVPR 2026](https://arxiv.org/abs/2603.22042) | [jeeit17/UNCHA](https://github.com/jeeit17/UNCHA) |
| `hyper3-clip-v1` | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/hyper3labs/hyper3-clip-v1) | — | [Hyper3Labs/hyper3-clip](https://github.com/Hyper3Labs/hyper3-clip) |
| `hyp-vit` | — | [CVPR 2022](https://arxiv.org/abs/2203.10833) | [htdt/hyp_metric](https://github.com/htdt/hyp_metric) |
| `hie` | — | [CVPR 2020](https://arxiv.org/abs/1904.02239) | [leymir/hyperbolic-image-embeddings](https://github.com/leymir/hyperbolic-image-embeddings) |
| `hcnn` | — | [ICLR 2024](https://openreview.net/forum?id=ekz1hN5QNh) | [kschwethelm/HyperbolicCV](https://github.com/kschwethelm/HyperbolicCV) |

### Hyperspherical

| Model            | Available | Paper | Code |
|------------------|:---------:|-------|------|
| `megadescriptor` (via timm) | [![HF](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/BVRA/MegaDescriptor-L-384) | [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.pdf) | [WildlifeDatasets/wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets) |
| `sphereface`     | — | [CVPR 2017](https://arxiv.org/abs/1704.08063) | [wy1iu/sphereface](https://github.com/wy1iu/sphereface) |
| `arcface`       | — | [CVPR 2019](https://arxiv.org/abs/1801.07698) | [deepinsight/insightface](https://github.com/deepinsight/insightface) |


### Product Manifolds

| Model | Available | Paper | Code |
|-------|:---------:|-------|------|
| `hyperbolics` | — | [ICLR 2019](https://openreview.net/forum?id=HJxeWnCcF7) | [HazyResearch/hyperbolics](https://github.com/HazyResearch/hyperbolics) |

## Export Tooling

This repo also contains tooling to export PyTorch models to ONNX:

```bash
cd export/hycoclip
uv run python export_onnx.py --checkpoint model.pth --onnx model.onnx
```

See [export/hycoclip/README.md](export/hycoclip/README.md) for details.

## References

- [HyCoCLIP](https://github.com/PalAvik/hycoclip)
- [MERU](https://github.com/facebookresearch/meru)
- [geoopt](https://github.com/geoopt/geoopt)
