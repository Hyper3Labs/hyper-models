## 0.3.2 - 2026-09-06

### Changed
- Rename the public Hyper3-CLIP catalog entry and Hub target to `hyper3-clip-v1`.
- Preserve `hyper3-clip-v0.5` as a hidden compatibility alias for existing callers.

## 0.3.1 - 2026-08-30

### Fixes
- Load the Hyper3-CLIP v0.5 text tower. The published checkpoint stores it under
  the full Hugging Face `CLIPTextModel` path while `Hyper3CLIP` binds its
  backbone one level deeper, and the loader tolerated the resulting "missing"
  keys. Every text weight was silently dropped, leaving a randomly initialised
  text encoder that still returned embeddings.
- Report the packaged version. `__version__` was a literal that stayed at
  "0.3.0"; it is now read from the installed distribution metadata.
- Install on Python 3.10 again. onnxruntime stopped publishing cp310 wheels
  after 1.23.x, so a universal resolution selected a release that could not be
  installed on a version this package claims to support.

### Features
- Add `Hyper3ClipTorchModel.encode_texts` for text queries in the image
  hyperboloid.
- Add `ModelInfo.modalities` so callers can discover which inputs a catalog
  entry encodes; `hyper3-clip-v0.5` now declares `("image", "text")`.

## 0.3.0 - 2026-06-04

### Features
- Add `hyper3-clip-v0.5` as a torch-backed Hyper3-CLIP catalog entry.
- Add the `hyper3-clip-torch` internal loader for Hyper3-CLIP safetensors checkpoints.
- Extend the `ml` extra with the transformer and safetensors dependencies needed by Hyper3-CLIP.

## 0.2.0 - 2026-04-12

### Features
- Add UNCHA catalog entries for `uncha-vit-s` and `uncha-vit-b`
- Introduce an internal loader abstraction so catalog entries can route to ONNX or optional torch-backed runtimes behind one public `hyper_models.load(...)` API
- Add the optional `ml` extra for torch-backed catalog entries

### Documentation
- Clarify that hyper-models is a timm-like catalog for non-Euclidean models
- Document the torch-free default install path and how HyperView uses hyper-models catalog entries
