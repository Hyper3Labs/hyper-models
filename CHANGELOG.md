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
