"""hyper-models: A model zoo for non-Euclidean embedding models.

Hyperbolic, spherical, and product manifold models exposed through one catalog
surface, with internal loaders such as ONNX and optional torch-backed runtimes.

Example:
    >>> import hyper_models
    >>> model = hyper_models.load("hycoclip-vit-s")
    >>> embeddings = model.encode_images([Image.open("photo.jpg")])
    >>> model.geometry  # 'hyperboloid'
    >>> model.dim       # 513
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

from hyper_models.loader import load
from hyper_models.loaders import list_loaders
from hyper_models.models import ONNXModel
from hyper_models.preprocessing import ImageConfig, preprocess_images
from hyper_models.registry import ModelInfo, get_model_info, list_models

__all__ = [
    "load",
    "list_loaders",
    "list_models",
    "get_model_info",
    "ModelInfo",
    "ONNXModel",
    "ImageConfig",
    "preprocess_images",
]
try:
    # Read the installed distribution so this cannot drift from pyproject.toml:
    # a hand-maintained literal here stayed at 0.3.0 through the 0.3.1 release,
    # and the demo Dockerfiles print this value to confirm what they installed.
    __version__ = _installed_version("hyper-models")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"
