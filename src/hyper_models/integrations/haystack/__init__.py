"""Haystack components for Hyper3-CLIP's native Lorentz embeddings.

Requires the optional ``hyper-models[ml,haystack]`` dependencies.
"""

try:
    from hyper_models.integrations.haystack.document_image_embedder import (
        Hyper3DocumentImageEmbedder,
    )
    from hyper_models.integrations.haystack.text_embedder import Hyper3TextEmbedder
except ModuleNotFoundError as error:
    if error.name == "haystack":
        raise ImportError(
            "Install the optional integration with: pip install 'hyper-models[ml,haystack]'"
        ) from error
    raise

__all__ = ["Hyper3DocumentImageEmbedder", "Hyper3TextEmbedder"]
