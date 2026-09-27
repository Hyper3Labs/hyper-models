# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 hyper³labs. Adapted from the former haystack-hyper3 integration.
"""Haystack image document component for Hyper3-CLIP's Lorentz points."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret
from PIL import Image

from hyper_models.integrations.haystack.backend import (
    DEFAULT_HUB_ID,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Hyper3EmbeddingBackendFactory,
)


@component
class Hyper3DocumentImageEmbedder:
    """Embed document images as native 513-coordinate Hyper3-CLIP Lorentz points."""

    def __init__(
        self,
        *,
        file_path_meta_field: str = "file_path",
        root_path: str = "",
        model: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_REVISION,
        device: ComponentDevice | None = None,
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        batch_size: int = 16,
        local_files_only: bool = False,
    ) -> None:
        """Configure file metadata, bounded image batches, and SDK model loading."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path
        self.model = model
        self.revision = revision
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._backend = Hyper3EmbeddingBackendFactory.get_backend(
            model=model,
            revision=revision,
            device=self.device,
            token=token,
            local_files_only=local_files_only,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component and its configuration."""
        return default_to_dict(
            self,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
            model=self.model,
            revision=self.revision,
            device=self.device,
            token=self.token,
            batch_size=self.batch_size,
            local_files_only=self.local_files_only,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hyper3DocumentImageEmbedder:
        """Deserialize the component."""
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """Prepare the shared SDK encoder."""
        self._backend.warm_up()

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Return document copies with their image's native Lorentz embedding."""
        if not isinstance(documents, list) or any(
            not isinstance(doc, Document) for doc in documents
        ):
            raise TypeError("Hyper3DocumentImageEmbedder expects a list of Documents")
        if not documents:
            return {"documents": []}

        paths: list[Path] = []
        for document in documents:
            raw_path = document.meta.get(self.file_path_meta_field)
            if not isinstance(raw_path, (str, Path)):
                raise ValueError(
                    f"Document {document.id!r} must contain an image path in meta[{self.file_path_meta_field!r}]"
                )
            path = Path(raw_path)
            if not path.is_absolute():
                path = Path(self.root_path) / path
            try:
                with Image.open(path) as image:
                    image.verify()
            except OSError as error:
                raise ValueError(
                    f"Could not load image for document {document.id!r}: {path}"
                ) from error
            paths.append(path)

        embeddings: list[list[float]] = []
        for start in range(0, len(paths), self.batch_size):
            images: list[Image.Image] = []
            try:
                for offset, path in enumerate(paths[start : start + self.batch_size]):
                    try:
                        with Image.open(path) as source_image:
                            images.append(source_image.convert("RGB"))
                    except OSError as error:
                        document = documents[start + offset]
                        raise ValueError(
                            f"Could not load image for document {document.id!r}: {path}"
                        ) from error
                embeddings.extend(self._backend.embed_images(images))
            finally:
                for image in images:
                    image.close()

        output = []
        for document, embedding in zip(documents, embeddings, strict=True):
            meta = {
                **document.meta,
                "embedding_source": {
                    "type": "image",
                    "model": DEFAULT_HUB_ID,
                    "revision": self.revision,
                    "geometry": "lorentz",
                    "file_path_meta_field": self.file_path_meta_field,
                },
            }
            output.append(replace(document, meta=meta, embedding=embedding))
        return {"documents": output}
