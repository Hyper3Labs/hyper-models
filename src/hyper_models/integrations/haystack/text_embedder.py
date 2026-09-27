# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 hyper³labs. Adapted from the former haystack-hyper3 integration.
"""Haystack text component for Hyper3-CLIP's native Lorentz points."""

from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret

from hyper_models.integrations.haystack.backend import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Hyper3EmbeddingBackendFactory,
)


@component
class Hyper3TextEmbedder:
    """Embed text as native 513-coordinate Hyper3-CLIP Lorentz points."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_REVISION,
        device: ComponentDevice | None = None,
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        local_files_only: bool = False,
    ) -> None:
        """Configure the catalog model, pinned revision, device, and Hub access."""
        self.model = model
        self.revision = revision
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.local_files_only = local_files_only
        self._backend = Hyper3EmbeddingBackendFactory.get_backend(
            model=model,
            revision=revision,
            device=self.device,
            token=token,
            local_files_only=local_files_only,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component; Haystack handles environment-backed secrets."""
        return default_to_dict(
            self,
            model=self.model,
            revision=self.revision,
            device=self.device,
            token=self.token,
            local_files_only=self.local_files_only,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hyper3TextEmbedder:
        """Deserialize the component."""
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """Prepare the shared SDK encoder."""
        self._backend.warm_up()

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        """Embed one non-empty query as a native Lorentz point."""
        if not isinstance(text, str):
            raise TypeError("Hyper3TextEmbedder expects a string as input")
        if not text.strip():
            raise ValueError("Hyper3TextEmbedder expects non-empty text")
        return {"embedding": self._backend.embed_text(text)}
