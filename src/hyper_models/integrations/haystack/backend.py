# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 hyper³labs. Adapted from the former haystack-hyper3 integration.
"""Lazy, shared access to the hyper-models Hyper3-CLIP encoder."""

from __future__ import annotations

import hashlib
import threading
from typing import Any, ClassVar

import numpy as np
from haystack.utils import ComponentDevice, Secret
from huggingface_hub.errors import GatedRepoError
from PIL import Image

import hyper_models

DEFAULT_MODEL = "hyper3-clip-v1"
DEFAULT_HUB_ID = "hyper3labs/hyper3-clip-v1"
DEFAULT_REVISION = "12a8d89022cec75a3fbac91047683231cbc0fa82"
_MODEL_ALIASES = {DEFAULT_HUB_ID: DEFAULT_MODEL}


class Hyper3EmbeddingBackend:
    """Load the SDK model once and encode both modalities through its public API."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None,
        device: ComponentDevice,
        token: Secret | None,
        local_files_only: bool,
    ) -> None:
        model = _MODEL_ALIASES.get(model, model)
        if model != DEFAULT_MODEL:
            raise ValueError(f"Haystack embedders currently support only {DEFAULT_MODEL!r}")
        if device.has_multiple_devices:
            raise ValueError("Hyper3 embedders support one device per component")
        self.model_name = model
        self.revision = revision
        self.device = device
        self.token = token
        self.local_files_only = local_files_only
        self.model: Any | None = None
        self._lock = threading.RLock()

    def warm_up(self) -> None:
        """Load the pinned model and initialize its image and text towers."""
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            try:
                model = hyper_models.load(
                    self.model_name,
                    revision=self.revision,
                    token=self.token.resolve_value() if self.token else None,
                    device=self.device.to_torch_str(),
                    local_files_only=self.local_files_only,
                )
            except GatedRepoError as error:
                raise RuntimeError(
                    "Request access at https://huggingface.co/hyper3labs/hyper3-clip-v1 "
                    "and authenticate with hf auth login, HF_TOKEN, or HF_API_TOKEN."
                ) from error
            if model.geometry != "hyperboloid" or model.dim != 513:
                raise ValueError("Hyper3-CLIP must produce 513-coordinate Lorentz embeddings")
            model.warm_up()
            self.model = model

    def embed_text(self, text: str) -> list[float]:
        """Return the native Lorentz point for one text query."""
        self.warm_up()
        with self._lock:
            assert self.model is not None
            embeddings = np.asarray(self.model.encode_texts([text]))
        if embeddings.shape != (1, 513):
            raise ValueError(f"Expected one 513-coordinate text embedding, got {embeddings.shape}")
        return embeddings[0].tolist()

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Return native Lorentz points for one bounded image batch."""
        self.warm_up()
        with self._lock:
            assert self.model is not None
            embeddings = np.asarray(self.model.encode_images(images))
        if embeddings.shape != (len(images), 513):
            raise ValueError(
                f"Expected {len(images)} 513-coordinate image embeddings, got {embeddings.shape}"
            )
        return embeddings.tolist()


class Hyper3EmbeddingBackendFactory:
    """Share compatible model instances across image and text components."""

    _instances: ClassVar[
        dict[tuple[str, str | None, str, str | None, bool], Hyper3EmbeddingBackend]
    ] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_backend(
        cls,
        *,
        model: str,
        revision: str | None,
        device: ComponentDevice,
        token: Secret | None,
        local_files_only: bool,
    ) -> Hyper3EmbeddingBackend:
        """Return a backend keyed without retaining the raw authentication token."""
        resolved_token = token.resolve_value() if token else None
        fingerprint = (
            hashlib.sha256(resolved_token.encode()).hexdigest() if resolved_token else None
        )
        canonical_model = _MODEL_ALIASES.get(model, model)
        key = (canonical_model, revision, device.to_torch_str(), fingerprint, local_files_only)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = Hyper3EmbeddingBackend(
                    model=canonical_model,
                    revision=revision,
                    device=device,
                    token=token,
                    local_files_only=local_files_only,
                )
            return cls._instances[key]
