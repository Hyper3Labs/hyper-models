"""ONNX model wrapper for hyper-models."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from hyper_models.preprocessing import ImageConfig, preprocess_images

__all__ = ["ONNXModel"]


class ONNXModel:
    """ONNX Runtime model wrapper for embedding inference."""

    def __init__(
        self,
        path: Path,
        geometry: str,
        dim: int,
        *,
        input_name: str = "image",
        output_name: str | None = None,
        image_config: ImageConfig | None = None,
    ) -> None:
        self._path = path
        self.geometry = geometry
        self.dim = dim
        self._input_name = input_name
        self._output_name = output_name
        self._image_config = image_config or ImageConfig()
        self._session = None
        self._artifact_directory = None

    def _ensure_session(self) -> None:
        if self._session is None:
            import onnxruntime as ort

            path = self._path
            # Hub snapshots link the graph and external weights to different
            # blob directories. New ONNX Runtime versions correctly reject
            # external data outside the resolved graph directory. Materialize
            # our catalog's graph and sidecars together, keeping validation on.
            artifacts = [path, *path.parent.glob(f"{path.name}.*")]
            if any(artifact.is_symlink() for artifact in artifacts):
                self._artifact_directory = TemporaryDirectory(prefix="hyper-models-onnx-")
                directory = Path(self._artifact_directory.name)
                for artifact in artifacts:
                    if artifact.is_file():
                        copyfile(artifact, directory / artifact.name)
                path = directory / path.name
            self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode preprocessed inputs (B, C, H, W) to embeddings (B, D)."""
        self._ensure_session()
        outputs = self._session.run(None, {self._input_name: inputs})

        if self._output_name:
            output_names = [o.name for o in self._session.get_outputs()]
            return np.asarray(outputs[output_names.index(self._output_name)], dtype=np.float32)

        return np.asarray(outputs[0], dtype=np.float32)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode PIL images to embeddings (B, D)."""
        return self.encode(preprocess_images(images, self._image_config))
