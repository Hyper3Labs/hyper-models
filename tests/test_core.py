"""Basic tests for hyper-models."""

import numpy as np
import pytest
from PIL import Image

import hyper_models


class TestRegistry:
    def test_list_models(self):
        models = hyper_models.list_models()
        assert len(models) >= 4
        assert "hycoclip-vit-s" in models
        assert "hyper3-clip-v1" in models

    def test_list_loaders(self):
        loaders = hyper_models.list_loaders()
        assert "onnx" in loaders
        assert "uncha-image-torch" in loaders
        assert "hyper3-clip-torch" in loaders

    def test_list_models_filter(self):
        hyperbolic = hyper_models.list_models(geometry="hyperboloid")
        assert "hyper3-clip-v1" in hyperbolic
        assert all(hyper_models.get_model_info(m).geometry == "hyperboloid" for m in hyperbolic)

    def test_get_model_info(self):
        info = hyper_models.get_model_info("hycoclip-vit-s")
        assert info.name == "hycoclip-vit-s"
        assert info.geometry == "hyperboloid"
        assert info.dim == 513

    def test_get_model_info_not_found(self):
        with pytest.raises(KeyError):
            hyper_models.get_model_info("not-a-model")

    def test_uncha_model_info(self):
        info = hyper_models.get_model_info("uncha-vit-s")
        assert info.geometry == "hyperboloid"
        assert info.dim == 513
        assert info.loader == "uncha-image-torch"
        assert info.optional_dependencies == ("ml",)
        assert info.hub_id == "hayeonkim/uncha"
        assert info.hub_path.endswith(".pth")

    def test_hyper3_clip_model_info(self):
        info = hyper_models.get_model_info("hyper3-clip-v1")
        assert info.geometry == "hyperboloid"
        assert info.dim == 513
        assert info.loader == "hyper3-clip-torch"
        assert info.optional_dependencies == ("ml",)
        assert info.hub_id == "hyper3labs/hyper3-clip-v1"
        assert info.hub_patterns == ("config.yaml", "model.safetensors")

    def test_hyper3_clip_legacy_alias(self):
        info = hyper_models.get_model_info("hyper3-clip-v0.5")
        assert info.name == "hyper3-clip-v1"


class TestPreprocessing:
    def test_preprocess_images(self):
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        batch = hyper_models.preprocess_images([img])
        assert batch.shape == (1, 3, 224, 224)
        assert batch.dtype == np.float32

    def test_preprocess_images_custom_config(self):
        img = Image.new("RGB", (512, 512))
        config = hyper_models.ImageConfig(size=384)
        batch = hyper_models.preprocess_images([img], config=config)
        assert batch.shape == (1, 3, 384, 384)

    def test_preprocess_grayscale(self):
        """Grayscale images should be converted to RGB."""
        img = Image.new("L", (224, 224))
        batch = hyper_models.preprocess_images([img])
        assert batch.shape == (1, 3, 224, 224)


class TestModel:
    @pytest.fixture
    def model(self):
        """Load model once for all tests in this class."""
        return hyper_models.load("hycoclip-vit-s")

    def test_load_model(self, model):
        assert model.geometry == "hyperboloid"
        assert model.dim == 513

    def test_encode_images(self, model):
        img = Image.new("RGB", (256, 256), color=(50, 100, 150))
        embeddings = model.encode_images([img])
        assert embeddings.shape == (1, 513)
        assert embeddings.dtype == np.float32

    def test_encode_batch(self, model):
        """Test encoding multiple images (one at a time due to ONNX batch=1 limitation)."""
        images = [Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50)) for i in range(3)]
        # Current ONNX models only support batch_size=1, so encode one at a time
        embeddings = [model.encode_images([img]) for img in images]
        embeddings = np.vstack(embeddings)
        assert embeddings.shape == (3, 513)
