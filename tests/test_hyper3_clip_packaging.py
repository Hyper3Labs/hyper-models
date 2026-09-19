"""Exercise the JSON-only Hub package and its image/text input contracts."""

import json

import numpy as np
import pytest
from PIL import Image

import hyper_models
from hyper_models.preprocessing import ImageConfig, preprocess_images


def test_hyper3_downloads_runtime_assets(monkeypatch, tmp_path):
    from hyper_models import loader

    seen = {}

    def download(repo, **kwargs):
        seen.update(repo=repo, **kwargs)
        return str(tmp_path)

    monkeypatch.setattr(loader, "snapshot_download", download)
    model = hyper_models.load("hyper3-clip-v1")
    assert model.dim == 513
    assert seen["repo"] == "hyper3labs/hyper3-clip-v1"
    assert {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"} <= set(
        seen["allow_patterns"]
    )
    assert "config.yaml" not in seen["allow_patterns"]


def test_hyper3_preserves_edges_of_non_square_images():
    # The red edge disappears under the default center crop.
    pixels = np.zeros((8, 24, 3), dtype=np.uint8)
    pixels[:, :5, 0] = 255
    image = Image.fromarray(pixels)
    config = hyper_models.get_model_info("hyper3-clip-v1").image_config
    actual = preprocess_images([image], config)[0]
    resized = np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    expected = (
        resized.transpose(2, 0, 1) - np.array(config.mean, dtype=np.float32)[:, None, None]
    ) / np.array(config.std, dtype=np.float32)[:, None, None]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    cropped = preprocess_images([image], ImageConfig(size=8))
    assert not cropped.any()


def test_json_only_loader_uses_bundled_text_config_and_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("timm")
    tokenizers = pytest.importorskip("tokenizers")
    from safetensors.torch import save_file

    import hyper3_clip.models.hyper3_clip as implementation
    from hyper_models.torch_models import Hyper3ClipTorchModel

    class TinyVision(torch.nn.Module):
        output_dim = 3

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, values):
            return values.mean(dim=(-1, -2))

    monkeypatch.setattr(implementation, "VisionEncoder", TinyVision)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<|startoftext|>": 0, "<|endoftext|>": 1, "sofa": 2},
            unk_token="<|endoftext|>",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=8,
    )
    tokenizer.save_pretrained(tmp_path)
    text_config = transformers.CLIPTextConfig(
        vocab_size=3,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        bos_token_id=0,
        eos_token_id=1,
    ).to_dict()
    config = dict(
        model_type="hyper3_clip",
        vision_backbone="test-vision",
        text_model_name="unused/clip",
        embed_dim=5,
        curv_init=1.0,
        learn_curv=True,
        text_config=text_config,
        image_size=32,
        max_text_length=8,
        curvature_min=0.1,
        curvature_max=10.0,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    original = implementation.Hyper3CLIP(
        **{
            key: config[key]
            for key in (
                "vision_backbone",
                "text_model_name",
                "embed_dim",
                "curv_init",
                "learn_curv",
                "text_config",
            )
        },
        entail_weight=0,
        inter_aperture_scale=0,
        intra_aperture_scale=0,
        vision_pretrained=False,
        text_pretrained=False,
        tokenizer_name_or_path=str(tmp_path),
    ).eval()
    state = {
        (
            key.replace("text_encoder.backbone.", "text_encoder.backbone.text_model.", 1)
            if key.startswith("text_encoder.backbone.")
            and not key.startswith("text_encoder.backbone.text_model.")
            else key
        ): value
        for key, value in original.state_dict().items()
    }
    save_file(state, tmp_path / "model.safetensors")
    model = Hyper3ClipTorchModel(
        tmp_path / "model.safetensors",
        geometry="hyperboloid",
        dim=6,
        device="cpu",
        image_config=hyper_models.get_model_info("hyper3-clip-v1").image_config,
    )
    texts = ["sofa " * 20]
    actual = model.encode_texts(texts)
    tokens = original.tokenizer(
        texts, padding=True, truncation=True, max_length=8, return_tensors="pt"
    )
    with torch.inference_mode():
        expected = original.encode_text(tokens["input_ids"], tokens["attention_mask"]).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (1, 6)
    assert model._image_config.size == 32
    assert model.encode_images([Image.new("RGB", (64, 32), color="red")]).shape == (1, 6)
    assert not (tmp_path / "config.yaml").exists()

    # A stale YAML left in a previously downloaded directory cannot override JSON.
    (tmp_path / "config.yaml").write_text("invalid legacy configuration")
    reloaded = Hyper3ClipTorchModel(
        tmp_path / "model.safetensors", geometry="hyperboloid", dim=6, device="cpu"
    )
    np.testing.assert_allclose(reloaded.encode_texts(texts), expected, atol=1e-6)

    # Legacy local YAML artifacts still load through the original model section.
    import yaml

    (tmp_path / "config.json").unlink()
    legacy_model_config = {
        key: value
        for key, value in config.items()
        if key
        in (
            "vision_backbone",
            "text_model_name",
            "embed_dim",
            "curv_init",
            "learn_curv",
            "text_config",
        )
    }
    legacy_model_config.update(
        entail_weight=0,
        inter_aperture_scale=0,
        intra_aperture_scale=0,
        tokenizer_name_or_path=str(tmp_path),
    )
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": legacy_model_config,
                "data": {"image_size": 32, "max_text_length": 8},
            }
        )
    )
    legacy = Hyper3ClipTorchModel(
        tmp_path / "model.safetensors", geometry="hyperboloid", dim=6, device="cpu"
    )
    np.testing.assert_allclose(legacy.encode_texts(texts), expected, atol=1e-6)
