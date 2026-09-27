"""Public loading controls retain the existing catalog and lazy model contracts."""

from pathlib import Path

import hyper_models
from hyper_models import loader
from hyper_models.torch_models import Hyper3ClipTorchModel, UNCHATorchModel


def test_hub_options_are_forwarded_and_torch_device_is_lazy(monkeypatch, tmp_path):
    seen = {}

    def download(repo, **kwargs):
        seen.update(repo=repo, **kwargs)
        return str(tmp_path)

    monkeypatch.setattr(loader, "snapshot_download", download)
    model = hyper_models.load(
        "hyper3-clip-v1",
        revision="abc123",
        token="private-token",
        local_files_only=True,
        device="cpu",
    )

    assert isinstance(model, Hyper3ClipTorchModel)
    assert model._device_override == "cpu"
    assert model._model is None
    assert seen["repo"] == "hyper3labs/hyper3-clip-v1"
    assert seen["revision"] == "abc123"
    assert seen["token"] == "private-token"
    assert seen["local_files_only"] is True


def test_existing_call_uses_hf_token_and_default_device(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setenv("HF_TOKEN", "existing-token")
    monkeypatch.setenv("HF_API_TOKEN", "fallback-token")

    def download(repo, **kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(loader, "snapshot_download", download)
    model = hyper_models.load("hyper3-clip-v1")
    assert isinstance(model, Hyper3ClipTorchModel)
    assert model._device_override is None
    assert seen["token"] == "existing-token"
    assert seen["revision"] is None
    assert seen["local_files_only"] is False


def test_token_fallback_and_explicit_auth_disable(monkeypatch, tmp_path):
    seen = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_API_TOKEN", "fallback-token")

    def download(repo, **kwargs):
        seen.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(loader, "snapshot_download", download)
    hyper_models.load("hycoclip-vit-s")
    hyper_models.load("hycoclip-vit-s", token=False)
    assert seen[0]["token"] == "fallback-token"
    assert seen[1]["token"] is False


def test_local_artifacts_skip_hub_and_route_device_only_to_torch(monkeypatch, tmp_path):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("local_path must skip Hub download")

    monkeypatch.setattr(loader, "snapshot_download", unexpected_download)
    torch_model = hyper_models.load(
        "uncha-vit-s", local_path=tmp_path / "checkpoint.pth", device="cpu"
    )
    assert isinstance(torch_model, UNCHATorchModel)
    assert torch_model._device_override == "cpu"

    onnx_model = hyper_models.load(
        "hycoclip-vit-s", local_path=tmp_path / "model.onnx", device="cpu"
    )
    assert onnx_model._path == Path(tmp_path / "model.onnx")


def test_public_warm_up_delegates_to_lazy_loader(monkeypatch, tmp_path):
    seen = []

    def ensure(self):
        seen.append(self)

    monkeypatch.setattr(Hyper3ClipTorchModel, "_ensure_model", ensure)
    model = Hyper3ClipTorchModel(tmp_path / "model.safetensors", geometry="hyperboloid", dim=513)
    model.warm_up()
    assert seen == [model]
