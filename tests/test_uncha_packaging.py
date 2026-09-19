"""UNCHA checkpoints include pickled OmegaConf metadata alongside weights."""

import pytest


def test_uncha_loads_checkpoint_with_omegaconf_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf

    from hyper_models.torch_models import UNCHATorchModel

    weights = {"visual_proj.weight": torch.ones(2, 3)}
    path = tmp_path / "uncha.pth"
    torch.save({"model": weights, "config": OmegaConf.create({"model": "vit_s"})}, path)
    model = UNCHATorchModel(path, geometry="hyperboloid", dim=513, variant="vit_s")
    model._torch = torch
    loaded = model._load_state_dict()
    assert set(loaded) == set(weights)
    torch.testing.assert_close(loaded["visual_proj.weight"], weights["visual_proj.weight"])
