"""Hyper3-CLIP ships a text tower; these tests keep it wired up.

The published checkpoint stores the text encoder under the full Hugging Face
``CLIPTextModel`` path while the model binds its backbone one level deeper. A
regression in that rename loads a randomly initialised text encoder and still
returns embeddings, so text search degrades silently rather than failing. These
tests pin both the rename and the capability declaration that HyperView reads.
"""

from __future__ import annotations

import hyper_models
from hyper_models.torch_models import _align_text_tower_keys


class TestTextTowerKeyAlignment:
    def test_checkpoint_text_prefix_is_rewritten_to_the_model_prefix(self) -> None:
        aligned = _align_text_tower_keys(
            {"text_encoder.backbone.text_model.embeddings.token_embedding.weight": 1}
        )

        assert aligned == {"text_encoder.backbone.embeddings.token_embedding.weight": 1}

    def test_non_text_keys_are_untouched(self) -> None:
        state = {
            "vision_encoder.backbone.encoder.layers.0.mlp.fc1.weight": 1,
            "image_proj.weight": 2,
            "text_proj.weight": 3,
            "textual_alpha": 4,
        }

        assert _align_text_tower_keys(state) == state

    def test_rename_only_strips_the_leading_occurrence(self) -> None:
        # A layer literally named "text_model" deeper in the tree must survive.
        aligned = _align_text_tower_keys(
            {"text_encoder.backbone.text_model.encoder.text_model.weight": 1}
        )

        assert aligned == {"text_encoder.backbone.encoder.text_model.weight": 1}


class TestTextCapabilityDeclaration:
    def test_hyper3_clip_declares_image_and_text(self) -> None:
        info = hyper_models.get_model_info("hyper3-clip-v1")

        assert info.modalities == ("image", "text")

    def test_image_only_entries_keep_the_image_default(self) -> None:
        info = hyper_models.get_model_info("hycoclip-vit-s")

        assert info.modalities == ("image",)
