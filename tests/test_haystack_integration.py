"""Haystack wiring and Lorentz retrieval tests without downloading model weights."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("haystack")

from haystack import Document, Pipeline, component  # noqa: E402
from haystack.components.converters import OutputAdapter  # noqa: E402
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever  # noqa: E402
from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: E402
from haystack.utils import ComponentDevice, Secret  # noqa: E402
from huggingface_hub.errors import GatedRepoError  # noqa: E402

from hyper_models.integrations.haystack import (  # noqa: E402
    Hyper3DocumentImageEmbedder,
    Hyper3TextEmbedder,
)
from hyper_models.integrations.haystack.backend import (  # noqa: E402
    DEFAULT_REVISION,
    Hyper3EmbeddingBackendFactory,
)


def _lorentz_point(value: float) -> np.ndarray:
    point = np.zeros(513, dtype=np.float32)
    point[0] = np.sqrt(1 + value * value)
    point[1] = value
    return point


class FakeModel:
    geometry = "hyperboloid"
    dim = 513

    def __init__(self) -> None:
        self.warm_up_calls = 0
        self.image_batch_sizes: list[int] = []

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([_lorentz_point(0.5 if "red" in text else -0.5) for text in texts])

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_batch_sizes.append(len(images))
        return np.stack(
            [_lorentz_point(0.5 if image.getpixel((0, 0))[0] > 100 else -0.5) for image in images]
        )


@pytest.fixture(autouse=True)
def clear_backend_cache():
    Hyper3EmbeddingBackendFactory._instances.clear()
    yield
    Hyper3EmbeddingBackendFactory._instances.clear()


@component
class TextSource:
    @component.output_types(text=str)
    def run(self, text: str) -> dict[str, str]:
        return {"text": text}


@component
class DocumentSource:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {"documents": documents}


def test_components_share_sdk_loader_and_serialize(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    model = FakeModel()
    with patch("hyper_models.load", return_value=model) as load:
        image = Hyper3DocumentImageEmbedder(device=ComponentDevice.from_str("cpu"), batch_size=2)
        text = Hyper3TextEmbedder(device=ComponentDevice.from_str("cpu"))
        assert image._backend is text._backend
        restored_image = Hyper3DocumentImageEmbedder.from_dict(image.to_dict())
        restored_text = Hyper3TextEmbedder.from_dict(text.to_dict())
        assert restored_image.batch_size == 2
        assert restored_text.revision == DEFAULT_REVISION
        assert restored_text.token.resolve_value() == "environment-token"
        image.warm_up()
        text.warm_up()
        assert model.warm_up_calls == 1
        load.assert_called_once_with(
            "hyper3-clip-v1",
            revision=DEFAULT_REVISION,
            token="environment-token",
            device="cpu",
            local_files_only=False,
        )


def test_cached_hub_auth_uses_loader_default(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("hyper_models.load", return_value=FakeModel()) as load:
        Hyper3TextEmbedder(token=None, local_files_only=True).warm_up()
    assert load.call_args.kwargs["token"] is None
    assert load.call_args.kwargs["local_files_only"] is True


def test_gated_model_access_has_actionable_error():
    response = Mock(headers={}, request=Mock())
    with patch(
        "hyper_models.load", side_effect=GatedRepoError("Access restricted", response=response)
    ):
        with pytest.raises(RuntimeError, match="Request access.*hf auth login"):
            Hyper3TextEmbedder().warm_up()


def test_revision_and_token_isolation():
    first = Hyper3TextEmbedder(revision="first", token=Secret.from_token("secret-a"))
    second = Hyper3TextEmbedder(revision="second", token=Secret.from_token("secret-a"))
    third = Hyper3TextEmbedder(revision="first", token=Secret.from_token("secret-b"))
    assert first._backend is not second._backend
    assert first._backend is not third._backend
    assert "secret-a" not in repr(Hyper3EmbeddingBackendFactory._instances)
    with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
        first.to_dict()


def test_legacy_model_id_is_an_alias_for_catalog_entry():
    legacy = Hyper3TextEmbedder(model="hyper3labs/hyper3-clip-v1")
    current = Hyper3TextEmbedder()
    assert legacy._backend is current._backend


def test_rejects_unsupported_geometry_or_modality():
    with pytest.raises(ValueError, match="only 'hyper3-clip-v1'"):
        Hyper3TextEmbedder(model="hycoclip-vit-s")
    wrong_model = FakeModel()
    wrong_model.dim = 512
    with patch("hyper_models.load", return_value=wrong_model):
        with pytest.raises(ValueError, match="513-coordinate"):
            Hyper3TextEmbedder().warm_up()


@pytest.mark.parametrize("value,error", [(123, TypeError), ("  ", ValueError)])
def test_text_requires_nonempty_string(value, error):
    with pytest.raises(error):
        Hyper3TextEmbedder().run(value)


def test_image_document_requires_existing_path(tmp_path):
    embedder = Hyper3DocumentImageEmbedder(root_path=str(tmp_path))
    with pytest.raises(ValueError, match="file_path"):
        embedder.run([Document(id="missing-meta")])
    with pytest.raises(ValueError, match="Could not load image"):
        embedder.run([Document(id="missing-file", meta={"file_path": "absent.png"})])


def test_image_files_are_batched_and_documents_are_copied(tmp_path):
    for name, color in (("red", "red"), ("blue", "blue"), ("red2", "red")):
        Image.new("RGB", (8, 8), color).save(tmp_path / f"{name}.png")
    documents = [
        Document(id=name, meta={"image": f"{name}.png"}) for name in ("red", "blue", "red2")
    ]
    model = FakeModel()
    with patch("hyper_models.load", return_value=model):
        embedder = Hyper3DocumentImageEmbedder(
            file_path_meta_field="image", root_path=str(tmp_path), batch_size=2
        )
        output = embedder.run(documents)["documents"]
    assert model.image_batch_sizes == [2, 1]
    assert all(original.embedding is None for original in documents)
    assert all(len(document.embedding) == 513 for document in output)
    assert output[0].meta["embedding_source"]["geometry"] == "lorentz"
    assert output[0].meta["embedding_source"]["model"] == "hyper3labs/hyper3-clip-v1"
    assert output[0].meta["image"] == "red.png"


def test_pipeline_sockets_and_lorentz_query_transform(tmp_path):
    Image.new("RGB", (8, 8), "red").save(tmp_path / "red.png")
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "blue.png")
    documents = [
        Document(id="red", meta={"file_path": "red.png"}),
        Document(id="blue", meta={"file_path": "blue.png"}),
    ]
    model = FakeModel()
    with patch("hyper_models.load", return_value=model):
        indexer = Pipeline()
        indexer.add_component("source", DocumentSource())
        indexer.add_component("images", Hyper3DocumentImageEmbedder(root_path=str(tmp_path)))
        indexer.connect("source.documents", "images.documents")
        indexed = indexer.run({"source": {"documents": documents}})["images"]["documents"]

        store = InMemoryDocumentStore(embedding_similarity_function="dot_product", shared=False)
        store.write_documents(indexed)
        query = Pipeline()
        query.add_component("source", TextSource())
        query.add_component("text", Hyper3TextEmbedder())
        query.add_component(
            "lorentz_query",
            OutputAdapter(
                template="{{ [-embedding[0]] + embedding[1:] }}", output_type=list[float]
            ),
        )
        query.add_component(
            "retriever",
            InMemoryEmbeddingRetriever(document_store=store, top_k=2, scale_score=False),
        )
        query.connect("source.text", "text.text")
        query.connect("text.embedding", "lorentz_query.embedding")
        query.connect("lorentz_query.output", "retriever.query_embedding")
        restored_query = Pipeline.loads(
            query.dumps(),
            allowed_modules=["test_haystack_integration", "hyper_models.integrations.haystack"],
        )
        # InMemoryDocumentStore data is deliberately not persisted by Haystack's pipeline serializer.
        restored_query.get_component("retriever").document_store = store
        hits = restored_query.run({"source": {"text": "red square"}})["retriever"]["documents"]

    assert [hit.id for hit in hits] == ["red", "blue"]
    query_point = _lorentz_point(0.5)
    direct = [
        -query_point[0] * document.embedding[0] + np.dot(query_point[1:], document.embedding[1:])
        for document in indexed
    ]
    np.testing.assert_allclose([hit.score for hit in hits], direct, rtol=1e-6, atol=1e-6)
