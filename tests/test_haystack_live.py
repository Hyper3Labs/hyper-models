"""Opt-in check against the real gated Hyper3-CLIP model and Haystack retriever.

Run after caching the pinned model: pytest -m integration tests/test_haystack_live.py
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("haystack")

from haystack import Document, Pipeline  # noqa: E402
from haystack.components.converters import OutputAdapter  # noqa: E402
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever  # noqa: E402
from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: E402
from haystack.utils import ComponentDevice  # noqa: E402

from hyper_models.integrations.haystack import (  # noqa: E402
    Hyper3DocumentImageEmbedder,
    Hyper3TextEmbedder,
)


@pytest.mark.integration
def test_sdk_embeddings_and_haystack_scores_match_lorentz_inner_product(tmp_path):
    Image.new("RGB", (224, 224), "red").save(tmp_path / "red.png")
    Image.new("RGB", (224, 224), "blue").save(tmp_path / "blue.png")
    documents = [
        Document(id="red", meta={"file_path": str(tmp_path / "red.png")}),
        Document(id="blue", meta={"file_path": str(tmp_path / "blue.png")}),
    ]
    device = ComponentDevice.from_str("cpu")
    image_embedder = Hyper3DocumentImageEmbedder(device=device, local_files_only=True)
    text_embedder = Hyper3TextEmbedder(device=device, local_files_only=True)
    assert image_embedder._backend is text_embedder._backend
    embedded = image_embedder.run(documents)["documents"]
    query_embedding = text_embedder.run("a red square")["embedding"]
    assert len(query_embedding) == 513
    assert all(len(document.embedding) == 513 for document in embedded)

    store = InMemoryDocumentStore(embedding_similarity_function="dot_product", shared=False)
    store.write_documents(embedded)
    pipeline = Pipeline()
    pipeline.add_component(
        "lorentz_query",
        OutputAdapter(template="{{ [-embedding[0]] + embedding[1:] }}", output_type=list[float]),
    )
    pipeline.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=2, scale_score=False)
    )
    pipeline.connect("lorentz_query.output", "retriever.query_embedding")
    hits = pipeline.run({"lorentz_query": {"embedding": query_embedding}})["retriever"]["documents"]

    query = np.asarray(query_embedding, dtype=np.float64)
    candidates = np.asarray([document.embedding for document in embedded], dtype=np.float64)
    direct_scores = -query[0] * candidates[:, 0] + candidates[:, 1:] @ query[1:]
    expected_order = np.argsort(-direct_scores)
    assert [hit.id for hit in hits] == [embedded[index].id for index in expected_order]
    np.testing.assert_allclose(
        [hit.score for hit in hits], direct_scores[expected_order], rtol=1e-5, atol=1e-5
    )
