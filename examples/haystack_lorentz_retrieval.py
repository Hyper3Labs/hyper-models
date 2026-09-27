"""Index local images and query their native Lorentz embeddings with Haystack.

Install hyper-models[ml,haystack] and authenticate for Hyper3-CLIP v1 first.
Replace the two image paths below with your own files.
"""

from haystack import Document, Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from hyper_models.integrations.haystack import Hyper3DocumentImageEmbedder, Hyper3TextEmbedder

documents = [
    Document(id="sofa", meta={"file_path": "sofa.jpg"}),
    Document(id="chair", meta={"file_path": "chair.jpg"}),
]
store = InMemoryDocumentStore(embedding_similarity_function="dot_product")
indexing = Pipeline()
indexing.add_component("embedder", Hyper3DocumentImageEmbedder())
indexing.add_component("writer", DocumentWriter(document_store=store))
indexing.connect("embedder.documents", "writer.documents")
indexing.run({"embedder": {"documents": documents}})

query = Pipeline()
query.add_component("embedder", Hyper3TextEmbedder())
query.add_component(
    "lorentz_query",
    OutputAdapter(template="{{ [-embedding[0]] + embedding[1:] }}", output_type=list[float]),
)
query.add_component(
    "retriever",
    InMemoryEmbeddingRetriever(document_store=store, top_k=2, scale_score=False),
)
query.connect("embedder.embedding", "lorentz_query.embedding")
query.connect("lorentz_query.output", "retriever.query_embedding")

result = query.run({"embedder": {"text": "a grey sofa"}})
for document in result["retriever"]["documents"]:
    print(document.id, document.score)
