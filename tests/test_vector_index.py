# tests/test_vector_index.py
import pytest

from agent_runtime.retrieval.embeddings import FakeEmbeddingProvider
from agent_runtime.retrieval.models import Chunk
from agent_runtime.retrieval.vector_index import VectorIndex


def test_semantic_similarity_alignment():
    """Semantically aligned text should be retrieved even without exact term overlap."""
    provider = FakeEmbeddingProvider(dimensions=4)
    index = VectorIndex(provider)

    chunks = [
        Chunk(chunk_id="c_time", document_id="d1", text="Mathematical modeling for historical time series data."),
        Chunk(chunk_id="c_trade", document_id="d2", text="Executing automated backtesting routines for systematic trading systems."),
    ]
    index.add_chunks(chunks)

    query = "forecast with stochastic sequential datasets"
    hits = index.search(query, top_k=1)

    assert len(hits) > 0
    assert hits[0]["chunk_id"] == "c_time"


def test_perfect_and_orthogonal_vectors():
    """Identical text should produce a similarity score close to 1.0."""
    provider = FakeEmbeddingProvider(dimensions=4)
    index = VectorIndex(provider)

    chunk = Chunk(chunk_id="c_exact", document_id="d_ex", text="Arbitrary quantitative formula placeholder.")
    index.add_chunks([chunk])

    hits = index.search("Arbitrary quantitative formula placeholder.", top_k=1)
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)
