# tests/test_hybrid_retrieval.py
import pytest

from agent_runtime.retrieval.bm25 import BM25Retriever
from agent_runtime.retrieval.embeddings import FakeEmbeddingProvider
from agent_runtime.retrieval.models import Chunk
from agent_runtime.retrieval.pipeline import HybridRetrievalPipeline
from agent_runtime.retrieval.vector_index import VectorIndex


@pytest.fixture
def initialized_pipeline():
    chunks = [
        Chunk(chunk_id="c_arima", document_id="doc_1", text="ValueError occurred during training standard ARIMA forecast series."),
        Chunk(chunk_id="c_backtest", document_id="doc_2", text="Evaluating data leakage boundaries using rigorous quantitative backtesting."),
    ]

    bm25 = BM25Retriever()
    bm25.fit(chunks)

    v_index = VectorIndex(FakeEmbeddingProvider(dimensions=4))
    v_index.add_chunks(chunks)

    return HybridRetrievalPipeline(bm25, v_index)


def test_hybrid_pipeline_dedup_and_rank(initialized_pipeline):
    """RRF should rank a chunk higher when both retrieval routes hit it."""
    query = "ValueError in ARIMA series"
    evidence_text, citation_map = initialized_pipeline.execute_pipeline(query, top_k=1)

    assert evidence_text != "insufficient_evidence"
    assert "Doc-1" in citation_map
    assert citation_map["Doc-1"]["chunk_id"] == "c_arima"


def test_insufficient_evidence_circuit_breaker(initialized_pipeline):
    """Unrelated queries should return insufficient_evidence instead of a weak match."""
    query = "How to make spicy noodle soup in kitchen?"
    evidence_text, citation_map = initialized_pipeline.execute_pipeline(query, top_k=2)

    assert evidence_text == "insufficient_evidence"
    assert citation_map == {}
