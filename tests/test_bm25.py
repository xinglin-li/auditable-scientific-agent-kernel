# tests/test_bm25.py
import pytest

from agent_runtime.retrieval.bm25 import BM25Retriever
from agent_runtime.retrieval.models import Chunk


@pytest.fixture
def populated_retriever():
    chunks = [
        Chunk(chunk_id="c1", document_id="doc_arima", text="Throw a ValueError in ARIMA(1,1,1) integration model execution."),
        Chunk(chunk_id="c2", document_id="doc_backtest", text="Systematic backtesting workflow within XL-Systematic platform data integrity."),
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)
    return retriever


def test_precise_keyword_match(populated_retriever):
    """Exact variable and proper-noun matches should rank first."""
    hits = populated_retriever.search("ARIMA(1,1,1)")
    assert len(hits) > 0
    assert hits[0].document_id == "doc_arima"


def test_rare_vs_common_word_weight(populated_retriever):
    """Rare terms should outweigh common terms."""
    query = "Where is the model of XL-Systematic?"
    hits = populated_retriever.search(query)

    assert hits[0].chunk_id == "c2"


def test_empty_and_no_match_query(populated_retriever):
    """Empty and unrelated queries should return no hits."""
    assert populated_retriever.search("") == []
    assert populated_retriever.search("completely unrelated garbage text") == []
