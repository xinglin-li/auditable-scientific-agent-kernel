# src/agent_runtime/retrieval/pipeline.py
from typing import Any, Dict, Tuple

from agent_runtime.retrieval.bm25 import BM25Retriever
from agent_runtime.retrieval.citations import CitationBuilder
from agent_runtime.retrieval.hybrid import ReciprocalRankFusion
from agent_runtime.retrieval.reranker import RuleBasedReranker
from agent_runtime.retrieval.vector_index import VectorIndex


class HybridRetrievalPipeline:
    def __init__(self, bm25_retriever: BM25Retriever, vector_index: VectorIndex, vector_only_min_score: float = 0.75):
        self.bm25 = bm25_retriever
        self.v_index = vector_index
        self.fusion = ReciprocalRankFusion()
        self.vector_only_min_score = vector_only_min_score

    def execute_pipeline(self, query: str, top_k: int = 2) -> Tuple[str, Dict[str, Any]]:
        """Run hybrid retrieval and return formatted evidence plus citation metadata."""
        bm25_hits = self.bm25.search(query, top_k=5)
        vector_hits = self.v_index.search(query, top_k=5)

        if not bm25_hits and not vector_hits:
            return "insufficient_evidence", {}

        if not bm25_hits and vector_hits:
            top_vector_score = vector_hits[0].get("score", 0.0)
            if top_vector_score < self.vector_only_min_score:
                return "insufficient_evidence", {}

        fused = self.fusion.fuse(bm25_hits, vector_hits)
        reranked = RuleBasedReranker.rerank(fused)

        final_candidates = reranked[:top_k]
        evidence_text, citation_map = CitationBuilder.build_citations(final_candidates)

        return evidence_text, citation_map
