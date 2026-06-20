# src/agent_runtime/retrieval/reranker.py
from typing import Any, Dict, List


class RuleBasedReranker:
    @staticmethod
    def rerank(fused_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply deterministic domain boosts after rank fusion."""
        for hit in fused_hits:
            if "arima" in hit["chunk_id"] or "backtest" in hit["chunk_id"]:
                hit["rrf_score"] += 0.05

        return sorted(fused_hits, key=lambda x: x["rrf_score"], reverse=True)
