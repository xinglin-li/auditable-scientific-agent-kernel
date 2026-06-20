# src/agent_runtime/retrieval/citations.py
from typing import Any, Dict, List, Tuple


class CitationBuilder:
    @staticmethod
    def build_citations(top_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Format retrieved chunks for model context and return traceable citation metadata."""
        formatted_evidence_blocks = []
        citation_map = {}

        for idx, chunk in enumerate(top_chunks, start=1):
            citation_id = f"Doc-{idx}"
            formatted_evidence_blocks.append(
                f"[{citation_id}] (Source Document ID: {chunk['document_id']})\nContent: {chunk['text']}"
            )
            citation_map[citation_id] = chunk

        evidence_text = "\n\n".join(formatted_evidence_blocks)
        return evidence_text, citation_map
