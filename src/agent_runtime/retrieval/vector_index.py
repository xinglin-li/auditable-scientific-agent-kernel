# src/agent_runtime/retrieval/vector_index.py
import math
from typing import Any, Dict, List, Tuple

from agent_runtime.retrieval.embeddings import BaseEmbeddingProvider
from agent_runtime.retrieval.models import Chunk


class VectorIndex:
    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.provider = embedding_provider
        self.storage: Dict[str, List[float]] = {}
        self.chunks: Dict[str, Chunk] = {}

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        sq_sum = sum(x**2 for x in vec)
        norm = math.sqrt(sq_sum)
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    @staticmethod
    def _dot_product(vec_a: List[float], vec_b: List[float]) -> float:
        """Return the dot product of two same-length vectors."""
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def add_chunks(self, chunks: List[Chunk]):
        for chunk in chunks:
            raw_vec = self.provider.get_embedding(chunk.text)
            normalized_vec = self._l2_normalize(raw_vec)

            self.storage[chunk.chunk_id] = normalized_vec
            self.chunks[chunk.chunk_id] = chunk

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Return the top-k chunks by cosine similarity."""
        if not self.storage or not query:
            return []

        query_raw = self.provider.get_embedding(query)
        query_vec = self._l2_normalize(query_raw)

        scored_hits: List[Tuple[str, float]] = []

        for chunk_id, chunk_vec in self.storage.items():
            similarity = self._dot_product(query_vec, chunk_vec)
            scored_hits.append((chunk_id, similarity))

        sorted_hits = sorted(scored_hits, key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_hits, start=1):
            chunk = self.chunks[chunk_id]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "score": round(score, 4),
                    "rank": rank,
                }
            )
        return results
