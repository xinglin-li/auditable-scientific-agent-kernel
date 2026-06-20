# src/agent_runtime/retrieval/bm25.py
import math
from typing import Dict, List, Set

from agent_runtime.retrieval.models import Chunk, RetrievalHit
from agent_runtime.retrieval.tokenizer import SimpleTokenizer


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: Dict[str, Chunk] = {}

        self.inverted_index: Dict[str, Set[str]] = {}
        self.chunk_tf: Dict[str, Dict[str, int]] = {}
        self.chunk_lengths: Dict[str, int] = {}

        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}

    def fit(self, chunks: List[Chunk]):
        self.chunks = {c.chunk_id: c for c in chunks}
        total_tokens_all_docs = 0

        for chunk in chunks:
            tokens = SimpleTokenizer.tokenize(chunk.text)
            self.chunk_lengths[chunk.chunk_id] = len(tokens)
            total_tokens_all_docs += len(tokens)

            self.chunk_tf[chunk.chunk_id] = {}
            for token in tokens:
                self.chunk_tf[chunk.chunk_id][token] = self.chunk_tf[chunk.chunk_id].get(token, 0) + 1

                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(chunk.chunk_id)

        num_docs = len(chunks)
        self.avgdl = total_tokens_all_docs / num_docs if num_docs > 0 else 0.0

        for token, containing_chunks in self.inverted_index.items():
            n_q = len(containing_chunks)
            self.idf[token] = math.log((num_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        query_tokens = SimpleTokenizer.tokenize(query)
        scores: Dict[str, float] = {}

        if not query_tokens or not self.chunks:
            return []

        for token in query_tokens:
            if token not in self.idf:
                continue

            target_chunk_ids = self.inverted_index.get(token, set())
            for chunk_id in target_chunk_ids:
                tf = self.chunk_tf[chunk_id].get(token, 0)
                doc_len = self.chunk_lengths[chunk_id]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))

                scores[chunk_id] = scores.get(chunk_id, 0.0) + self.idf[token] * (numerator / denominator)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        hits = []
        for rank, (chunk_id, score) in enumerate(sorted_scores, start=1):
            chunk = self.chunks[chunk_id]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=round(score, 4),
                    rank=rank,
                )
            )
        return hits
