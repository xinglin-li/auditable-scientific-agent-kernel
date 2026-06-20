# src/agent_runtime/skills/selector.py
from typing import Dict, List, Optional

from agent_runtime.retrieval.bm25 import BM25Retriever
from agent_runtime.retrieval.embeddings import FakeEmbeddingProvider
from agent_runtime.retrieval.models import Chunk
from agent_runtime.retrieval.vector_index import VectorIndex
from agent_runtime.skills.loader import SkillLoader
from agent_runtime.skills.models import ActivatedSkill, SkillMetadata


class SkillSelector:
    def __init__(self, loader: SkillLoader):
        self.loader = loader
        self.catalog: List[SkillMetadata] = self.loader.discover_catalog()
        self.skill_by_name: Dict[str, SkillMetadata] = {meta.name: meta for meta in self.catalog}

        self.bm25 = BM25Retriever()
        self.embedding_provider = FakeEmbeddingProvider(dimensions=4)
        self.vector_index = VectorIndex(self.embedding_provider)

        self._build_skill_indices()

    def _build_skill_indices(self) -> None:
        """Index lightweight skill metadata, never full SKILL.md bodies."""
        chunks = []
        for meta in self.catalog:
            chunk_id = f"skill_chunk_{meta.name}"
            allowed_tools = " ".join(meta.allowed_tools)
            text = f"Skill: {meta.name}. Description: {meta.description}. Keywords: {allowed_tools}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=f"doc_skill_{meta.name}",
                    text=text,
                    metadata={"skill_name": meta.name, "meta_ref": meta},
                    trust_level="application_trusted",
                )
            )

        if not chunks:
            return

        self.bm25.fit(chunks)
        self.vector_index.add_chunks(chunks)

    def select_and_activate(self, query: str) -> Optional[ActivatedSkill]:
        """
        Select a skill from lightweight metadata, then load the full skill only
        after a match is established.
        """
        if not self.catalog:
            return None

        lowered_query = query.lower()

        # Fast path: explicit skill names should always win.
        for meta in self.catalog:
            if meta.name.lower() in lowered_query:
                return self.loader.load_full_skill(meta)

        bm25_hits = self.bm25.search(query, top_k=2)
        if not bm25_hits:
            return None

        vector_hits = self.vector_index.search(query, top_k=2)
        rrf_scores: Dict[str, float] = {}
        k_constant = 60

        for rank, hit in enumerate(bm25_hits, start=1):
            skill_name = hit.chunk_id.replace("skill_chunk_", "", 1)
            rrf_scores[skill_name] = rrf_scores.get(skill_name, 0.0) + 1.0 / (k_constant + rank)

        for rank, hit in enumerate(vector_hits, start=1):
            skill_name = hit["chunk_id"].replace("skill_chunk_", "", 1)
            rrf_scores[skill_name] = rrf_scores.get(skill_name, 0.0) + 1.0 / (k_constant + rank)

        best_skill_name = max(rrf_scores, key=rrf_scores.get)
        meta = self.skill_by_name.get(best_skill_name)
        if meta is None:
            return None

        return self.loader.load_full_skill(meta)
