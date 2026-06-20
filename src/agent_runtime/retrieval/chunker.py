# src/agent_runtime/retrieval/chunker.py
from typing import List

from agent_runtime.retrieval.models import Chunk, Document


class SimpleChunker:
    def __init__(self, chunk_size: int = 150, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: Document) -> List[Chunk]:
        chunks = []
        content = doc.content
        length = len(content)
        start = 0
        chunk_idx = 0

        if length <= self.chunk_size:
            return [Chunk(chunk_id=f"{doc.document_id}_c0", document_id=doc.document_id, text=content, trust_level=doc.trust_level)]

        while start < length:
            end = min(start + self.chunk_size, length)
            text_slice = content[start:end]

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}_c{chunk_idx}",
                    document_id=doc.document_id,
                    text=text_slice,
                    metadata={"source_path": doc.source_path, "title": doc.title},
                    trust_level=doc.trust_level,
                )
            )

            chunk_idx += 1
            start += self.chunk_size - self.overlap
            if end == length:
                break

        return chunks
