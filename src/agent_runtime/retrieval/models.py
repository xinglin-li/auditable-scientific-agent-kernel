# src/agent_runtime/retrieval/models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Document(BaseModel):
    document_id: str
    title: str
    source_path: str
    content: str
    trust_level: str = "retrieved_untrusted"
    
class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trust_level: str = "retrieved_untrusted"

class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int