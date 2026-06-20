# src/agent_runtime/memory/models.py
from datetime import UTC, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ── V2 Memory type constants ───────────────────────────────────

MEMORY_TYPE_WORKING = "working"
MEMORY_TYPE_EPISODIC = "episodic"
MEMORY_TYPE_SEMANTIC = "semantic"
MEMORY_TYPE_LONG_TERM = "long_term"

ALL_MEMORY_TYPES = {MEMORY_TYPE_WORKING, MEMORY_TYPE_EPISODIC, MEMORY_TYPE_SEMANTIC, MEMORY_TYPE_LONG_TERM}


class MemoryRecord(BaseModel):
    """A single memory record in long-term store.

    Now supports V2 memory_type classification:
      - working: current run state / context
      - episodic: past run trace, failure, decision, task outcome
      - semantic: curated knowledge, RAG documents, skill metadata
      - long_term: cross-session facts, preferences, rules
    """
    memory_id: str
    namespace: str
    key: str
    memory_type: str = MEMORY_TYPE_LONG_TERM
    content: str
    importance: float = 1.0
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: Dict[str, str] = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    """A structured snapshot of a past agent run for episodic memory.

    Stored alongside MemoryRecord in the same SQLite store
    with memory_type="episodic" and episode_id in metadata.
    """
    episode_id: str
    run_id: str
    task_summary: str
    outcome: str  # completed, failed, cancelled, max_steps_exceeded
    key_decisions: List[str] = Field(default_factory=list)
    error_messages: List[str] = Field(default_factory=list)
    step_count: int = 0
    recorded_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_memory_record(self, namespace: str) -> MemoryRecord:
        """Convert episode to a storable MemoryRecord."""
        return MemoryRecord(
            memory_id=self.episode_id,
            namespace=namespace,
            key=f"episode_{self.run_id}",
            memory_type=MEMORY_TYPE_EPISODIC,
            content=self.task_summary,
            importance=max(0.1, 1.0 - len(self.error_messages) * 0.2),
            metadata={
                "episode_id": self.episode_id,
                "run_id": self.run_id,
                "outcome": self.outcome,
                "step_count": str(self.step_count),
                "decisions": "|".join(self.key_decisions),
                "errors": "|".join(self.error_messages),
            },
        )

    @classmethod
    def from_memory_record(cls, record: MemoryRecord) -> "EpisodeRecord":
        meta = record.metadata
        return cls(
            episode_id=meta.get("episode_id", record.memory_id),
            run_id=meta.get("run_id", ""),
            task_summary=record.content,
            outcome=meta.get("outcome", "unknown"),
            key_decisions=meta.get("decisions", "").split("|") if meta.get("decisions") else [],
            error_messages=meta.get("errors", "").split("|") if meta.get("errors") else [],
            step_count=int(meta.get("step_count", 0)),
            recorded_at=record.updated_at,
        )
