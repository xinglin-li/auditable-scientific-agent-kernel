# src/agent_runtime/memory/policy.py
import uuid

from agent_runtime.memory.models import MemoryRecord
from agent_runtime.memory.sqlite_store import SQLiteMemoryStore


class MemoryWritePolicy:
    FORBIDDEN_KEYWORDS = {"ignore previous", "delete all", "system instruction", "override runtime"}

    def __init__(self, store: SQLiteMemoryStore):
        self.store = store

    def inspect_and_commit(self, namespace: str, key: str, content: str, memory_type: str = "semantic") -> str:
        """
        Review a candidate memory before persistence.

        Returns "committed", "rejected_sensitive", or "skipped_noise".
        """
        lowered_content = content.lower()

        for kw in self.FORBIDDEN_KEYWORDS:
            if kw in lowered_content:
                return "rejected_sensitive"

        if len(content.strip()) < 5:
            return "skipped_noise"

        record = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            namespace=namespace,
            key=key,
            memory_type=memory_type,
            content=content,
        )
        self.store.put_memory(record)
        return "committed"
