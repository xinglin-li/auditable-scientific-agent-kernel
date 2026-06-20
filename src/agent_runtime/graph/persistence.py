# src/agent_runtime/graph/persistence.py
import sqlite3
from typing import ClassVar, Optional

from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointPersistenceManager:
    """Manage the lifecycle of a LangGraph SQLite checkpointer."""

    _shared_memory_connections: ClassVar[dict[str, sqlite3.Connection]] = {}

    def __init__(self, conn_string: str = ":memory:"):
        self._conn_string = conn_string
        self._conn: Optional[sqlite3.Connection] = None
        self.checkpointer: Optional[SqliteSaver] = None

    def __enter__(self) -> SqliteSaver:
        uri = self._conn_string.startswith("file:")
        self._conn = sqlite3.connect(
            self._conn_string,
            check_same_thread=False,
            uri=uri,
        )
        if self._is_shared_memory_uri:
            self._shared_memory_connections.setdefault(self._conn_string, self._conn)
        self.checkpointer = SqliteSaver(self._conn)
        return self.checkpointer

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._conn is not None and self._shared_memory_connections.get(self._conn_string) is not self._conn:
            self._conn.close()
        self._conn = None
        self.checkpointer = None

    @property
    def _is_shared_memory_uri(self) -> bool:
        return self._conn_string.startswith("file:") and "mode=memory" in self._conn_string

    @staticmethod
    def create_sqlite_checkpointer(conn_string: str = ":memory:") -> "CheckpointPersistenceManager":
        """Create a context-managed SQLite checkpointer factory."""
        return CheckpointPersistenceManager(conn_string)
