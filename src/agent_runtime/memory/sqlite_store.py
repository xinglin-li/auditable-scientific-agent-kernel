# src/agent_runtime/memory/sqlite_store.py
import json
import sqlite3
from typing import List, Optional

from agent_runtime.memory.models import MemoryRecord


class SQLiteMemoryStore:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(namespace, key)
            )
            """
        )
        self.conn.commit()

    def put_memory(self, record: MemoryRecord):
        """Insert or update one memory record using namespace/key as the stable identity."""
        self.conn.execute(
            """
            INSERT INTO memories (memory_id, namespace, key, memory_type, content, importance, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
                content = excluded.content,
                updated_at = excluded.updated_at,
                importance = excluded.importance,
                metadata = excluded.metadata
            """,
            (
                record.memory_id,
                record.namespace,
                record.key,
                record.memory_type,
                record.content,
                record.importance,
                record.updated_at,
                json.dumps(record.metadata),
            ),
        )
        self.conn.commit()

    def get_memory_by_key(self, namespace: str, key: str) -> Optional[MemoryRecord]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT memory_id, namespace, key, memory_type, content, importance, updated_at, metadata "
            "FROM memories WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return MemoryRecord(
            memory_id=row[0],
            namespace=row[1],
            key=row[2],
            memory_type=row[3],
            content=row[4],
            importance=row[5],
            updated_at=row[6],
            metadata=json.loads(row[7]),
        )

    def list_namespace_memories(self, namespace: str) -> List[MemoryRecord]:
        """Return only memories that belong to the requested namespace."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT memory_id, namespace, key, memory_type, content, importance, updated_at, metadata "
            "FROM memories WHERE namespace = ?",
            (namespace,),
        )
        rows = cursor.fetchall()
        return [
            MemoryRecord(
                memory_id=r[0],
                namespace=r[1],
                key=r[2],
                memory_type=r[3],
                content=r[4],
                importance=r[5],
                updated_at=r[6],
                metadata=json.loads(r[7]),
            )
            for r in rows
        ]

    def list_episodes_by_namespace(self, namespace: str, limit: int = 10) -> List[MemoryRecord]:
        """Return recent episodic memories for a namespace, ordered by recency."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT memory_id, namespace, key, memory_type, content, importance, updated_at, metadata
               FROM memories
               WHERE namespace = ? AND memory_type = 'episodic'
               ORDER BY updated_at DESC
               LIMIT ?""",
            (namespace, limit),
        )
        rows = cursor.fetchall()
        return [
            MemoryRecord(
                memory_id=r[0],
                namespace=r[1],
                key=r[2],
                memory_type=r[3],
                content=r[4],
                importance=r[5],
                updated_at=r[6],
                metadata=json.loads(r[7]),
            )
            for r in rows
        ]

    def put_episode(self, record: MemoryRecord) -> None:
        """Persist an episodic memory record (convenience wrapper)."""
        if record.memory_type != "episodic":
            raise ValueError(
                f"put_episode requires memory_type='episodic', got '{record.memory_type}'"
            )
        self.put_memory(record)

    def close(self):
        self.conn.close()
