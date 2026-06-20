# tests/test_memory_store.py
from agent_runtime.memory.models import MemoryRecord
from agent_runtime.memory.sqlite_store import SQLiteMemoryStore


def test_namespace_strict_isolation():
    """Memories from one namespace must not leak into another namespace."""
    store = SQLiteMemoryStore(":memory:")

    rec_xinglin = MemoryRecord(
        memory_id="m1",
        namespace="users/xinglin/pref",
        key="baseline",
        memory_type="semantic",
        content="Use seasonal-naive",
    )
    rec_hacker = MemoryRecord(
        memory_id="m2",
        namespace="users/hacker/pref",
        key="baseline",
        memory_type="semantic",
        content="Use linear-trend",
    )

    store.put_memory(rec_xinglin)
    store.put_memory(rec_hacker)

    xinglin_list = store.list_namespace_memories("users/xinglin/pref")
    assert len(xinglin_list) == 1
    assert xinglin_list[0].content == "Use seasonal-naive"

    assert "Use linear-trend" not in [m.content for m in xinglin_list]


def test_upsert_on_conflict():
    """A new memory with the same namespace/key should replace the old value."""
    store = SQLiteMemoryStore(":memory:")
    r1 = MemoryRecord(memory_id="id1", namespace="proj_1", key="risk_limit", memory_type="semantic", content="Max limit 5%")
    r2 = MemoryRecord(memory_id="id2", namespace="proj_1", key="risk_limit", memory_type="semantic", content="Max limit 2%")

    store.put_memory(r1)
    store.put_memory(r2)

    res = store.get_memory_by_key("proj_1", "risk_limit")
    assert res.content == "Max limit 2%"
