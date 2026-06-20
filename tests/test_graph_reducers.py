# tests/test_graph_reducers.py
import pytest
from agent_runtime.graph.reducers import (
    reduce_append,
    reduce_unique_str_list,
    reduce_artifact_refs
)

def test_reduce_append_chaining():
    """Verify that trace_events are append-only and preserve order."""
    left = [{"event": "start", "step": 0}]
    right = [{"event": "node_processed", "step": 1}]
    res = reduce_append(left, right)
    assert len(res) == 2
    assert res[0]["event"] == "start"
    assert res[1]["event"] == "node_processed"

def test_reduce_unique_str_list_dedup():
    """Verify that concurrent job submissions do not duplicate asynchronous job IDs."""
    left = ["job_001", "job_002"]
    right = ["job_002", "job_003"]
    res = reduce_unique_str_list(left, right)
    assert res == ["job_001", "job_002", "job_003"]

def test_reduce_artifact_refs_conflict_resolution():
    """Verify that replayed nodes upsert report and file references by ID."""
    left = [{"artifact_id": "cpi_report", "path": "old/path.csv"}]
    right = [
        {"artifact_id": "cpi_report", "path": "new/actual_path.csv"},
        {"artifact_id": "unemployment_report", "path": "unemp.csv"}
    ]
    res = reduce_artifact_refs(left, right)
    assert len(res) == 2
    # Verify that new metadata merges on an ID conflict instead of being appended blindly.
    cpi_ref = next(r for r in res if r["artifact_id"] == "cpi_report")
    assert cpi_ref["path"] == "new/actual_path.csv"
