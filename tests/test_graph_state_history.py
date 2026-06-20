# tests/test_graph_state_history.py
import pytest
from agent_runtime.graph.service import MacroAgentGraphService

def test_state_history_and_artifact_ref_boundary():
    """Verify that checkpoints store references to large results rather than payloads."""
    service = MacroAgentGraphService()
    
    res = service.run_workflow(thread_id="thread_travel_999", run_id="run_999", user_query="Heavy analysis")
    
    # 1. Retrieve the state-history chain.
    history = service.get_state_history(thread_id="thread_travel_999")
    
    # Verify that state evolution records several explicit snapshots across the workflow.
    assert len(history) >= 3
    
    # 2. Verify the lean agent architecture's data-layer boundary.
    latest_values = history[0]["values"]
    # State contains only useful references (artifact_id and URI), never a large raw CSV payload.
    assert "artifact_refs" in latest_values
    ref_item = latest_values["artifact_refs"][0]
    assert "uri" in ref_item
    assert "storage://macro/" in ref_item["uri"]
    # Confirm the lean design by rejecting any large entity payload in state.
    assert "large_raw_csv_data_content" not in ref_item
