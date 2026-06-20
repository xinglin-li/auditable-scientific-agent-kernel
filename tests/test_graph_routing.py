# tests/test_graph_routing.py
import pytest
from agent_runtime.graph.builder import create_macro_agent_graph

def test_direct_answer_routing_path():
    """Verify direct routing to finalize when a request needs no external tool."""
    graph = create_macro_agent_graph()
    
    # Use the 'direct' trigger to select the direct_answer intent.
    initial_state = {"user_query": "Please give me a direct definition of CPI."}
    
    result = graph.invoke(initial_state)
    
    assert result["status"] == "completed"
    assert "direct definition of CPI" in result["final_answer"]
    # Verify that no unnecessary external job was submitted.
    assert len(result.get("active_job_ids", [])) == 0
    # Verify that trace_events contain the minimal direct-route audit trail.
    events = [e["event"] for e in result["trace_events"]]
    assert "request_parsed" in events
    assert "context_assembled" in events
    assert "workflow_finalized" in events

def test_tool_loop_and_collection_difference_resolution():
    """Verify the complete macro-job loop, set-difference tracking, and final delivery."""
    graph = create_macro_agent_graph()
    initial_state = {"user_query": "Analyze macro unemployment statistics."}
    
    result = graph.invoke(initial_state)
    
    # 1. Verify that set-difference routing reaches the terminal completed state.
    assert result["status"] == "completed"
    assert "unemployment statistics" in result["final_answer"]
    
    # 2. Verify that reducers append unique entries without losing or overwriting history.
    assert len(result["active_job_ids"]) == 1
    assert len(result["completed_job_results"]) == 1
    assert len(result["artifact_refs"]) == 1
    
    # 3. Verify that submitted and completed job IDs match exactly.
    assert result["active_job_ids"][0] == result["completed_job_results"][0]["job_id"]
    assert result["artifact_refs"][0]["artifact_id"] == f"artifact_{result['active_job_ids'][0]}"

def test_max_steps_hard_guardrail_melting():
    """Verify that the hard guard terminates a loop when execution exceeds its limit."""
    # Seed state with ten trace events to simulate reaching the execution limit.
    malicious_state = {
        "user_query": "Loop forever please.",
        "trace_events": [{"event": "fake_loop", "step": i} for i in range(10)]
    }
    
    graph = create_macro_agent_graph()
    result = graph.invoke(malicious_state)
    
    # Verify that the conditional edge terminates after assemble_context without reaching finalize.
    assert result.get("status") is None  # Execution stopped at END without assigning finalized status.
    assert result.get("final_answer") is None
    # Verify that the append-only reducer preserves the original ten audit events on termination.
    assert len(result["trace_events"]) == 12  # Includes the new parse_request and assemble_context events.
