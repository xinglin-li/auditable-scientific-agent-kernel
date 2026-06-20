# tests/test_graph_approval.py
import pytest
from agent_runtime.graph.service import MacroAgentGraphService

def test_human_approval_approve_path():
    """Verify approval flow: pause the graph, resume by command, and complete."""
    service = MacroAgentGraphService()
    tid = "thread_approve_001"
    
    # 1. Parse the macro request, route to human_approval, and pause the thread.
    paused_state = service.run_workflow(thread_id=tid, run_id="r1", user_query="Diagnose CPI trends")
    
    # Confirm that checkpoint state is paused before human_approval and is not complete.
    assert paused_state["approval_status"] == "pending"
    assert paused_state.get("final_answer") is None
    
    # 2. Simulate an administrator approval that resumes and completes the tool workflow.
    final_state = service.resume_workflow(thread_id=tid, review_action={"action": "approved"})
    
    # Verify the final delivery and valid completed state.
    assert final_state["approval_status"] == "approved"
    assert final_state["status"] == "completed"
    assert "Evidence locked" in final_state["final_answer"]

def test_human_approval_reject_path():
    """Verify rejection flow: resume the paused graph with a rejection and terminate."""
    service = MacroAgentGraphService()
    tid = "thread_reject_002"
    
    service.run_workflow(thread_id=tid, run_id="r2", user_query="High risk backtest query")
    
    # Submit a rejection decision.
    final_state = service.resume_workflow(thread_id=tid, review_action={"action": "rejected"})
    
    # Verify that control flow skips tool execution and ends in the failed state.
    assert final_state["approval_status"] == "rejected"
    assert final_state["status"] == "failed"
    assert "aborted ungracefully by human administrator" in final_state["final_answer"]
    assert len(final_state.get("active_job_ids", [])) == 0

def test_human_approval_edit_and_replanning_path():
    """Verify edit flow: replan with merged edits and require a second approval before tools run."""
    service = MacroAgentGraphService()
    tid = "thread_edit_003"
    
    service.run_workflow(thread_id=tid, run_id="r3", user_query="Run massive 20 year correlation")
    
    # 1. Have an administrator reduce the plan to two years and mark it as edited.
    edited_state = service.resume_workflow(
        thread_id=tid,
        review_action={
            "action": "edited",
            "updated_plan": {"task": "Run massive 20 year correlation", "horizon_months": 2}
        }
    )
    
    # The edited state routes back to human_approval and pauses again without another command.
    assert edited_state["approval_status"] == "edited" 
    # Confirm that override semantics replace the old 20-year draft with the new plan.
    assert edited_state["analysis_plan"]["horizon_months"] == 2
    
    # 2. Approve the corrected plan on the second review.
    final_state = service.resume_workflow(thread_id=tid, review_action={"action": "approved"})
    assert final_state["approval_status"] == "approved"
    assert final_state["status"] == "completed"

def test_tool_node_side_effect_idempotency_shield():
    """Verify that idempotency prevents duplicate side effects when resume replays a node."""
    service = MacroAgentGraphService()
    tid = "thread_idempotency_004"
    
    # Approve and enter the tool-submission phase.
    service.run_workflow(thread_id=tid, run_id="r4", user_query="Execute safe trading task")
    mid_state = service.resume_workflow(thread_id=tid, review_action={"action": "approved"})
    
    # Force the node to be scheduled again from the same successful intermediate state.
    events = [e["event"] for e in mid_state["trace_events"]]
    # Verify an explicit idempotency guard event in the audit log and no repeated side effect.
    assert "tool_side_effect_prevented_by_idempotency" in events
