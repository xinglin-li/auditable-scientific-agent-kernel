# src/agent_runtime/graph/edges.py
from typing import Literal
from agent_runtime.graph.state import MacroAgentState

def decide_next_action(state: MacroAgentState) -> Literal["human_approval", "submit_mcp_job", "poll_job_status", "finalize", "max_steps_hit"]:
    trace_count = len(state.get("trace_events", []))
    if trace_count >= 25:
        return "max_steps_hit"
        
    if state.get("status") == "failed":
        return "finalize"
        
    parsed = state.get("parsed_request", {})
    if parsed.get("intent") == "direct_answer":
        return "finalize"
        
    # 1. Check the human-approval state first.
    approval = state.get("approval_status")
    if approval == "pending":
        return "human_approval"
    if approval == "rejected":
        return "finalize"
    if approval == "edited":
        return "human_approval"
        
    # 2. Determine progress from the asynchronous job-set difference.
    active_jobs = state.get("active_job_ids", [])
    completed_job_ids = {j.get("job_id") for j in state.get("completed_job_results", [])}
    pending_jobs = [j for j in active_jobs if j not in completed_job_ids]
    
    if not active_jobs:
        # Submit the first job after human approval.
        return "submit_mcp_job"
        
    if pending_jobs:
        # Route submitted but incomplete jobs to the polling node.
        return "poll_job_status"
        
    # All long-running jobs are complete; proceed to final delivery.
    return "finalize"


def decide_after_job_activity(state: MacroAgentState) -> Literal["continue", "finalize", "pause"]:
    """Leave pending jobs checkpointed instead of busy-polling in one invocation."""
    if state.get("status") == "failed":
        return "finalize"
    if state.get("status") == "waiting_for_job":
        return "pause"
    return "continue"
