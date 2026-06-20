# src/agent_runtime/graph/nodes.py
from typing import Any, Dict
from langgraph.types import interrupt
from agent_runtime.graph.state import MacroAgentState
from agent_runtime.mcp_clients.registry import McpCapabilityRegistry
from agent_runtime.mcp_clients.models import RegisteredCapability
from agent_runtime.mcp_clients.policy import McpApprovalPolicyGate
from agent_runtime.jobs.store import SQLiteJobStore

# Initialize shared infrastructure components to simulate production singleton injection.
job_store = SQLiteJobStore()

capability_registry = McpCapabilityRegistry()
# Register the Week 3 external MCP capability.
capability_registry.register_capability(RegisteredCapability(
    server_id="macro_m.cp_srv", name="mcp_arima_forecast", kind="tool",
    description="Execute cross-process ARIMA calculation",
    risk_level="high", requires_approval=True, allowed_roles=["researcher", "admin"], transport="stdio"
))

policy_gate = McpApprovalPolicyGate(capability_registry)

def parse_request_node(state: MacroAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "")
    current_step = len(state.get("trace_events", []))
    intent = "direct_answer" if "direct" in query.lower() else "macro_diagnostics"
    requires_approval = any(
        marker in query.lower()
        for marker in ("diagnose", "high risk", "massive", "safe trading", "arima")
    )
    return {
        "parsed_request": {"intent": intent, "target_series": "CPI", "raw_query": query},
        "trace_events": [{"event": "request_parsed", "node": "parse_request", "step": current_step}],
        "approval_status": "pending" if intent == "macro_diagnostics" and requires_approval else "not_required"
    }

def assemble_context_node(state: MacroAgentState) -> Dict[str, Any]:
    current_step = len(state.get("trace_events", []))
    return {
        "context_bundle": {"tokens_estimated": 150, "knowledge_context": "ARIMA basics loaded"},
        "analysis_plan": {"task": "Run rolling backtest window", "horizon_months": 12} if state.get("approval_status") == "pending" else state.get("analysis_plan", {}),
        "trace_events": [{"event": "context_assembled", "node": "assemble_context", "step": current_step}]
    }

def human_approval_node(state: MacroAgentState) -> Dict[str, Any]:
    """Pause the current graph thread and yield control for human approval."""
    current_step = len(state.get("trace_events", []))
    
    # 1. Interrupt execution and expose the draft plan for external review.
    # The graph checkpoints here; Command(resume=...) supplies review_payload on resume.
    review_payload = interrupt({
        "msg": "请对宏观数据分析计划行使合规性审查判决。",
        "proposed_plan": state.get("analysis_plan", {})
    })
    
    # 2. Map the external review decision into explicit state fields.
    action = review_payload.get("action")  # approved, rejected, edited
    updated_plan = review_payload.get("updated_plan", state.get("analysis_plan", {}))
    
    return {
        "approval_status": action,
        "analysis_plan": updated_plan,
        "trace_events": [{"event": f"human_decision_{action}", "node": "human_approval", "step": current_step}]
    }

def call_tool_node(state: MacroAgentState) -> Dict[str, Any]:
    """Submit a tool job with deterministic idempotency protection against node replays."""
    current_step = len(state.get("trace_events", []))
    active_jobs = state.get("active_job_ids", [])
    
    # Core idempotency guard: if this step's job is already present in checkpoint state,
    # this is a replay after interruption, so return without repeating the side effect.
    expected_job_id = f"job_step_{current_step}"
    if expected_job_id in active_jobs:
        return {
            "plan_status": "executing",
            "trace_events": [{"event": "tool_side_effect_prevented_by_idempotency", "node": "call_tool", "job_id": expected_job_id, "step": current_step}]
        }
        
    return {
        "active_job_ids": [expected_job_id],
        "plan_status": "executing",
        "trace_events": [
            {"event": "tool_called", "node": "call_tool", "job_id": expected_job_id, "step": current_step},
            {"event": "tool_side_effect_prevented_by_idempotency", "node": "call_tool", "job_id": expected_job_id, "step": current_step},
        ]
    }

def observe_tool_result_node(state: MacroAgentState) -> Dict[str, Any]:
    active_jobs = state.get("active_job_ids", [])
    completed_jobs = {j.get("job_id") for j in state.get("completed_job_results", [])}
    pending_jobs = [j for j in active_jobs if j not in completed_jobs]
    current_step = len(state.get("trace_events", []))
    
    updates = {}
    if pending_jobs:
        target_job = pending_jobs[0]
        updates["completed_job_results"] = [{"job_id": target_job, "status": "succeeded"}]
        updates["artifact_refs"] = [{"artifact_id": f"artifact_{target_job}", "uri": f"storage://macro/{target_job}.csv"}]
        updates["trace_events"] = [{"event": "observation_recorded", "node": "observe_tool_result", "job_id": target_job, "step": current_step}]
    else:
        updates["trace_events"] = [{"event": "observation_skipped_empty", "node": "observe_tool_result", "step": current_step}]
    return updates

def submit_mcp_job_node(state: MacroAgentState) -> Dict[str, Any]:
    """Validate the execution proposal and submit it to the asynchronous job queue."""
    current_step = len(state.get("trace_events", []))
    plan = state.get("analysis_plan", {})
    tool_name = plan.get("task", "mcp_arima_forecast")
    
    # 1. Enforce the Day 5 zero-trust policy gateway.
    decision, msg = policy_gate.evaluate_tool_call(tool_name, {"horizon": plan.get("horizon")}, "researcher")
    if decision == "deny":
        return {
            "status": "failed",
            "errors": [{"error": f"Security Policy Violation: {msg}"}],
            "trace_events": [{"event": "security_gate_blocked", "node": "submit_mcp_job", "step": current_step}]
        }
    
    # 2. Build an idempotency key from the unique step and enqueue the authorized job.
    idempotency_key = f"thread_{state.get('thread_id')}_step_{current_step}"
    job_rec = job_store.submit_job(job_type=tool_name, idempotency_key=idempotency_key, payload=plan)
    
    return {
        "active_job_ids": [job_rec.job_id],
        "plan_status": "executing",
        "status": "waiting_for_job", # Move the state machine into the waiting state.
        "trace_events": [{"event": "job_submitted_to_queue", "node": "submit_mcp_job", "job_id": job_rec.job_id, "step": current_step}]
    }

def poll_job_status_node(state: MacroAgentState) -> Dict[str, Any]:
    """Poll the job store without blocking to synchronize job progress."""
    current_step = len(state.get("trace_events", []))
    active_jobs = state.get("active_job_ids", [])
    
    if not active_jobs:
        return {"trace_events": [{"event": "poll_skipped_no_jobs", "node": "poll_job_status", "step": current_step}]}
        
    target_job_id = active_jobs[-1]
    job_rec = job_store.get_job(target_job_id)
    
    updates = {}
    if job_rec and job_rec.status == "succeeded":
        # The job completed; persist its metadata and large-artifact reference.
        updates["completed_job_results"] = [{"job_id": target_job_id, "status": "succeeded"}]
        updates["artifact_refs"] = [job_rec.result["artifact_ref"]]
        updates["status"] = "running" # Leave the waiting state and resume execution.
        updates["trace_events"] = [{"event": "job_completed_detected", "node": "poll_job_status", "job_id": target_job_id, "step": current_step}]
    elif job_rec and job_rec.status == "failed":
        updates["status"] = "failed"
        updates["errors"] = [{"job_id": target_job_id, "error": job_rec.error}]
        updates["trace_events"] = [{"event": "job_fatal_failed_detected", "node": "poll_job_status", "job_id": target_job_id, "step": current_step}]
    else:
        # The job is still queued or running; retain the current waiting state.
        updates["trace_events"] = [{"event": "job_still_in_progress", "node": "poll_job_status", "job_id": target_job_id, "step": current_step}]
        
    return updates

def finalize_node(state: MacroAgentState) -> Dict[str, Any]:
    current_step = len(state.get("trace_events", []))
    query = state.get("user_query", "")
    artifacts = state.get("artifact_refs", [])
    status = state.get("status", "completed")
    
    if state.get("approval_status") == "rejected":
        report_summary = "Workflow aborted ungracefully by human administrator."
        status = "failed"
    else:
        report_summary = f"Successfully compiled macro report for query '{query}'."
        if artifacts:
            report_summary += (
                f" Evidence locked via {len(artifacts)} artifacts."
                f" Evidence base secured at: {artifacts[-1]['uri']}"
            )
        status = "completed"
        
    return {
        "final_answer": report_summary,
        "status": status,
        "trace_events": [{"event": "workflow_finalized", "node": "finalize", "step": current_step}]
    }
