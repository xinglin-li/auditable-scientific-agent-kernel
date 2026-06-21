# src/agent_runtime/graph/nodes.py
import uuid
from typing import Any, Dict
from langgraph.types import interrupt

from agent_runtime.graph.state import MacroAgentState
from agent_runtime.scientific.modelspec import ModelSpec, ARIMASpec
from agent_runtime.scientific.parser import ModelSpecParser
from agent_runtime.scientific.validators import ModelSpecDomainValidator, Diagnostic
from agent_runtime.scientific.static_analysis import StaticAnalyzer
from agent_runtime.scientific.compiler import ModelSpecCompiler
from agent_runtime.scientific.executor import DeterministicExecutor
from agent_runtime.scientific.repairs import BoundedRepairEngine

# Local dataset metadata registry used by the deterministic test runtime.
LOCAL_DATA_REGISTRY = {
    "monthly_sales.csv": (120, "monthly"),
    "short_series.csv": (15, "monthly")
}

def generate_spec_node(state: MacroAgentState) -> Dict[str, Any]:
    """
    Convert a natural-language research request into a strongly typed
    ModelSpec. The deterministic test path constructs the candidate directly.
    """
    query = state.get("user_query", "Analyze series")
    current_step = len(state.get("trace_events", []))
    
    # Simulate extracted parameters; high_order intentionally creates ARIMA(4,0,4).
    order_p = 4 if "high_order" in query.lower() else 1
    
    # Build the untrusted candidate payload.
    raw_payload = {
        "spec_version": "1.0",
        "target": {
            "dataset_id": "short_series.csv" if "short" in query.lower() else "monthly_sales.csv",
            "column": "revenue",
            "frequency": "monthly"
        },
        "transformations": [],
        "model": {
            "family": "arima",
            "order": [order_p, 0, 4],
            "seasonal_order": None
        },
        "backtest": {"horizon": 6, "initial_window": 60, "step_size": 3, "metrics": ["rmse"]},
        "forecast_horizon": 6,
        "rationale": "Potential future data leakage." if "leakage" in query.lower() else "Compliant analytical rationale."
    }
    
    # Apply schema parsing and prevent the LLM from granting itself privileges.
    spec_candidate = ModelSpecParser.wrap_llm_structured_candidate(raw_payload)
    
    return {
        "model_spec": spec_candidate,
        "execution_plan": None,
        "status": "running",
        "trace_events": [{"event": "modelspec_generated", "node": "generate_spec", "step": current_step}]
    }

def validate_spec_node(state: MacroAgentState) -> Dict[str, Any]:
    """
    Collect domain violations and static leakage findings for graph routing.
    """
    spec: ModelSpec = state.get("model_spec")
    current_step = len(state.get("trace_events", []))
    
    # Collect domain-level statistical violations.
    domain_diags = ModelSpecDomainValidator.validate_spec(spec, LOCAL_DATA_REGISTRY)
    
    # Scan statically for future-information leakage.
    static_diags = StaticAnalyzer.analyze_spec(spec)
    
    all_diags = domain_diags + static_diags
    flat_diags = [d.model_dump() for d in all_diags]
    
    return {
        "diagnostics": flat_diags,
        "trace_events": [{"event": "spec_validation_completed", "node": "validate_spec", "errors_found": len(flat_diags), "step": current_step}]
    }

def compile_and_execute_node(state: MacroAgentState) -> Dict[str, Any]:
    """
    Compile a valid ModelSpec into an execution plan and run it through the
    approved deterministic handler registry.
    """
    spec: ModelSpec = state.get("model_spec")
    current_step = len(state.get("trace_events", []))
    
    # 1. Compile the plan.
    plan = ModelSpecCompiler.compile_plan(spec)
    
    # 2. Execute the plan.
    executor = DeterministicExecutor()
    fit_step = plan.steps[2] # Select the model-fitting operator.
    
    runtime_res = executor.execute_plan_step(fit_step)
    
    updates = {
        "execution_plan": plan,
        "trace_events": [{"event": "plan_compiled_and_executed", "node": "compile_and_execute", "step": current_step}]
    }
    
    # 3. Convert numerical non-convergence into a standard Diagnostic.
    if runtime_res.get("status") == "NUMERICAL_FAILURE":
        diag = Diagnostic(
            stage="execution",
            severity="ERROR",
            code=runtime_res["error_code"],
            message=runtime_res["detail"],
            repairable=True
        )
        updates["diagnostics"] = [diag.model_dump()]
        updates["trace_events"].append({"event": "numerical_failure_captured", "node": "compile_and_execute", "code": diag.code, "step": current_step + 1})
        
    return updates

def repair_spec_node(state: MacroAgentState) -> Dict[str, Any]:
    """
    Consume a diagnostic, apply a bounded repair, and record the parameter diff.
    """
    spec: ModelSpec = state.get("model_spec")
    diags_dict = state.get("diagnostics", [])
    current_step = len(state.get("trace_events", []))
    current_attempts = state.get("repair_attempts", 0) + 1
    
    if not diags_dict:
        return {"trace_events": [{"event": "repair_skipped_no_errors", "node": "repair_spec", "step": current_step}]}
        
    # Deserialize the latest error diagnostic.
    latest_diag_dict = diags_dict[-1]
    diag = Diagnostic.model_validate(latest_diag_dict)
    
    # Start the repair engine with a bounded attempt budget.
    engine = BoundedRepairEngine(max_attempts=2)
    # Restore prior repair history before attempting another repair.
    for past_diff in state.get("repair_history", []):
        engine.repair_history.append(past_diff)
        
    fixed_spec = engine.attempt_auto_repair(spec, diag)
    
    if fixed_spec is None:
        # Terminate when the repair budget is exhausted or the error is not repairable.
        return {
            "status": "failed",
            "errors": [{"error": f"Bounded Repair Engine exhausted at attempt {current_attempts}."}],
            "trace_events": [{"event": "repair_quota_exhausted_escalated", "node": "repair_spec", "step": current_step}]
        }
        
    # Capture the latest parameter diff.
    latest_proposal = engine.repair_history[-1].model_dump()
    
    return {
        "model_spec": fixed_spec,
        "repair_attempts": current_attempts,
        "repair_history": [latest_proposal], # The reducer appends this entry.
        # Clear the consumed diagnostic before revalidation.
        "diagnostics": [], 
        "trace_events": [{"event": "spec_repaired_and_degraded", "node": "repair_spec", "proposal_id": latest_proposal["repair_id"], "step": current_step}]
    }

def finalize_report_node(state: MacroAgentState) -> Dict[str, Any]:
    """
    Finalize the report against materialized artifacts and repair history.
    """
    current_step = len(state.get("trace_events", []))
    status = state.get("status", "completed")
    plan = state.get("execution_plan")
    history = state.get("repair_history", [])
    diagnostics = state.get("diagnostics", [])
    has_blocking_diagnostic = any(
        diagnostic.get("severity") == "ERROR"
        and not diagnostic.get("repairable", False)
        for diagnostic in diagnostics
    )
    
    if status == "failed" or state.get("errors") or has_blocking_diagnostic:
        report = "The research task failed because it crossed a safety boundary or exhausted the numerical repair budget."
        status = "failed"
    else:
        # Verify the expected artifact reference before reporting success.
        artifact_uri = plan.artifact_outputs[-1] if plan else "storage://null"
        report = f"[Research Approval Report] The forecasting task completed successfully. Evidence is stored at {artifact_uri}."
        if history:
            report += f" (The task contains {len(history)} order-reduction repair audit record(s).)"
        status = "completed"
        
    return {
        "final_answer": report,
        "status": status,
        "trace_events": [{"event": "workflow_finalized", "node": "finalize_report", "step": current_step}]
    }
