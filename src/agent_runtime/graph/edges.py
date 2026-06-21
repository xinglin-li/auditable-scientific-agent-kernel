# src/agent_runtime/graph/edges.py
from typing import Literal
from agent_runtime.graph.state import MacroAgentState

def decide_next_scientific_action(state: MacroAgentState) -> Literal["repair_spec", "compile_and_execute", "finalize_report"]:
    """
    Route the graph from the current diagnostic snapshot.
    """
    diags_list = state.get("diagnostics", [])
    
    if not diags_list:
        # Compile when validation succeeds or repaired errors have been cleared.
        return "compile_and_execute"
        
    # Inspect the most recent diagnostic.
    latest_diag = diags_list[-1]
    severity = latest_diag.get("severity")
    repairable = latest_diag.get("repairable", False)
    
    # Non-repairable errors such as future leakage terminate execution.
    if severity == "ERROR" and not repairable:
        return "finalize_report"
        
    # Route repairable statistical failures to the repair node.
    if severity == "ERROR" and repairable:
        return "repair_spec"
        
    # Warnings do not block execution.
    return "compile_and_execute"


def decide_after_execution(state: MacroAgentState) -> Literal["repair_spec", "finalize_report"]:
    """Route after the model-fitting operator completes."""
    diags = state.get("diagnostics", [])
    # Execution diagnostics must reach the repair node before validation
    # replaces the current diagnostic snapshot.
    if diags and diags[-1].get("severity") == "ERROR":
        return "repair_spec"
    return "finalize_report"
