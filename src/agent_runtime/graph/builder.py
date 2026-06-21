# src/agent_runtime/graph/builder.py
from typing import Optional, Any
from langgraph.graph import StateGraph, START, END
from agent_runtime.graph.state import MacroAgentState
from agent_runtime.graph.nodes import (
    generate_spec_node, validate_spec_node, compile_and_execute_node,
    repair_spec_node, finalize_report_node
)
from agent_runtime.graph.edges import decide_next_scientific_action, decide_after_execution

def create_macro_agent_graph(checkpointer: Optional[Any] = None):
    """
    Build a testable and controllable scientific-agent graph constrained by
    a strict intermediate representation.
    """
    workflow = StateGraph(MacroAgentState)
    
    # Register the five core protocol nodes.
    workflow.add_node("generate_spec", generate_spec_node)
    workflow.add_node("validate_spec", validate_spec_node)
    workflow.add_node("compile_and_execute", compile_and_execute_node)
    workflow.add_node("repair_spec", repair_spec_node)
    workflow.add_node("finalize_report", finalize_report_node)
    
    # Connect the graph topology.
    workflow.add_edge(START, "generate_spec")
    workflow.add_edge("generate_spec", "validate_spec")
    
    # Route valid specs, leakage failures, and repairable convergence failures.
    workflow.add_conditional_edges(
        "validate_spec",
        decide_next_scientific_action,
        {
            "repair_spec": "repair_spec",
            "compile_and_execute": "compile_and_execute",
            "finalize_report": "finalize_report"
        }
    )
    
    # Revalidate the complete repaired specification before execution.
    workflow.add_edge("repair_spec", "validate_spec")
    
    # Check for numerical failures after execution.
    workflow.add_conditional_edges(
        "compile_and_execute",
        decide_after_execution,
        {
            "repair_spec": "repair_spec",
            "finalize_report": "finalize_report"
        }
    )
    
    workflow.add_edge("finalize_report", END)
    
    return workflow.compile(checkpointer=checkpointer)
