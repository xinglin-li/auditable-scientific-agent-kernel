# src/agent_runtime/graph/builder.py
from typing import Optional, Any
from langgraph.graph import StateGraph, START, END
from agent_runtime.graph.state import MacroAgentState
from agent_runtime.graph.nodes import (
    parse_request_node, assemble_context_node, human_approval_node,
    submit_mcp_job_node, poll_job_status_node, finalize_node
)
from agent_runtime.graph.edges import decide_after_job_activity, decide_next_action

def create_macro_agent_graph(checkpointer: Optional[Any] = None):
    workflow = StateGraph(MacroAgentState)
    
    workflow.add_node("parse_request", parse_request_node)
    workflow.add_node("assemble_context", assemble_context_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("submit_mcp_job", submit_mcp_job_node)
    workflow.add_node("poll_job_status", poll_job_status_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.add_edge(START, "parse_request")
    workflow.add_edge("parse_request", "assemble_context")
    
    # Attach the central conditional routing edge.
    workflow.add_conditional_edges(
        "assemble_context",
        decide_next_action,
        {
            "human_approval": "human_approval",
            "submit_mcp_job": "submit_mcp_job",
            "poll_job_status": "poll_job_status",
            "finalize": "finalize",
            "max_steps_hit": END
        }
    )
    
    # Return every computation and loop node to the router for another state check.
    workflow.add_edge("human_approval", "assemble_context")
    job_routes = {
        "continue": "assemble_context",
        "finalize": "finalize",
        "pause": END,
    }
    workflow.add_conditional_edges("submit_mcp_job", decide_after_job_activity, job_routes)
    workflow.add_conditional_edges("poll_job_status", decide_after_job_activity, job_routes)
    workflow.add_edge("finalize", END)
    
    return workflow.compile(checkpointer=checkpointer)
