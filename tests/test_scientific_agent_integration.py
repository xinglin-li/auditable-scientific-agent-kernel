# tests/test_scientific_agent_integration.py
import pytest
from langgraph.checkpoint.memory import MemorySaver
from agent_runtime.graph.builder import create_macro_agent_graph

def test_scientific_agent_leakage_hard_gate_violation():
    """Verify that future-information leakage terminates the graph immediately."""
    checkpointer = MemorySaver()
    app = create_macro_agent_graph(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_thread_leak"}}
    initial_state = {
        "user_query": "Run estimation with future leakage metrics.", # Triggers the static leakage detector.
        "repair_attempts": 0
    }
    
    final_state = app.invoke(initial_state, config=config)
    
    # The hard gate must fail before an execution plan is compiled.
    assert final_state["status"] == "failed"
    assert final_state["execution_plan"] is None
    
    events = [e["event"] for e in final_state["trace_events"]]
    assert "spec_validation_completed" in events
    assert "modelspec_generated" in events

def test_scientific_agent_numerical_failure_auto_repair_lifecycle():
    """
    Exercise the complete lifecycle: high-order non-convergence, execution
    failure, bounded order reduction, clean retry, and audited completion.
    """
    checkpointer = MemorySaver()
    app = create_macro_agent_graph(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_thread_repair_loop"}}
    initial_state = {
        "user_query": "Please fit high_order ARIMA pipelines.", # Intentionally triggers ARIMA(4,0,4) non-convergence.
        "repair_attempts": 0
    }
    
    final_state = app.invoke(initial_state, config=config)
    
    # 1. The state machine completes after bounded repair.
    assert final_state["status"] == "completed"
    
    # 2. The final ModelSpec is reduced to ARIMA(1,0,1).
    final_spec = final_state["model_spec"]
    assert final_spec.model.order == (1, 0, 1)
    
    # 3. The counter and diff history record exactly one repair.
    assert final_state["repair_attempts"] == 1
    assert len(final_state["repair_history"]) == 1
    assert final_state["repair_history"][0]["diagnostic_code"] == "ARIMA_NON_CONVERGENCE"
    
    # 4. The final report includes provenance and repair disclosure.
    assert "Research Approval Report" in final_state["final_answer"]
    assert "order-reduction repair audit" in final_state["final_answer"]
    
    # Print the ordered trace for manual inspection.
    print("\n--- Scientific Agent Audit Trace ---")
    for event in final_state["trace_events"]:
        print(f"Step [{event.get('step')}] -> Node: {event.get('node'):<20} | Event: {event.get('event')}")
