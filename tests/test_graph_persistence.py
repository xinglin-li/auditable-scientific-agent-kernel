# tests/test_graph_persistence.py
import pytest
from agent_runtime.graph.service import MacroAgentGraphService

def test_durable_crash_recovery_lifecycle():
    """Verify that checkpoints restore a thread's state after a service restart."""
    # Simulate the first service start with shared storage (or a temporary file for restart tests).
    # Destroy the in-memory service instance while retaining its connection string to simulate shutdown.
    db_url = "file:memdb_day02?mode=memory&cache=shared"
    
    service_v1 = MacroAgentGraphService(db_path=db_url)
    
    # Run a normal long-running request and identify its checkpoints by thread_id.
    res_v1 = service_v1.run_workflow(thread_id="thread_macro_001", run_id="run_001", user_query="Analyze CPI gaps")
    assert res_v1["status"] == "completed"
    
    # --- Simulate a crash and restart by destroying the old instance ---
    del service_v1
    
    # Reinitialize the service with the same database.
    service_v2 = MacroAgentGraphService(db_path=db_url)
    
    # The new instance must restore thread_macro_001 from storage before invoking any work.
    history = service_v2.get_state_history(thread_id="thread_macro_001")
    assert len(history) > 0
    # Confirm that the final result remains intact.
    assert "Analyze CPI gaps" in history[0]["values"]["final_answer"]

def test_multi_tenant_thread_strict_isolation():
    """Verify state isolation between two thread IDs in the same persistent store."""
    service = MacroAgentGraphService()
    
    # Thread A runs a time-series analysis.
    res_a = service.run_workflow(thread_id="tenant_user_xinglin", run_id="run_a", user_query="Xinglin task")
    # Thread B runs an unrelated direct-answer request.
    res_b = service.run_workflow(thread_id="tenant_user_hacker", run_id="run_b", user_query="Hacker task direct")
    
    # Verify that thread A contains no intent or parsed fields from thread B.
    assert res_a["parsed_request"]["raw_query"] == "Xinglin task"
    assert "Hacker" not in res_a["final_answer"]
    
    assert res_b["parsed_request"]["raw_query"] == "Hacker task direct"
