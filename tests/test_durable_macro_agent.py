# tests/test_durable_macro_agent.py
import pytest
import asyncio
from agent_runtime.graph.service import MacroAgentGraphService
from agent_runtime.graph.nodes import job_store
from agent_runtime.jobs.worker import AsyncMacroJobWorker

@pytest.mark.asyncio
async def test_durable_macro_research_agent_grand_lifecycle_and_crash_recovery():
    """Week 3 end-to-end integration test.
    Flow: request -> human edit -> approval -> async job -> polling pause -> crash -> recovery.
    """
    db_url = "file:memdb_grand_final_2026?mode=memory&cache=shared"
    tid = "grand_final_thread_999"
    
    # 1. Initialize the first runtime instance and start its long-lived worker.
    service_v1 = MacroAgentGraphService(db_path=db_url)
    worker = AsyncMacroJobWorker(job_store)
    await worker.start()
    
    try:
        # 2. Trigger the long-running workflow.
        state_step1 = service_v1.run_workflow(thread_id=tid, run_id="run_g1", user_query="Run full rolling ARIMA for sales")
        # Verify that the workflow first pauses at the human_approval boundary.
        assert state_step1["approval_status"] == "pending"
        assert state_step1["status"] == "running"
        
        # 3. Simulate an expressive human edit and replanning action.
        state_step2 = service_v1.resume_workflow(thread_id=tid, review_action={
            "action": "edited", "updated_plan": {"task": "mcp_arima_forecast", "horizon": 3}
        })
        # Verify that the new plan overrides the old one and pauses for a second decision.
        assert state_step2["approval_status"] == "edited"
        assert state_step2["analysis_plan"]["horizon"] == 3
        
        # 4. Submit formal approval on the second review.
        state_step3 = service_v1.resume_workflow(thread_id=tid, review_action={"action": "approved"})
        
        # Confirm that policy checks pass, an async job is submitted, and the graph waits for it.
        assert state_step3["status"] == "waiting_for_job"
        assert len(state_step3["active_job_ids"]) == 1
        target_job_id = state_step3["active_job_ids"][-1]
        
        # 5. Inject a hard service failure while the asynchronous worker continues computing.
        del service_v1
        
        # Let the worker finish and persist the successful result and artifact reference.
        await asyncio.sleep(0.08)
        assert job_store.get_job(target_job_id).status == "succeeded"
        
        # 6. Reinitialize the service to simulate a restart.
        service_v2 = MacroAgentGraphService(db_path=db_url)
        
        # Invoke again with the same thread_id. The runtime should restore its checkpoint,
        # observe the completed job through polling, and proceed to delivery.
        final_state = service_v2.run_workflow(thread_id=tid, run_id="run_g2", user_query="Resume placeholder")
        
        # 7. Verify complete recovery without lost state.
        assert final_state["status"] == "completed"
        assert "Evidence base secured at: storage://macro/" in final_state["final_answer"]
        # Verify that the deduplicating reducer preserves the final artifact reference.
        assert len(final_state["artifact_refs"]) == 1
        assert final_state["artifact_refs"][0]["artifact_id"] == f"art_{target_job_id}"
        
        # 8. Verify a clear, auditable, and replayable state history.
        history = service_v2.get_state_history(thread_id=tid)
        assert len(history) >= 4
        
    finally:
        await worker.stop()
