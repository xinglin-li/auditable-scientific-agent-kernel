# tests/test_job_cancellation.py
import pytest
import asyncio
from agent_runtime.jobs.store import SQLiteJobStore
from agent_runtime.jobs.worker import AsyncMacroJobWorker

@pytest.mark.asyncio
async def test_job_graceful_cancellation_via_heartbeat_checkpoint():
    """Verify cooperative cancellation at a worker heartbeat checkpoint."""
    store = SQLiteJobStore()
    worker = AsyncMacroJobWorker(store)
    await worker.start()
    
    try:
        # 1. Submit a normal job.
        job = store.submit_job(job_type="heavy_loop", idempotency_key="key_cancel", payload={"loop": True})
        
        # 2. Simulate a client cancellation while the job is running.
        store.update_job_status(job.job_id, "cancel_requested")
        
        # Give the worker enough time to reach its cancellation checkpoint.
        await asyncio.sleep(0.08)
        
        # 3. Confirm that the worker cleans up and transitions the job to cancelled without deleting it.
        final_job = store.get_job(job.job_id)
        assert final_job.status == "cancelled"
        assert "Cancelled gracefully" in final_job.error
        
    finally:
        await worker.stop()
