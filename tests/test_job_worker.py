# tests/test_job_worker.py
import pytest
import asyncio
from agent_runtime.jobs.store import SQLiteJobStore
from agent_runtime.jobs.worker import AsyncMacroJobWorker

@pytest.mark.asyncio
async def test_job_store_idempotency_and_retry_backoff_lifecycle():
    """Verify strict idempotency and the lifecycle of a job that exhausts its retries."""
    store = SQLiteJobStore()
    worker = AsyncMacroJobWorker(store)
    await worker.start()
    
    try:
        # 1. Submit the same analysis job twice using one idempotency key.
        job_v1 = store.submit_job(job_type="arima", idempotency_key="key_cpi_2026", payload={"series": "CPI"})
        job_v2 = store.submit_job(job_type="arima", idempotency_key="key_cpi_2026", payload={"series": "CPI"})
        
        # Confirm that the second submission returns the original job ID without duplicate work.
        assert job_v1.job_id == job_v2.job_id
        
        # 2. Submit a job designed to fail convergence with max_attempts=2.
        fail_job = store.submit_job(job_type="backtest", idempotency_key="key_fail", payload={"force_fail": True})
        fail_job.max_attempts = 2
        
        # Wait for the worker to claim, retry, and finally fail the job.
        await asyncio.sleep(0.1)
        
        final_job = store.get_job(fail_job.job_id)
        assert final_job.status == "failed"
        # Verify that exactly two attempts ran before the terminal failure.
        assert final_job.attempt_count == 2
        assert "convergence error" in final_job.error
        
    finally:
        await worker.stop()
