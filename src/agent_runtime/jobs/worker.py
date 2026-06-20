# src/agent_runtime/jobs/worker.py
import asyncio
from typing import Callable, Dict, Any, Optional
from agent_runtime.jobs.store import SQLiteJobStore

class AsyncMacroJobWorker:
    """Long-lived asynchronous worker with polling, retries, and cooperative cancellation."""
    
    def __init__(self, store: SQLiteJobStore):
        self.store = store
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the long-lived background consumer loop."""
        self._running = True
        self._loop_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        """Shut down the worker gracefully."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()

    async def _process_loop(self):
        """Run the persistent polling loop."""
        while self._running:
            cancelled_job = self.store.fetch_next_cancel_requested_job()
            if cancelled_job:
                self.store.update_job_status(
                    cancelled_job.job_id,
                    "cancelled",
                    error="Cancelled gracefully by human request.",
                )

            job = self.store.fetch_next_queued_job()
            if job:
                # 1. Claim the job and lock its state as running.
                self.store.update_job_status(job.job_id, "running")
                job.attempt_count += 1
                
                # 2. Dispatch the long-running macro computation.
                asyncio.create_task(self._execute_job_core(job.job_id))
            await asyncio.sleep(0.01) # Avoid CPU-intensive busy waiting.

    async def _execute_job_core(self, job_id: str):
        """Simulate a resource-intensive time-series model computation."""
        job = self.store.get_job(job_id)
        if not job:
            return

        try:
            # --- Simulated compute-intensive phase one ---
            await asyncio.sleep(0.02)
            
            # Cooperative-cancellation checkpoint between compute-intensive phases.
            # Release resources and exit as soon as a cancellation request is observed.
            current_job = self.store.get_job(job_id)
            if current_job and current_job.status == "cancel_requested":
                self.store.update_job_status(job_id, "cancelled", error="Cancelled gracefully by human request.")
                return

            # --- Simulated phase two, including transient network or convergence failures ---
            if job.payload.get("force_fail"):
                raise RuntimeError("Transient math convergence error.")

            # Computation succeeded; return a reference to the large result artifact.
            artifact_ref = {"artifact_id": f"art_{job_id}", "uri": f"storage://macro/processed_{job_id}.csv"}
            self.store.update_job_status(job_id, "succeeded", result={"summary": "ARIMA rolling done.", "artifact_ref": artifact_ref})
            
        except Exception as e:
            # 3. Determine whether the failed job is eligible for exponential-backoff retry.
            if job.attempt_count < job.max_attempts:
                # Retry budget remains; requeue the job for a later polling cycle.
                self.store.update_job_status(job_id, "queued")
            else:
                # Retry budget is exhausted; transition to the terminal failed state.
                self.store.update_job_status(job_id, "failed", error=str(e))
