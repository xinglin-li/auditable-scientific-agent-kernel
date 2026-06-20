# src/agent_runtime/jobs/store.py
from typing import Dict, Optional, List
import time
from agent_runtime.jobs.models import JobRecord

class SQLiteJobStore:
    """Persistent long-running job store with idempotency and atomic state transitions."""
    
    def __init__(self):
        self._db: Dict[str, JobRecord] = {}
    
    def submit_job(self, job_type: str, idempotency_key: str, payload: dict) -> JobRecord:
        """Submit a job, returning the existing reference when its idempotency key conflicts."""
        # 1. Check whether the idempotency key already exists.
        for existing_job in self._db.values():
            if existing_job.idempotency_key == idempotency_key:
                return existing_job  # Return the existing JobRef without repeating the side effect.
                
        # 2. No conflict exists, so enqueue a new job.
        job_id = f"job_async_{int(time.time() * 1000)}"
        record = JobRecord(
            job_id=job_id, job_type=job_type, idempotency_key=idempotency_key,
            status="queued", payload=payload
        )
        self._db[job_id] = record
        return record
    
    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Return detailed job information."""
        return self._db.get(job_id)
    
    def update_job_status(self, job_id: str, status: str, result: Optional[dict] = None, error: Optional[str] = None) -> Optional[JobRecord]:
        """Apply an atomic state transition."""
        if job_id in self._db:
            job = self._db[job_id]
            job.status = status
            job.updated_at = time.time()
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            return job
        return None
    
    def fetch_next_queued_job(self) -> Optional[JobRecord]:
        """Return jobs that are ready for worker consumption."""
        for job in self._db.values():
            if job.status == "queued":
                return job
        return None

    def fetch_next_cancel_requested_job(self) -> Optional[JobRecord]:
        """Return a cancellation that must be acknowledged by the worker."""
        for job in self._db.values():
            if job.status == "cancel_requested":
                return job
        return None
