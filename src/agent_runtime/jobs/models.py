# src/agent_runtime/jobs/models.py
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
import time

class JobRecord(BaseModel):
    """Canonical job-store entity for the underlying task."""
    job_id: str
    job_type: str
    idempotency_key: str  # Primary key for strict idempotency enforcement.
    status: Literal["queued", "running", "succeeded", "failed", "cancel_requested", "cancelled"]
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 3
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
