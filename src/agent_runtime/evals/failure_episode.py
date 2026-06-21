# src/agent_runtime/evals/failure_episode.py
import uuid
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ---- Standard failure taxonomy ----
FAILURE_TOOL_SELECTION = "tool_selection_failure"       # Selected or hallucinated an invalid tool.
FAILURE_TOOL_ARGUMENT = "tool_argument_failure"         # Tool arguments are invalid or missing required fields.
FAILURE_FUTURE_LEAKAGE = "future_leakage_failure"       # Time-series modeling leaked future information.
FAILURE_MATH_CONVERGENCE = "math_convergence_failure"   # Statistical fitting diverged or failed to converge.
FAILURE_SECURITY_VIOLATION = "security_violation_failure" # A policy gate blocked the action or data leaked.
FAILURE_MAX_STEPS_EXCEEDED = "max_steps_exceeded_failure" # Execution failed to converge within the step limit.
FAILURE_UNKNOWN = "unknown_failure"                     # Any other unclassified runtime failure.

class FailureEpisode(BaseModel):
    """A compact failure record used by offline Reflexion and prompt optimization."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    run_id: str
    trial_id: str
    failure_type: str = FAILURE_UNKNOWN                 # Must map to the taxonomy above.
    error_message: Optional[str] = None
    failed_node: Optional[str] = Field(None, description="LangGraph node or tool where the failure occurred")
    failed_step: int = 0
    trace_summary: List[str] = Field(default_factory=list, description="Compact chronological event summary")
    root_cause_hypothesis: Optional[str] = Field(None, description="Root-cause hypothesis generated from the trace")
    candidate_patches: List[str] = Field(default_factory=list, description="Repair prompts or policies proposed offline")
    created_at: float = Field(default_factory=time.time)
