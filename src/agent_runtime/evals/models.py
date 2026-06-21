# src/agent_runtime/evals/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvalTask(BaseModel):
    """Definition of an evaluation task."""
    task_id: str = Field(..., description="Unique deterministic task identifier, such as task_arima_001")
    name: str = Field(..., description="Human-readable task name")
    user_input: str = Field(..., description="Original natural-language query sent to the agent")
    expected_outcome: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Expected external state change, such as {'file_exists': 'series/monthly_sales.csv'}"
    )
    trajectory_rules: List[str] = Field(
        default_factory=list, 
        description="Trajectory rule identifiers, such as ['require_approval_before_mcp', 'no_untrusted_injection']"
    )
    limits: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Execution limits, such as max_steps and batch_timeout"
    )
    
class GraderResult(BaseModel):
    """Result produced by one grader for one trial."""
    grader_name: str
    passed: bool
    score: float = Field(1.0, ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    reason: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

class TrialResult(BaseModel):
    """Complete context snapshot captured for one isolated trial."""
    trial_id: str
    task_id: str
    run_id: str
    status: str  # completed, failed, max_steps_exceeded, cancelled
    final_answer: Optional[str] = None
    trace_events: List[Dict[str, Any]] = Field(default_factory=list, description="Flat chronological event trace")
    grader_results: List[GraderResult] = Field(default_factory=list, description="Multidimensional grader results")
    duration_ms: float = 0.0
    step_count: int = 0
