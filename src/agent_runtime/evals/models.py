# src/agent_runtime/evals/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvalTask(BaseModel):
    """评估任务的基本定义"""
    task_id: str = Field(..., description="任务的唯一确定性标识符，如 task_arima_001")
    name: str = Field(..., description="任务的可读名称")
    user_input: str = Field(..., description="输入给 Agent 的原始自然语言查询")
    expected_outcome: Dict[str, Any] = Field(
        default_factory=dict, 
        description="预期对环境造成的改变，例如：{'file_exists': 'series/monthly_sales.csv'}"
    )
    trajectory_rules: List[str] = Field(
        default_factory=list, 
        description="硬编码的轨迹审查规则标识符，如 ['require_approval_before_mcp', 'no_untrusted_injection']"
    )
    limits: Dict[str, Any] = Field(
        default_factory=dict, 
        description="执行约束，如 max_steps, batch_timeout"
    )
    
class GraderResult(BaseModel):
    """单个评分器对单次 Trial 的判定结果"""
    grader_name: str
    passed: bool
    score: float = Field(1.0, ge=0.0, le=1.0, description="得分，必须在 0 到 1 之间")
    reason: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

class TrialResult(BaseModel):
    """单次独立执行（Trial）捕获的完整上下文快照"""
    trial_id: str
    task_id: str
    run_id: str
    status: str  # completed, failed, max_steps_exceeded, cancelled
    final_answer: Optional[str] = None
    trace_events: List[Dict[str, Any]] = Field(default_factory=list, description="扁平的时序事件链桩")
    grader_results: List[GraderResult] = Field(default_factory=list, description="多维度裁判结果集")
    duration_ms: float = 0.0
    step_count: int = 0