# src/agent_runtime/evals/failure_episode.py
import uuid
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ---- Failure Taxonomy 工业级标准失败分类字典 ----
FAILURE_TOOL_SELECTION = "tool_selection_failure"       # 工具选错或幻觉了不存在的工具
FAILURE_TOOL_ARGUMENT = "tool_argument_failure"         # 工具参数类型错误或缺失核心字段
FAILURE_FUTURE_LEAKAGE = "future_leakage_failure"       # 时间序列建模中发生了严重的未来数据泄漏
FAILURE_MATH_CONVERGENCE = "math_convergence_failure"   # 统计模型拟合时数值爆炸或不收敛
FAILURE_SECURITY_VIOLATION = "security_violation_failure" # 触发了权限网关拦截或发生数据外泄
FAILURE_MAX_STEPS_EXCEEDED = "max_steps_exceeded_failure" # 在有限步长内未能收敛，陷入死循环
FAILURE_UNKNOWN = "unknown_failure"                     # 其它未分类的运行时崩溃

class FailureEpisode(BaseModel):
    """离线 Reflexion 与提示词优化最核心的黄金资产：失败片段实体"""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    run_id: str
    trial_id: str
    failure_type: str = FAILURE_UNKNOWN                 # 必须映射到上面的分类字典
    error_message: Optional[str] = None
    failed_node: Optional[str] = Field(None, description="崩溃发生时所处的 LangGraph 节点或工具名")
    failed_step: int = 0
    trace_summary: List[str] = Field(default_factory=list, description="精简版的时序行为事件摘要")
    root_cause_hypothesis: Optional[str] = Field(None, description="根据 Trace 现场自动生成的根因假设")
    candidate_patches: List[str] = Field(default_factory=list, description="离线学习层推荐的修复 Prompt 或规约")
    created_at: float = Field(default_factory=time.time)