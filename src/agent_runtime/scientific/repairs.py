# src/agent_runtime/scientific/repairs.py
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from agent_runtime.scientific.modelspec import ModelSpec, ARIMASpec
from agent_runtime.scientific.validators import Diagnostic

class RepairProposal(BaseModel):
    """自动修复历史审计存根（不准静默作弊）"""
    repair_id: str
    diagnostic_code: str
    proposed_changes: Dict[str, Any] = Field(..., description="本次自修复对参数的物理改动覆盖面")
    rationale: str
    attempt_number: int
    timestamp: float = Field(default_factory=time.time)

class BoundedRepairEngine:
    """工业级受限修复发动机：实现修改留痕、有界收敛与熔断升级机制"""

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        # 物理挂载在当前运行时生命周期内的自修复差分日志链
        self.repair_history: List[RepairProposal] = []

    def attempt_auto_repair(self, spec: ModelSpec, diagnostic: Diagnostic) -> Optional[ModelSpec]:
        """
        根据结构化诊断书，对合法的 ModelSpec 树实施针对性降阶、收缩或算子对齐
        """
        current_attempt = len(self.repair_history) + 1
        
        # 红线 1：次数超限，强行挂起并升级给人类，打破无限死循环的 Token 泄露
        if current_attempt > self.max_attempts:
            logger_msg = f"[REPAIR EXHAUSTED] 修复次数耗尽 ({len(self.repair_history)}), 升级交付投研经理进行人工干预。"
            return None

        # 红线 2：如果是不可静默修复的红线故障（如未来泄漏），立刻拒绝并阻断
        if not diagnostic.repairable:
            return None

        # 核心策略 1：处理经典的 ARIMA_NON_CONVERGENCE 不收敛故障 -> 自动化柔性降阶
        if diagnostic.code == "ARIMA_NON_CONVERGENCE" and spec.model.family == "arima":
            old_p, old_d, old_q = spec.model.order
            # 暴力压低高阶自回归，强行逼近稳定平稳区间 ARIMA(1, d, 1)
            new_order = (1, old_d, 1)
            
            # 构造强类型 Diff 日志存根
            proposal = RepairProposal(
                repair_id=f"rep_{diagnostic.code.lower()}_{current_attempt}",
                diagnostic_code=diagnostic.code,
                proposed_changes={"old_order": [old_p, old_d, old_q], "new_order": list(new_order)},
                rationale="统计不收敛自修复：高阶自回归估计参数矩阵不 invertible，系统实施柔性降阶防御机制。",
                attempt_number=current_attempt
            )
            self.repair_history.append(proposal)
            
            # 进行原地树枝属性覆写，克隆全新的 Candidate Spec 并返回
            updated_spec = spec.model_copy(deep=True)
            updated_spec.model = ARIMASpec(order=new_order, seasonal_order=spec.model.seasonal_order)
            return updated_spec

        # 核心策略 2：处理样本量枯竭风险 -> 自动收缩 initial_window 对齐真实物理总样本
        if diagnostic.code == "INSUFFICIENT_SAMPLE":
            actual_rows = diagnostic.evidence.get("actual_rows", 20)
            # 动态压缩回测窗口以满足数学拟合下限：initial_window = 物理行数 - horizon - 2
            new_window = max(10, actual_rows - spec.backtest.horizon - 2)
            
            proposal = RepairProposal(
                repair_id=f"rep_sample_{current_attempt}",
                diagnostic_code=diagnostic.code,
                proposed_changes={"old_window": spec.backtest.initial_window, "new_window": new_window},
                rationale="数据枯竭自修复：历史物理样本短缺，自适应压缩滚动回测初始基准训练窗口。",
                attempt_number=current_attempt
            )
            self.repair_history.append(proposal)
            
            updated_spec = spec.model_copy(deep=True)
            updated_spec.backtest.initial_window = new_window
            return updated_spec

        return None