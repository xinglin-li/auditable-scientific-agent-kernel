# src/agent_runtime/scientific/validators.py
import logging
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field
from agent_runtime.scientific.modelspec import ModelSpec

logger = logging.getLogger("modelspec-validator")

class Diagnostic(BaseModel):
    """标准结构化诊断书实体（符合 Day 6 严密审计规约）"""
    stage: str = "domain_validation"
    severity: str  # ERROR, WARNING, INFO
    code: str      # 对应 Failure Taxonomy 字典中的标准错误码
    message: str
    repairable: bool
    evidence: Dict[str, Any] = Field(default_factory=dict)

class ModelSpecDomainValidator:
    """升级版：具备错误全量累积（Error Accumulation）能力的领域校验引擎"""
    
    @staticmethod
    def validate_spec(spec: ModelSpec, mock_dataset_registry: Dict[str, Tuple[int, str]]) -> List[Diagnostic]:
        """
        不再短路抛出单个异常，而是完整扫描整张 AST/IR 树，搜集所有深层数学与计量经济学违规
        """
        diagnostics: List[Diagnostic] = []
        dataset_id = spec.target.dataset_id
        
        # 1. 深度核对数据源存在性
        if dataset_id not in mock_dataset_registry:
            diagnostics.append(Diagnostic(
                severity="ERROR",
                code="DATASET_NOT_FOUND",
                message=f"数据源缺失：指定的物理数据集 '{dataset_id}' 在本地投研仓库中不存在。",
                repairable=False,
                evidence={"requested_dataset_id": dataset_id}
            ))
            # 如果源头文件都不存在，后续样本量检查失去基准，直接返回当前断层
            return diagnostics
            
        actual_rows, actual_freq = mock_dataset_registry[dataset_id]
        
        # 2. 交叉检查时间频率是否冲突
        if spec.target.frequency != actual_freq:
            diagnostics.append(Diagnostic(
                severity="ERROR",
                code="FREQUENCY_MISMATCH",
                message=f"数据频率冲突：ModelSpec 声明频率为 [{spec.target.frequency}]，但物理文件实际为 [{actual_freq}]。",
                repairable=True, # 允许模型降级重新对齐频率
                evidence={"spec_freq": spec.target.frequency, "actual_freq": actual_freq}
            ))
            
        # 3. 校验 ETS 模型的数学完备性
        if spec.model.family == "ets":
            if spec.model.seasonal != "none" and not spec.model.seasonal_period:
                diagnostics.append(Diagnostic(
                    severity="ERROR",
                    code="INVALID_SEASONAL_PERIOD",
                    message="ETS 模型选型冲突：当激活季节性趋势算子时，必须显式指定非空的 seasonal_period 周期数值。",
                    repairable=True,
                    evidence={"family": "ets", "seasonal": spec.model.seasonal}
                ))

        # 4. 校验 ARIMA 阶数极限与复杂度
        if spec.model.family == "arima":
            p, d, q = spec.model.order
            if p < 0 or d < 0 or q < 0:
                diagnostics.append(Diagnostic(
                    severity="ERROR",
                    code="SCHEMA_VALIDATION_FAILURE",
                    message="ARIMA 参数矩阵失效：(p, d, q) 阶数绝对不允许出现负数。",
                    repairable=True,
                    evidence={"order": [p, d, q]}
                ))
            
            param_complexity = p + q
            if param_complexity > 12:
                diagnostics.append(Diagnostic(
                    severity="WARNING", # 警告级别，提示复杂度过高
                    code="MODEL_COMPLEXITY_EXCESSIVE",
                    message=f"模型复杂度超限风险：当前非季节性参数阶数过高 ({param_complexity})，极易引发非收敛或过拟合崩溃。",
                    repairable=True,
                    evidence={"param_complexity": param_complexity}
                ))

        # 5. 计量经济学硬防线：回测样本量与初始训练窗口核对
        required_min_window = spec.backtest.initial_window + spec.backtest.horizon
        if actual_rows < required_min_window:
            diagnostics.append(Diagnostic(
                severity="ERROR",
                code="INSUFFICIENT_SAMPLE",
                message=f"样本量严重枯竭：物理文件总样本仅 {actual_rows} 行，但滚动回测初始窗口 ({spec.backtest.initial_window}) + 预测步长 ({spec.backtest.horizon}) 至少需要 {required_min_window} 行可用观测值。",
                repairable=True, # 允许收缩 initial_window 进行自我修复
                evidence={"actual_rows": actual_rows, "required_rows": required_min_window}
            ))
            
        return diagnostics


def format_diagnostics_for_llm(diagnostics: List[Diagnostic]) -> str:
    """
    将全量收集到的错误打包编译成高可读性的自修复提示词，迫使 LLM 进行单次批量修正
    """
    if not diagnostics:
        return "SUCCESS"
        
    sb = ["### 🚨 检测到 ModelSpec 存在以下编译及计量经济学领域违规：\n"]
    for idx, diag in enumerate(diagnostics, start=1):
        sb.append(f"{idx}. **[{diag.severity}] 错误码: {diag.code}**")
        sb.append(f"   - **具体信息**: {diag.message}")
        sb.append(f"   - **现场证据**: {diag.evidence}")
        sb.append(f"   - **是否允许系统自我修复**: {'是 (可通过调整相关参数尝试收敛)' if diag.repairable else '否 (必须重新推演逻辑)'}\n")
        
    sb.append("---")
    sb.append("**💡 自修复指令规约：**")
    sb.append("请认真复盘上述所有警告与错误。禁止逐个修复，你必须在下一轮输出中**一次性全量修正**所有列出的违规字段，并重新生成满足边界要求的全新标准 ModelSpec JSON 实体！")
    
    return "\n".join(sb)