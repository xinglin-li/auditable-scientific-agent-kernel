# src/agent_runtime/scientific/compiler.py
import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from agent_runtime.scientific.modelspec import ModelSpec

class ExecutionStep(BaseModel):
    """物理可执行单元计划（符合 Eval 审计钩子设计）"""
    step_id: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list, description="前置拓扑依赖的步骤 ID 列表")
    timeout_seconds: int = 20
    idempotency_key: str
    # 你的核心解法：Eval 评估专属标签锚点
    expected_artifact_output: str = Field(..., description="这一步预期会落地的物理物理特征或数值结果路径")

class ExecutionPlan(BaseModel):
    """物理执行计划总案"""
    plan_id: str
    spec_version: str
    steps: List[ExecutionStep]
    artifact_outputs: List[str]
    requires_human_approval: bool

class ModelSpecCompiler:
    """拓扑计划编译器：将描述性数学规格实体翻译为底层受控分布式管道"""

    @staticmethod
    def compile_plan(spec: ModelSpec) -> ExecutionPlan:
        steps: List[ExecutionStep] = []
        plan_id = f"plan_{uuid.uuid4().hex[:10]}"
        
        # 步骤 1：拉取并解包物理数据集
        step1_id = f"step_1_load_{spec.target.dataset_id[:6]}"
        steps.append(ExecutionStep(
            step_id=step1_id,
            operation="load_dataset",
            parameters={"dataset_id": spec.target.dataset_id, "column": spec.target.column},
            idempotency_key=f"idem_{plan_id}_load",
            expected_artifact_output=f"artifacts/{plan_id}/raw_series.pkl"
        ))

        # 步骤 2：对数或差分平稳化特征算子变换
        step2_id = f"step_2_transform"
        steps.append(ExecutionStep(
            step_id=step2_id,
            operation="apply_transformations",
            parameters={"transforms": [t.model_dump() for t in spec.transformations]},
            depends_on=[step1_id],
            idempotency_key=f"idem_{plan_id}_trans",
            expected_artifact_output=f"artifacts/{plan_id}/stationary_series.pkl"
        ))

        # 步骤 3：核心统计模型管道拟合与求解
        step3_id = f"step_3_fit_{spec.model.family}"
        steps.append(ExecutionStep(
            step_id=step3_id,
            operation=f"fit_{spec.model.family}",
            parameters={"model_spec": spec.model.model_dump(), "backtest_spec": spec.backtest.model_dump()},
            depends_on=[step2_id],
            idempotency_key=f"idem_{plan_id}_fit",
            expected_artifact_output=f"artifacts/{plan_id}/fitted_metrics.json" # 后置 Grader 的关键数值重算来源
        ))

        # 步骤 4：生成前瞻预测与证据链核对报告
        step4_id = f"step_4_forecast"
        steps.append(ExecutionStep(
            step_id=step4_id,
            operation="generate_forecast",
            parameters={"horizon": spec.forecast_horizon},
            depends_on=[step3_id],
            idempotency_key=f"idem_{plan_id}_fore",
            expected_artifact_output=f"artifacts/{plan_id}/forecast_report.csv"
        ))

        return ExecutionPlan(
            plan_id=plan_id,
            spec_version=spec.spec_version,
            steps=steps,
            artifact_outputs=[s.expected_artifact_output for s in steps],
            requires_human_approval=spec.require_human_approval
        )