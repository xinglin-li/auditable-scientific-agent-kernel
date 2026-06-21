# src/agent_runtime/scientific/compiler.py
import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from agent_runtime.scientific.modelspec import ModelSpec

class ExecutionStep(BaseModel):
    """Executable plan step with evaluation audit hooks."""
    step_id: str
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list, description="IDs of prerequisite plan steps")
    timeout_seconds: int = 20
    idempotency_key: str
    # Evaluation-specific audit anchor.
    expected_artifact_output: str = Field(..., description="Expected materialized feature or numeric result path")

class ExecutionPlan(BaseModel):
    """Complete deterministic execution plan."""
    plan_id: str
    spec_version: str
    steps: List[ExecutionStep]
    artifact_outputs: List[str]
    requires_human_approval: bool

class ModelSpecCompiler:
    """Compile a declarative model specification into a controlled pipeline."""

    @staticmethod
    def compile_plan(spec: ModelSpec) -> ExecutionPlan:
        steps: List[ExecutionStep] = []
        plan_id = f"plan_{uuid.uuid4().hex[:10]}"
        
        # Step 1: load the dataset.
        step1_id = f"step_1_load_{spec.target.dataset_id[:6]}"
        steps.append(ExecutionStep(
            step_id=step1_id,
            operation="load_dataset",
            parameters={"dataset_id": spec.target.dataset_id, "column": spec.target.column},
            idempotency_key=f"idem_{plan_id}_load",
            expected_artifact_output=f"artifacts/{plan_id}/raw_series.pkl"
        ))

        # Step 2: apply stationarity transformations.
        step2_id = f"step_2_transform"
        steps.append(ExecutionStep(
            step_id=step2_id,
            operation="apply_transformations",
            parameters={"transforms": [t.model_dump() for t in spec.transformations]},
            depends_on=[step1_id],
            idempotency_key=f"idem_{plan_id}_trans",
            expected_artifact_output=f"artifacts/{plan_id}/stationary_series.pkl"
        ))

        # Step 3: fit the statistical model.
        step3_id = f"step_3_fit_{spec.model.family}"
        steps.append(ExecutionStep(
            step_id=step3_id,
            operation=f"fit_{spec.model.family}",
            parameters={"model_spec": spec.model.model_dump(), "backtest_spec": spec.backtest.model_dump()},
            depends_on=[step2_id],
            idempotency_key=f"idem_{plan_id}_fit",
            expected_artifact_output=f"artifacts/{plan_id}/fitted_metrics.json" # Numerical source for downstream graders.
        ))

        # Step 4: generate the forecast and evidence report.
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
