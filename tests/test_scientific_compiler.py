# tests/test_scientific_compiler.py
import pytest
from agent_runtime.scientific.modelspec import ModelSpec
from agent_runtime.scientific.static_analysis import StaticAnalyzer
from agent_runtime.scientific.compiler import ModelSpecCompiler
from agent_runtime.scientific.executor import DeterministicExecutor
from agent_runtime.scientific.repairs import BoundedRepairEngine
from agent_runtime.scientific.validators import Diagnostic

@pytest.fixture
def base_arima_payload():
    """Return an uncompiled high-order model payload."""
    return {
        "spec_version": "1.0",
        "target": {"dataset_id": "monthly_sales.csv", "column": "revenue", "frequency": "monthly"},
        "transformations": [],
        "model": {
            "family": "arima",
            "order": [4, 0, 4], # 4+4 > 5 triggers the executor's non-convergence stub.
            "seasonal_order": None
        },
        "backtest": {"horizon": 6, "initial_window": 60, "step_size": 3, "metrics": ["rmse"]},
        "forecast_horizon": 6,
        "rationale": "Fit a stationary high-order model to the target time series."
    }

def test_static_analyzer_blocks_future_data_leakage(base_arima_payload):
    """Verify that future-information leakage is fatal and non-repairable."""
    leaked_payload = base_arima_payload.copy()
    leaked_payload["rationale"] = "Malicious leakage injection: use future observations from test_set to optimize the estimate."
    
    spec = ModelSpec.model_validate(leaked_payload)
    logs = StaticAnalyzer.analyze_spec(spec)
    
    assert len(logs) == 1
    assert logs[0].code == "LEAKAGE_DETECTED"
    assert logs[0].repairable is False # A hard policy violation cannot be auto-repaired.

def test_end_to_end_compiler_and_bounded_repair_loop(base_arima_payload):
    """
    Exercise the full lifecycle: high-order non-convergence, execution failure,
    automatic order reduction, successful retry, and complete diff tracking.
    """
    spec = ModelSpec.model_validate(base_arima_payload)
    
    # Create the compiler, controlled runtime, and repair engine.
    compiler = ModelSpecCompiler()
    executor = DeterministicExecutor()
    repair_engine = BoundedRepairEngine(max_attempts=2)
    
    # --- First execution pass ---
    plan = compiler.compile_plan(spec)
    assert len(plan.steps) == 4
    # Verify topological dependencies.
    assert plan.steps[2].depends_on == ["step_2_transform"]
    
    # Execute the core fitting step.
    fit_step = plan.steps[2]
    res1 = executor.execute_fit_arima(fit_step.parameters)
    
    # Confirm that ARIMA(4,0,4) causes non-convergence on the first pass.
    assert res1["status"] == "NUMERICAL_FAILURE"
    assert res1["error_code"] == "ARIMA_NON_CONVERGENCE"
    
    # Convert the runtime failure into a standard Diagnostic.
    diag = Diagnostic(
        stage="execution", severity="ERROR", 
        code=res1["error_code"], message=res1["detail"], repairable=True
    )
    
    # --- Apply one automatic order-reduction repair ---
    fixed_spec = repair_engine.attempt_auto_repair(spec, diag)
    assert fixed_spec is not None
    # The parameters are reduced safely to (1, 0, 1).
    assert fixed_spec.model.order == (1, 0, 1) 
    
    # --- Compile and execute the clean second pass ---
    new_plan = compiler.compile_plan(fixed_spec)
    new_fit_step = new_plan.steps[2]
    res2 = executor.execute_fit_arima(new_fit_step.parameters)
    
    # The lower-order model converges successfully.
    assert res2["status"] == "SUCCESS"
    assert res2["converged"] is True
    
    # The repair history must contain an explicit parameter diff.
    assert len(repair_engine.repair_history) == 1
    diff_log = repair_engine.repair_history[0]
    assert diff_log.proposed_changes["old_order"] == [4, 0, 4]
    assert diff_log.proposed_changes["new_order"] == [1, 0, 1]

def test_repair_exhaustion_escalates_to_human(base_arima_payload):
    """Verify that the repair engine stops after exhausting its budget."""
    spec = ModelSpec.model_validate(base_arima_payload)
    repair_engine = BoundedRepairEngine(max_attempts=1) # Allow only one repair.
    
    diag = Diagnostic(stage="execution", severity="ERROR", code="ARIMA_NON_CONVERGENCE", message="Crash", repairable=True)
    
    # The first repair is allowed.
    spec_v2 = repair_engine.attempt_auto_repair(spec, diag)
    assert spec_v2 is not None
    
    # The second attempt exhausts the budget and returns None.
    spec_v3 = repair_engine.attempt_auto_repair(spec_v2, diag)
    assert spec_v3 is None
    assert len(repair_engine.repair_history) == 1
