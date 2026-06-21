# src/agent_runtime/scientific/executor.py
from typing import Dict, Any, Callable
from agent_runtime.scientific.compiler import ExecutionPlan, ExecutionStep

class DeterministicExecutor:
    """Deterministic runtime constrained by an approved operation registry."""

    def __init__(self):
        # Allowlist of approved operation handlers.
        self.registry: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "load_dataset": lambda p: {"status": "SUCCESS", "rows": 120},
            "apply_transformations": lambda p: {"status": "SUCCESS", "stationarity": "PASSED"},
            "fit_arima": self.execute_fit_arima, # Internal handler with numerical diagnostics.
            "generate_forecast": lambda p: {"status": "SUCCESS", "horizon_aligned": True}
        }

    def execute_fit_arima(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate numerical boundaries of the statistical engine."""
        model_spec = params.get("model_spec", {})
        p, d, q = model_spec.get("order", (0, 0, 0))
        
        # Inject non-convergence when the combined ARIMA order exceeds five.
        if p + q > 5:
            return {
                "status": "NUMERICAL_FAILURE",
                "error_code": "ARIMA_NON_CONVERGENCE",
                "detail": "Numerical failure: initial MA coefficients are non-invertible or non-stationary, and ARIMA maximum-likelihood estimation did not converge."
            }
        return {"status": "SUCCESS", "rmse": 1.15, "converged": True}

    def execute_plan_step(self, step: ExecutionStep) -> Dict[str, Any]:
        op = step.operation
        if op not in self.registry:
            return {
                "status": "FATAL",
                "error_code": "UNAUTHORIZED_OPERATION",
                "detail": f"Kernel gate blocked unapproved operation [{op}]."
            }
        
        # Dispatch to the deterministic handler.
        return self.registry[op](step.parameters)
