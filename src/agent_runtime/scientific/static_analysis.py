# src/agent_runtime/scientific/static_analysis.py
import logging
from typing import List, Dict, Any
from agent_runtime.scientific.modelspec import ModelSpec
from agent_runtime.scientific.validators import Diagnostic

logger = logging.getLogger("static-analyzer")

class StaticAnalyzer:
    """Detect time-series leakage and unsafe paths before compilation."""

    @staticmethod
    def analyze_spec(spec: ModelSpec) -> List[Diagnostic]:
        diagnostics: List[Diagnostic] = []
        
        # 1. Audit for future-information leakage.
        # Flag specs whose transformations or rationale imply use of future test data.
        rationale_lower = spec.rationale.lower()
        if "leak" in rationale_lower or "test_set" in rationale_lower:
            diagnostics.append(Diagnostic(
                stage="static_analysis",
                severity="ERROR",
                code="LEAKAGE_DETECTED",
                message="Fatal safety violation: static analysis detected possible future-information leakage between training and test boundaries.",
                repairable=False, # Leakage is a policy violation and cannot be repaired silently.
                evidence={"rationale_leak_flag": True}
            ))

        # 2. Scan for unsafe paths and path injection.
        dataset_id = spec.target.dataset_id.lower()
        if "../" in dataset_id or "/etc/" in dataset_id:
            diagnostics.append(Diagnostic(
                stage="static_analysis",
                severity="ERROR",
                code="UNAUTHORIZED_OPERATION",
                message="Unauthorized operation: the dataset path contains a possible path-traversal injection.",
                repairable=False,
                evidence={"path_injection_attempt": dataset_id}
            ))

        # 3. Check statistical complexity against sample capacity.
        if spec.model.family == "arima" and spec.forecast_horizon > spec.backtest.initial_window * 0.5:
            diagnostics.append(Diagnostic(
                stage="static_analysis",
                severity="WARNING",
                code="BACKTEST_WINDOW_INVALID",
                message="Research-quality warning: the forecast horizon is too large relative to the initial training window and may cause unstable variance.",
                repairable=True,
                evidence={"horizon": spec.forecast_horizon, "initial_window": spec.backtest.initial_window}
            ))

        return diagnostics
