# src/agent_runtime/scientific/validators.py
import logging
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field
from agent_runtime.scientific.modelspec import ModelSpec

logger = logging.getLogger("modelspec-validator")

class Diagnostic(BaseModel):
    """Standard structured diagnostic record."""
    stage: str = "domain_validation"
    severity: str  # ERROR, WARNING, INFO
    code: str      # Standard error code from the failure taxonomy.
    message: str
    repairable: bool
    evidence: Dict[str, Any] = Field(default_factory=dict)

class ModelSpecDomainValidator:
    """Domain validator that collects all applicable diagnostics."""
    
    @staticmethod
    def validate_spec(spec: ModelSpec, mock_dataset_registry: Dict[str, Tuple[int, str]]) -> List[Diagnostic]:
        """
        Scan the full AST/IR and collect all mathematical and econometric
        violations instead of stopping at the first error.
        """
        diagnostics: List[Diagnostic] = []
        dataset_id = spec.target.dataset_id
        
        # 1. Verify that the dataset exists.
        if dataset_id not in mock_dataset_registry:
            diagnostics.append(Diagnostic(
                severity="ERROR",
                code="DATASET_NOT_FOUND",
                message=f"Dataset missing: '{dataset_id}' does not exist in the local research repository.",
                repairable=False,
                evidence={"requested_dataset_id": dataset_id}
            ))
            # Without the source dataset, later sample-size checks have no baseline.
            return diagnostics
            
        actual_rows, actual_freq = mock_dataset_registry[dataset_id]
        
        # 2. Check for frequency conflicts.
        if spec.target.frequency != actual_freq:
            diagnostics.append(Diagnostic(
                severity="ERROR",
                code="FREQUENCY_MISMATCH",
                message=f"Frequency mismatch: ModelSpec declares [{spec.target.frequency}], but the dataset frequency is [{actual_freq}].",
                repairable=True, # Frequency alignment can be repaired.
                evidence={"spec_freq": spec.target.frequency, "actual_freq": actual_freq}
            ))
            
        # 3. Validate mathematical completeness of ETS models.
        if spec.model.family == "ets":
            if spec.model.seasonal != "none" and not spec.model.seasonal_period:
                diagnostics.append(Diagnostic(
                    severity="ERROR",
                    code="INVALID_SEASONAL_PERIOD",
                    message="Invalid ETS configuration: seasonal_period is required when a seasonal component is enabled.",
                    repairable=True,
                    evidence={"family": "ets", "seasonal": spec.model.seasonal}
                ))

        # 4. Validate ARIMA order limits and complexity.
        if spec.model.family == "arima":
            p, d, q = spec.model.order
            if p < 0 or d < 0 or q < 0:
                diagnostics.append(Diagnostic(
                    severity="ERROR",
                    code="SCHEMA_VALIDATION_FAILURE",
                    message="Invalid ARIMA parameters: (p, d, q) orders cannot be negative.",
                    repairable=True,
                    evidence={"order": [p, d, q]}
                ))
            
            param_complexity = p + q
            if param_complexity > 12:
                diagnostics.append(Diagnostic(
                    severity="WARNING", # Warn about excessive complexity.
                    code="MODEL_COMPLEXITY_EXCESSIVE",
                    message=f"Excessive model complexity: non-seasonal order {param_complexity} risks non-convergence or overfitting.",
                    repairable=True,
                    evidence={"param_complexity": param_complexity}
                ))

        # 5. Check sample size against the initial backtest window.
        required_min_window = spec.backtest.initial_window + spec.backtest.horizon
        if actual_rows < required_min_window:
            diagnostics.append(Diagnostic(
                severity="ERROR",
                code="INSUFFICIENT_SAMPLE",
                message=f"Insufficient sample: {actual_rows} rows are available, but initial window {spec.backtest.initial_window} plus horizon {spec.backtest.horizon} requires at least {required_min_window} rows.",
                repairable=True, # Repair by contracting initial_window.
                evidence={"actual_rows": actual_rows, "required_rows": required_min_window}
            ))
            
        return diagnostics


def format_diagnostics_for_llm(diagnostics: List[Diagnostic]) -> str:
    """
    Format all diagnostics as a readable repair prompt for one batch correction.
    """
    if not diagnostics:
        return "SUCCESS"
        
    sb = ["### 🚨 ModelSpec compilation and econometric violations detected:\n"]
    for idx, diag in enumerate(diagnostics, start=1):
        sb.append(f"{idx}. **[{diag.severity}] Error code: {diag.code}**")
        sb.append(f"   - **Details**: {diag.message}")
        sb.append(f"   - **Evidence**: {diag.evidence}")
        sb.append(f"   - **Automatic repair allowed**: {'Yes (adjust parameters and retry convergence)' if diag.repairable else 'No (the specification logic must be reconsidered)'}\n")
        
    sb.append("---")
    sb.append("**💡 Repair instructions:**")
    sb.append("Review every warning and error above. Correct all listed violations in one pass and regenerate a complete ModelSpec JSON object that satisfies every boundary constraint.")
    
    return "\n".join(sb)
