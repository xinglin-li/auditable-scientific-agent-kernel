# src/agent_runtime/scientific/repairs.py
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from agent_runtime.scientific.modelspec import ModelSpec, ARIMASpec
from agent_runtime.scientific.validators import Diagnostic

class RepairProposal(BaseModel):
    """Auditable record of an automatic repair."""
    repair_id: str
    diagnostic_code: str
    proposed_changes: Dict[str, Any] = Field(..., description="Material parameter changes made by this repair")
    rationale: str
    attempt_number: int
    timestamp: float = Field(default_factory=time.time)

class BoundedRepairEngine:
    """Bounded repair engine with diff tracking and escalation."""

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        # Repair-diff history for the current runtime lifecycle.
        self.repair_history: List[RepairProposal] = []

    def attempt_auto_repair(self, spec: ModelSpec, diagnostic: Diagnostic) -> Optional[ModelSpec]:
        """
        Apply targeted order reduction, window contraction, or operator
        alignment to a valid ModelSpec based on a structured diagnostic.
        """
        current_attempt = len(self.repair_history) + 1
        
        # Guard 1: stop and escalate when the repair budget is exhausted.
        if current_attempt > self.max_attempts:
            logger_msg = f"[REPAIR EXHAUSTED] Repair budget exhausted after {len(self.repair_history)} attempt(s); escalating for human review."
            return None

        # Guard 2: reject non-repairable failures such as future leakage.
        if not diagnostic.repairable:
            return None

        # Strategy 1: reduce ARIMA order after non-convergence.
        if diagnostic.code == "ARIMA_NON_CONVERGENCE" and spec.model.family == "arima":
            old_p, old_d, old_q = spec.model.order
            # Reduce the model to the stable ARIMA(1, d, 1) region.
            new_order = (1, old_d, 1)
            
            # Create a strongly typed repair diff.
            proposal = RepairProposal(
                repair_id=f"rep_{diagnostic.code.lower()}_{current_attempt}",
                diagnostic_code=diagnostic.code,
                proposed_changes={"old_order": [old_p, old_d, old_q], "new_order": list(new_order)},
                rationale="Non-convergence repair: the high-order autoregressive parameter matrix is not invertible, so the system reduced model order.",
                attempt_number=current_attempt
            )
            self.repair_history.append(proposal)
            
            # Clone the candidate spec and replace the affected model branch.
            updated_spec = spec.model_copy(deep=True)
            updated_spec.model = ARIMASpec(order=new_order, seasonal_order=spec.model.seasonal_order)
            return updated_spec

        # Strategy 2: contract initial_window when the sample is too small.
        if diagnostic.code == "INSUFFICIENT_SAMPLE":
            actual_rows = diagnostic.evidence.get("actual_rows", 20)
            # Set initial_window to rows minus horizon minus a two-row buffer.
            new_window = max(10, actual_rows - spec.backtest.horizon - 2)
            
            proposal = RepairProposal(
                repair_id=f"rep_sample_{current_attempt}",
                diagnostic_code=diagnostic.code,
                proposed_changes={"old_window": spec.backtest.initial_window, "new_window": new_window},
                rationale="Insufficient-sample repair: the system contracted the initial rolling-backtest window to fit the available history.",
                attempt_number=current_attempt
            )
            self.repair_history.append(proposal)
            
            updated_spec = spec.model_copy(deep=True)
            updated_spec.backtest.initial_window = new_window
            return updated_spec

        return None
