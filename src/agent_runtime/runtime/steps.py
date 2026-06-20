# src/agent_runtime/runtime/steps.py
"""AgentStep builder for the V2 Structured ReAct runtime."""

from agent_runtime.models import (
    AgentStep,
    PlannerRationale,
    ToolAction,
    ToolObservation,
    StopReason,
)
from typing import Optional


class AgentStepBuilder:
    def __init__(self, run_id: str, step: int) -> None:
        self._run_id = run_id
        self._step = step
        self._rationale: Optional[PlannerRationale] = None
        self._action: Optional[ToolAction] = None
        self._observation: Optional[ToolObservation] = None
        self._stop_reason: Optional[StopReason] = None

    def set_rationale(self, rationale: PlannerRationale) -> None:
        self._rationale = rationale

    def set_action(self, action: ToolAction) -> None:
        self._action = action

    def set_observation(self, observation: ToolObservation) -> None:
        self._observation = observation

    def set_stop_reason(self, stop_reason: StopReason) -> None:
        self._stop_reason = stop_reason

    def build(self) -> AgentStep:
        return AgentStep(
            run_id=self._run_id,
            step=self._step,
            rationale=self._rationale,
            action=self._action,
            observation=self._observation,
            stop_reason=self._stop_reason,
        )

    @staticmethod
    def build_direct_answer_step(
        run_id: str,
        step: int,
        rationale: Optional[PlannerRationale] = None,
    ) -> AgentStep:
        builder = AgentStepBuilder(run_id, step)
        if rationale:
            builder.set_rationale(rationale)
        builder.set_stop_reason(
            StopReason(reason="final_answer", step=step)
        )
        return builder.build()

    @staticmethod
    def build_stop_step(
        run_id: str,
        step: int,
        reason: str,
        message: Optional[str] = None,
    ) -> AgentStep:
        builder = AgentStepBuilder(run_id, step)
        builder.set_stop_reason(
            StopReason(reason=reason, message=message, step=step)
        )
        return builder.build()