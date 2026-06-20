# src/agent_runtime/memory/working_memory.py
"""V2 Working Memory: explicit model for the agent's current cognitive state.

Working memory captures what the agent currently knows and is constrained by
during a single run. It is distinct from:
  - AgentState (the full state machine with messages and steps)
  - MemoryRecord (persistent long-term storage)
  - EpisodeRecord (condensed past run snapshot)

Contents:
  - Current run identity and namespace
  - Task constraints (max_steps, token budget, concurrency)
  - Active context bundle snapshot
  - Active skill
  - Current step / loop position
  - Trust and priority boundaries
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from agent_runtime.context.models import ContextBudgetReport


class WorkingMemory(BaseModel):
    """The agent's active cognitive state for one run.

    Built at run start and updated on each loop iteration.
    Provides a structured view of "what the agent currently works with".
    """

    # -- identity --
    run_id: str
    namespace: str = "default_project"

    # -- task constraints --
    max_steps: int = 5
    max_token_budget: int = 1500
    max_tool_retries: int = 3
    max_concurrency: int = 1

    # -- active context --
    active_skill_name: Optional[str] = None
    active_trust_boundary: str = "retrieved_untrusted"  # strictest trust_level in context
    last_budget_report: Optional[ContextBudgetReport] = None

    # -- loop position --
    current_step: int = 0
    pending_tool_calls: List[str] = Field(default_factory=list)  # call_ids

    # -- provenance --
    evidence_sources: List[str] = Field(default_factory=list)
    activated_citations: List[str] = Field(default_factory=list)

    def is_budget_critical(self) -> bool:
        """Return True if token budget utilization is dangerously high."""
        if self.last_budget_report is None:
            return False
        return (
            self.last_budget_report.utilization_ratio > 0.9
            and self.last_budget_report.lost_in_middle_warning
        )

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.current_step)

    def snapshot(self) -> dict:
        """Return a minimal dict suitable for logging / trace recording."""
        return {
            "run_id": self.run_id,
            "namespace": self.namespace,
            "current_step": self.current_step,
            "remaining_steps": self.remaining_steps(),
            "max_token_budget": self.max_token_budget,
            "active_skill": self.active_skill_name,
            "budget_critical": self.is_budget_critical(),
            "pending_tool_calls": len(self.pending_tool_calls),
            "evidence_sources": len(self.evidence_sources),
        }