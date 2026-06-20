# src/agent_runtime/memory/condenser.py
"""Memory condensation: convert verbose AgentState into compact EpisodeRecord.

The condenser produces an EpisodeRecord that can be stored as episodic memory
for future retrieval. It extracts summary, key decisions, errors, and outcome
from a complete AgentState, discarding the raw message history.

This is the V2 "memory condensation" step that prevents unbounded growth
of raw trace storage.
"""
import uuid
from typing import List

from agent_runtime.memory.models import EpisodeRecord
from agent_runtime.runtime.state import AgentState


class MemoryCondenser:
    """Condense a completed AgentState into a storable EpisodeRecord."""

    @staticmethod
    def condense(state: AgentState) -> EpisodeRecord:
        """Build a compact EpisodeRecord from a finished run.

        Args:
            state: A completed (or terminated) AgentState.

        Returns:
            EpisodeRecord with task summary, key decisions, errors, and outcome.
        """
        # task summary: user's original query
        user_msg = next(
            (m.content for m in state.messages if m.role == "user"),
            "Unknown task",
        )

        # outcome maps from agent status
        outcome_map = {
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "max_steps_exceeded": "max_steps_exceeded",
        }
        outcome = outcome_map.get(state.status, state.status)

        # key decisions: summary of each tool call from AgentStep trajectory
        key_decisions: List[str] = []
        for step in state.steps:
            if step.action:
                decision = f"called {step.action.tool_name} (call_id={step.action.call_id})"
                if step.rationale:
                    decision += f": {step.rationale.summary}"
                key_decisions.append(decision)

        # errors extracted from tool observations
        error_messages: List[str] = []
        for step in state.steps:
            if step.observation and not step.observation.ok and step.observation.error:
                error_msg = f"{step.observation.tool_name}: {step.observation.error.get('message', 'unknown error')}"
                error_messages.append(error_msg)

        return EpisodeRecord(
            episode_id=str(uuid.uuid4()),
            run_id=state.run_id,
            task_summary=user_msg or "No user message recorded.",
            outcome=outcome,
            key_decisions=key_decisions,
            error_messages=error_messages,
            step_count=state.step_count,
        )