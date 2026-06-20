# tests/test_working_memory.py
"""V2 Working Memory tests: model construction, budget critical, remaining steps."""

from agent_runtime.context.models import ContextBudgetReport
from agent_runtime.memory.working_memory import WorkingMemory


def test_default_working_memory():
    """WorkingMemory should initialize with sensible defaults."""
    wm = WorkingMemory(run_id="run-001")
    assert wm.run_id == "run-001"
    assert wm.namespace == "default_project"
    assert wm.max_steps == 5
    assert wm.current_step == 0
    assert wm.remaining_steps() == 5
    assert wm.pending_tool_calls == []
    assert wm.active_skill_name is None
    assert wm.is_budget_critical() is False


def test_remaining_steps_decreases_with_current_step():
    """remaining_steps should reflect (max_steps - current_step)."""
    wm = WorkingMemory(run_id="run-002", max_steps=10, current_step=3)
    assert wm.remaining_steps() == 7

    wm.current_step = 10
    assert wm.remaining_steps() == 0

    wm.current_step = 12
    assert wm.remaining_steps() == 0  # never negative


def test_budget_critical_when_high_utilization():
    """is_budget_critical should return True when >90% utilized AND lost_in_middle."""
    # normal: no report -> not critical
    wm = WorkingMemory(run_id="run-003")
    assert wm.is_budget_critical() is False

    # high utilization but no lost-in-middle warning -> not critical
    report_safe = ContextBudgetReport(
        total_items_submitted=10,
        items_after_dedup=10,
        items_pruned=0,
        items_dropped_by_budget=0,
        items_retained=10,
        estimated_tokens_used=950,
        max_tokens=1000,
        utilization_ratio=0.95,
        lost_in_middle_warning=False,
    )
    wm.last_budget_report = report_safe
    assert wm.is_budget_critical() is False

    # high utilization WITH lost-in-middle -> critical
    report_critical = ContextBudgetReport(
        total_items_submitted=10,
        items_after_dedup=10,
        items_pruned=0,
        items_dropped_by_budget=0,
        items_retained=10,
        estimated_tokens_used=950,
        max_tokens=1000,
        utilization_ratio=0.95,
        lost_in_middle_warning=True,
    )
    wm.last_budget_report = report_critical
    assert wm.is_budget_critical() is True


def test_snapshot_contains_key_fields():
    """snapshot() should return a dict with essential working memory fields."""
    wm = WorkingMemory(
        run_id="run-004",
        namespace="finance",
        max_token_budget=2000,
        active_skill_name="rolling-backtest",
        current_step=2,
        pending_tool_calls=["c1", "c2"],
        evidence_sources=["doc_arima_basics"],
    )

    snap = wm.snapshot()
    assert isinstance(snap, dict)
    assert snap["run_id"] == "run-004"
    assert snap["namespace"] == "finance"
    assert snap["current_step"] == 2
    assert snap["remaining_steps"] == 3
    assert snap["max_token_budget"] == 2000
    assert snap["active_skill"] == "rolling-backtest"
    assert snap["budget_critical"] is False
    assert snap["pending_tool_calls"] == 2
    assert snap["evidence_sources"] == 1