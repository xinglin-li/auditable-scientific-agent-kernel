# tests/test_context_pruning.py
"""V2 Context pruning tests: stalenes, overflow cut, and lost-in-the-middle mitigation."""

from agent_runtime.context.assembler import ContextAssembler
from agent_runtime.context.models import ContextItem, ContextBudgetReport


def test_pruning_removes_stale_low_priority_items():
    """Items with age_in_steps > max_age and low priority should be pruned."""
    items = [
        ContextItem(item_id="sys", kind="system_instruction", content="System rule.", priority=100, trust_level="system_trusted"),
        ContextItem(item_id="fresh_user", kind="user_message", content="Current query.", priority=95, trust_level="user_supplied", age_in_steps=0),
        ContextItem(item_id="stale_tool", kind="tool_result", content="Old tool result.", priority=30, trust_level="system_trusted", age_in_steps=15),
    ]
    assembler = ContextAssembler(max_tokens=1000)
    bundle, report = assembler.assemble(items)

    retained_ids = [i.item_id for i in bundle.items]
    assert "sys" in retained_ids
    assert "fresh_user" in retained_ids
    # stale low-priority item should be pruned
    assert "stale_tool" not in retained_ids
    assert report.items_pruned >= 1


def test_budget_report_trust_distribution():
    """BudgetReport should correctly count items by trust_level."""
    items = [
        ContextItem(item_id="sys", kind="system_instruction", content="Rule.", priority=100, trust_level="system_trusted"),
        ContextItem(item_id="user", kind="user_message", content="Query.", priority=95, trust_level="user_supplied"),
    ]
    assembler = ContextAssembler(max_tokens=1000)
    _, report = assembler.assemble(items)

    assert isinstance(report, ContextBudgetReport)
    assert report.trust_distribution.get("system_trusted", 0) == 1
    assert report.trust_distribution.get("user_supplied", 0) == 1


def test_budget_report_kind_distribution():
    """BudgetReport should correctly count items by kind."""
    items = [
        ContextItem(item_id="sys", kind="system_instruction", content="Rule.", priority=100, trust_level="system_trusted"),
        ContextItem(item_id="rag", kind="retrieved_evidence", content="Evidence.", priority=70, trust_level="retrieved_untrusted"),
        ContextItem(item_id="user", kind="user_message", content="Query.", priority=95, trust_level="user_supplied"),
    ]
    assembler = ContextAssembler(max_tokens=1000)
    _, report = assembler.assemble(items)

    assert isinstance(report, ContextBudgetReport)
    assert report.kind_distribution.get("system_instruction", 0) == 1
    assert report.kind_distribution.get("retrieved_evidence", 0) == 1
    assert report.kind_distribution.get("user_message", 0) == 1


def test_lost_in_middle_warning_on_high_utilization():
    """Lost-in-the-middle warning should trigger when >90% utilized with many items."""
    # Create many items to trigger lost-in-the-middle
    items = [ContextItem(item_id="sys", kind="system_instruction", content="System.", priority=100, trust_level="system_trusted")]
    for i in range(12):
        items.append(
            ContextItem(
                item_id=f"msg_{i}",
                kind="user_message" if i % 2 == 0 else "tool_result",
                content=f"Message content number {i}.",
                priority=50,
                trust_level="user_supplied",
            )
        )
    # Tight budget to force >90% utilization
    assembler = ContextAssembler(max_tokens=50)
    _, report = assembler.assemble(items)

    assert isinstance(report, ContextBudgetReport)
    # With a tight budget and many items, utilization should be high
    assert report.utilization_ratio >= 0.9
    assert report.lost_in_middle_warning is True


def test_assembler_returns_bundle_and_report():
    """assemble() should return a tuple of (ContextBundle, ContextBudgetReport)."""
    items = [
        ContextItem(item_id="1", kind="system_instruction", content="You are safe.", priority=100, trust_level="system_trusted"),
    ]
    assembler = ContextAssembler()
    result = assembler.assemble(items)

    from agent_runtime.context.models import ContextBundle, ContextBudgetReport
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], ContextBundle)
