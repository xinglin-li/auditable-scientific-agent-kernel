# tests/test_context_budget.py
from agent_runtime.context.assembler import ContextAssembler
from agent_runtime.context.models import ContextItem


def test_context_within_budget():
    """All context items should be retained when the total size is under budget."""
    items = [
        ContextItem(item_id="1", kind="system_instruction", content="You are a system.", priority=100, trust_level="system_trusted"),
        ContextItem(item_id="2", kind="user_message", content="Help me buy a stock.", priority=95, trust_level="user_supplied"),
    ]
    assembler = ContextAssembler(max_tokens=1000)
    bundle, report = assembler.assemble(items)

    assert len(bundle.items) == 2
    assert len(bundle.dropped_items) == 0
    # V2: budget report should be produced
    assert report.items_retained == 2
    assert report.utilization_ratio < 1.0


def test_context_over_budget_truncation():
    """Low-priority retrieved evidence should be dropped first when the budget is tight."""
    items = [
        ContextItem(item_id="sys", kind="system_instruction", content="Keep safe.", priority=100, trust_level="system_trusted"),
        ContextItem(item_id="user", kind="user_message", content="Do task.", priority=95, trust_level="user_supplied"),
        ContextItem(item_id="rag_junk", kind="retrieved_evidence", content="Junk text " * 100, priority=20, trust_level="retrieved_untrusted"),
    ]

    assembler = ContextAssembler(max_tokens=30)
    bundle, report = assembler.assemble(items)

    retained_ids = [item.item_id for item in bundle.items]
    assert "sys" in retained_ids
    assert "user" in retained_ids

    dropped_ids = [item.item_id for item in bundle.dropped_items]
    assert "rag_junk" in dropped_ids
    # V2: budget report
    assert report.items_dropped_by_budget == 1


def test_system_instruction_is_undroppable():
    """Priority 100 system instructions must be retained even when they exceed budget."""
    items = [
        ContextItem(item_id="sys_heavy", kind="system_instruction", content="System Rule " * 50, priority=100, trust_level="system_trusted"),
        ContextItem(item_id="low_rag", kind="retrieved_evidence", content="Some text", priority=10, trust_level="retrieved_untrusted"),
    ]

    assembler = ContextAssembler(max_tokens=10)
    bundle, report = assembler.assemble(items)

    assert "sys_heavy" in [item.item_id for item in bundle.items]
    assert "low_rag" in [item.item_id for item in bundle.dropped_items]
    # V2: system instructions survive with priority >= 100
    assert report.items_retained >= 1
