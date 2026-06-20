# tests/test_memory_episodic.py
"""V2 Episodic memory tests: condensation + storage + retrieval."""

from agent_runtime.memory.condenser import MemoryCondenser
from agent_runtime.memory.models import EpisodeRecord, MEMORY_TYPE_EPISODIC
from agent_runtime.memory.sqlite_store import SQLiteMemoryStore
from agent_runtime.runtime.state import AgentState
from agent_runtime.models import AgentMessage, ToolCall, PlannerRationale, ToolAction, ToolObservation, StopReason, AgentStep


def _make_state_with_tool_call():
    """Build a realistic completed AgentState for condensation testing."""
    return AgentState(
        run_id="run-001",
        messages=[
            AgentMessage(role="user", content="What is 2 + 3?"),
        ],
        step_count=2,
        status="completed",
        stop_reason="final_answer",
        final_answer="The result of 2 + 3 is 5.",
        steps=[
            AgentStep(
                run_id="run-001",
                step=1,
                rationale=PlannerRationale(
                    summary="Need to add 2 and 3 per user request.",
                    confidence=0.95,
                ),
                action=ToolAction(
                    call_id="call_001",
                    tool_name="add_numbers",
                    arguments={"a": 2, "b": 3},
                ),
                observation=ToolObservation(
                    call_id="call_001",
                    tool_name="add_numbers",
                    ok=True,
                    output={"result": 5},
                ),
            ),
            AgentStep(
                run_id="run-001",
                step=2,
                stop_reason=StopReason(reason="final_answer", step=2),
            ),
        ],
    )


def _make_state_with_error():
    """Build a failed AgentState for condensation testing."""
    return AgentState(
        run_id="run-002",
        messages=[
            AgentMessage(role="user", content="Run unknown tool."),
        ],
        step_count=1,
        status="failed",
        stop_reason="non_retryable_error",
        steps=[
            AgentStep(
                run_id="run-002",
                step=1,
                action=ToolAction(
                    call_id="c_fatal",
                    tool_name="unknown_tool",
                    arguments={},
                ),
                observation=ToolObservation(
                    call_id="c_fatal",
                    tool_name="unknown_tool",
                    ok=False,
                    error={"message": "Unknown tool: 'unknown_tool'"},
                ),
            ),
        ],
    )


def test_condense_produces_valid_episode():
    """MemoryCondenser should produce a well-formed EpisodeRecord."""
    state = _make_state_with_tool_call()
    episode = MemoryCondenser.condense(state)

    assert isinstance(episode, EpisodeRecord)
    assert episode.run_id == "run-001"
    assert episode.task_summary == "What is 2 + 3?"
    assert episode.outcome == "completed"
    assert episode.step_count == 2
    # At least one key decision captured
    assert len(episode.key_decisions) > 0
    assert any("add_numbers" in d for d in episode.key_decisions)
    # No errors in a successful run
    assert len(episode.error_messages) == 0


def test_condense_captures_key_decisions():
    """Episodes from successful runs should contain tool call summaries."""
    state = _make_state_with_tool_call()
    episode = MemoryCondenser.condense(state)

    decisions = episode.key_decisions
    assert any("add_numbers" in d for d in decisions)
    # Rationale should be captured
    assert any("Need to add 2 and 3" in d for d in decisions)


def test_condense_extracts_errors():
    """Episodes from failed runs should capture error messages."""
    state = _make_state_with_error()
    episode = MemoryCondenser.condense(state)

    assert episode.outcome == "failed"
    assert len(episode.error_messages) > 0
    assert any("unknown_tool" in e for e in episode.error_messages)


def test_condensed_episode_to_memory_record_roundtrip():
    """EpisodeRecord -> MemoryRecord -> EpisodeRecord should be lossless."""
    state = _make_state_with_tool_call()
    episode = MemoryCondenser.condense(state)

    # Convert to MemoryRecord and back
    record = episode.to_memory_record(namespace="test-ns")
    assert record.memory_type == "episodic"
    assert record.namespace == "test-ns"

    restored = EpisodeRecord.from_memory_record(record)
    assert restored.episode_id == episode.episode_id
    assert restored.run_id == episode.run_id
    assert restored.task_summary == episode.task_summary
    assert restored.outcome == episode.outcome
    assert restored.step_count == episode.step_count
    assert restored.key_decisions == episode.key_decisions


def test_sqlite_store_episodic_crud():
    """SQLiteMemoryStore should store and retrieve episodic memories."""
    store = SQLiteMemoryStore()
    try:
        state = _make_state_with_tool_call()
        episode = MemoryCondenser.condense(state)
        record = episode.to_memory_record(namespace="test-ns")

        # Store the episode
        store.put_episode(record)

        # Retrieve by namespace
        episodes = store.list_episodes_by_namespace("test-ns")
        assert len(episodes) == 1
        assert episodes[0].memory_type == "episodic"
        assert episodes[0].content == "What is 2 + 3?"
        assert episodes[0].metadata["run_id"] == "run-001"

        # Non-episodic namespaces return empty
        assert len(store.list_episodes_by_namespace("other-ns")) == 0
    finally:
        store.close()