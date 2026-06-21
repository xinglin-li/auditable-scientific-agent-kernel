# tests/test_eval_dataset.py
import pytest
from pathlib import Path
from agent_runtime.evals.dataset import EvalDatasetLoader
from agent_runtime.evals.harness import EvalHarness
from agent_runtime.evals.failure_episode import FAILURE_MAX_STEPS_EXCEEDED
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool
from agent_runtime.models import AgentMessage, ToolCall
from agent_runtime.runtime.loop import AgentRuntime

def test_dataset_loader_and_schema_enforcement(tmp_path):
    """Verify that the dataset loader accepts valid rows and rejects malformed data."""
    good_file = tmp_path / "good.jsonl"
    good_file.write_text(
        '{"task_id": "t1", "name": "N", "user_input": "Q", "expected_outcome": {}, "trajectory_rules": [], "limits": {}}\n',
        encoding="utf-8"
    )
    tasks = EvalDatasetLoader.load_jsonl(str(good_file))
    assert len(tasks) == 1
    assert tasks[0].task_id == "t1"

    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"task_id": "missing_other_required_fields"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Field required"):
        EvalDatasetLoader.load_jsonl(str(bad_file))

def test_harness_trial_isolation_and_failure_capture():
    """Verify trial isolation and detection of a non-terminating tool loop."""
    
    # Simulate an assistant that repeatedly emits the same tool call.
    loop_responses = [
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
    ]
    
    def mock_runtime_factory():
        # Create a clean container for every trial.
        reg = ToolRegistry()
        reg.register(AddNumbersTool())
        # Copy responses because pop() would otherwise leak state across trials.
        prov = FakeProvider(list(loop_responses))
        return AgentRuntime(provider=prov, tool_registry=reg)

    harness = EvalHarness(runtime_factory=mock_runtime_factory)
    
    from agent_runtime.evals.models import EvalTask
    mock_task = EvalTask(
        task_id="task_test_loop",
        name="Test loop boundary",
        user_input="Test",
        limits={"max_steps": 2} # Enforce a small step budget.
    )
    
    # Run two independent trials.
    results = harness.run_task_suite([mock_task], num_trials=2)
    assert len(results) == 2
    
    for trial in results:
        assert trial.status == "max_steps_exceeded"
        episode = harness.extract_failure_episode(trial)
        assert episode is not None
        assert episode.failure_type == FAILURE_MAX_STEPS_EXCEEDED
        assert "max_steps_exceeded" in episode.root_cause_hypothesis
