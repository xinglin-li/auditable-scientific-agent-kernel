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
    """测试数据集解析引擎是否能精准识别脏数据与标准加载"""
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
    """验证 Harness 在执行任务时是否做到了状态绝对隔离，并能精准捕获死循环故障"""
    
    # 模拟一个会故意引发死循环（不断抛出 tool_call）的有害 Assistant 响应
    loop_responses = [
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
    ]
    
    def mock_runtime_factory():
        # 每次实例化全新的干净容器
        reg = ToolRegistry()
        reg.register(AddNumbersTool())
        # 注意：每次都要重新复制一份，否则 pop() 会导致多 trial 共享状态污染
        prov = FakeProvider(list(loop_responses))
        return AgentRuntime(provider=prov, tool_registry=reg)

    harness = EvalHarness(runtime_factory=mock_runtime_factory)
    
    from agent_runtime.evals.models import EvalTask
    mock_task = EvalTask(
        task_id="task_test_loop",
        name="测试死循环边界",
        user_input="测试",
        limits={"max_steps": 2} # 强行限流
    )
    
    # 跑 2 次独立试炼
    results = harness.run_task_suite([mock_task], num_trials=2)
    assert len(results) == 2
    
    for trial in results:
        assert trial.status == "max_steps_exceeded"
        episode = harness.extract_failure_episode(trial)
        assert episode is not None
        assert episode.failure_type == FAILURE_MAX_STEPS_EXCEEDED
        assert "max_steps_exceeded" in episode.root_cause_hypothesis