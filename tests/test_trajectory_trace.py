# tests/test_trajectory_trace.py
"""V2 Structured ReAct trajectory trace tests."""

from agent_runtime.models import (
    AgentMessage,
    ToolCall,
    PlannerRationale,
    StopReason,
    AgentStep,
)
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.runtime.loop import AgentRuntime


def _registry():
    reg = ToolRegistry()
    reg.register(AddNumbersTool())
    return reg


def test_direct_answer_produces_agent_step_with_stop_reason():
    fake_responses = [AgentMessage(role="assistant", content="The answer is 42.")]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=_registry(),
    )
    state = runtime.run("What is the answer?")
    assert state.status == "completed"
    assert state.stop_reason == "final_answer"
    assert len(state.steps) == 1
    step = state.steps[0]
    assert isinstance(step, AgentStep)
    assert step.stop_reason is not None
    assert isinstance(step.stop_reason, StopReason)
    assert step.stop_reason.reason == "final_answer"


def test_tool_call_produces_agent_step_with_rationale_and_observation():
    fake_responses = [
        AgentMessage(
            role="assistant",
            rationale=PlannerRationale(
                summary="Need to add 2 and 3 per user request.",
                confidence=0.95,
            ),
            tool_calls=[ToolCall(call_id="call_001", tool_name="add_numbers", arguments={"a": 2, "b": 3})],
        ),
        AgentMessage(role="assistant", content="2 + 3 = 5."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=_registry(),
    )
    state = runtime.run("What is 2 + 3?")
    assert state.status == "completed"
    assert len(state.steps) == 2
    tool_step = state.steps[0]
    assert tool_step.rationale is not None
    assert tool_step.rationale.summary == "Need to add 2 and 3 per user request."
    assert tool_step.action is not None
    assert tool_step.action.call_id == "call_001"
    assert tool_step.observation is not None
    assert tool_step.observation.ok is True
    answer_step = state.steps[1]
    assert answer_step.stop_reason is not None
    assert answer_step.stop_reason.reason == "final_answer"


def test_rationale_recorded_before_tool_started():
    fake_responses = [
        AgentMessage(
            role="assistant",
            rationale=PlannerRationale(summary="Adding numbers."),
            tool_calls=[ToolCall(call_id="c1", tool_name="add_numbers", arguments={"a": 1, "b": 2})],
        ),
        AgentMessage(role="assistant", content="Done."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=_registry(),
    )
    state = runtime.run("Add 1 + 2")
    events = state.trace_events
    rationale_idx = next(i for i, e in enumerate(events) if e.event_type == "rationale_recorded")
    tool_started_idx = next(i for i, e in enumerate(events) if e.event_type == "tool_started")
    assert rationale_idx < tool_started_idx


def test_observation_recorded_after_tool_execution():
    fake_responses = [
        AgentMessage(
            role="assistant",
            tool_calls=[ToolCall(call_id="c1", tool_name="add_numbers", arguments={"a": 5, "b": 7})],
        ),
        AgentMessage(role="assistant", content="Result is 12."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=_registry(),
    )
    state = runtime.run("Add 5 + 7")
    events = state.trace_events
    tool_end_idx = next(i for i, e in enumerate(events) if e.event_type == "tool_succeeded")
    obs_idx = next(i for i, e in enumerate(events) if e.event_type == "observation_recorded")
    assert tool_end_idx < obs_idx


def test_stop_reason_recorded_before_run_completed():
    fake_responses = [AgentMessage(role="assistant", content="Direct answer.")]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=_registry(),
    )
    state = runtime.run("Hello")
    events = state.trace_events
    stop_idx = next(i for i, e in enumerate(events) if e.event_type == "stop_reason_recorded")
    completed_idx = next(i for i, e in enumerate(events) if e.event_type == "run_completed")
    assert stop_idx < completed_idx


def test_tool_action_created_after_tool_call_received():
    fake_responses = [
        AgentMessage(
            role="assistant",
            tool_calls=[ToolCall(call_id="c1", tool_name="add_numbers", arguments={"a": 1, "b": 1})],
        ),
        AgentMessage(role="assistant", content="Done."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=_registry(),
    )
    state = runtime.run("Add 1 + 1")
    events = state.trace_events
    received_idx = next(i for i, e in enumerate(events) if e.event_type == "tool_call_received")
    action_idx = next(i for i, e in enumerate(events) if e.event_type == "tool_action_created")
    assert received_idx < action_idx


def test_no_raw_cot_in_trace():
    fake_responses = [
        AgentMessage(
            role="assistant",
            rationale=PlannerRationale(
                summary="Simple addition requested.",
                confidence=0.9,
                cited_state_keys=["user_input"],
            ),
            tool_calls=[ToolCall(call_id="c1", tool_name="add_numbers", arguments={"a": 3, "b": 4})],
        ),
        AgentMessage(role="assistant", content="3 + 4 = 7."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=_registry(),
    )
    state = runtime.run("Add 3 + 4")
    tool_step = state.steps[0]
    assert isinstance(tool_step.rationale, PlannerRationale)
    assert tool_step.rationale.summary == "Simple addition requested."
    rationale_events = [e for e in state.trace_events if e.event_type == "rationale_recorded"]
    assert len(rationale_events) == 1
    assert "summary" in rationale_events[0].payload