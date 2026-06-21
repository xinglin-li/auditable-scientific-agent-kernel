# tests/test_eval_feedback_loop.py
import pytest
from agent_runtime.evals.models import TrialResult, GraderResult
from agent_runtime.evals.harness import EvalHarness
from agent_runtime.evals.failure_episode import FAILURE_MAX_STEPS_EXCEEDED
from agent_runtime.evals.optimizer import ReflexionOptimizer
from agent_runtime.evals.skills_tracker import SkillsRegressionGuard

def test_automated_reflexion_and_skills_regression_lifecycle():
    # 1. Simulate a historical trial that failed in a tool loop.
    broken_trial = TrialResult(
        trial_id="tr_failed_loop_01",
        task_id="task_heavy_forecasting",
        run_id="run_abc",
        status="max_steps_exceeded",
        step_count=5,
        duration_ms=450.0,
        trace_events=[
            {"event_type": "context_assembled", "step": 1},
            {"event_type": "model_requested", "step": 2},
            {"event_type": "tool_called", "step": 3, "payload": {"tool": "fit_arima"}},
            {"event_type": "model_requested", "step": 4},
            {"event_type": "tool_called", "step": 5, "payload": {"tool": "fit_arima"}} # Repeated loop event.
        ]
    )
    
    # 2. Condense the trial into a standard failure episode.
    episode = EvalHarness.extract_failure_episode(broken_trial)
    assert episode is not None
    assert episode.failure_type == FAILURE_MAX_STEPS_EXCEEDED
    
    # 3. Send the compact episode to the offline optimizer.
    optimizer = ReflexionOptimizer()
    patch = optimizer.diagnose_and_suggest(episode)
    
    # Verify that the taxonomy selected the expected defensive instruction.
    assert "LOOP PREVENTION" in patch.injected_instruction
    assert "Diagnostic report" in episode.root_cause_hypothesis
    
    # 4. Simulate candidate performance after injecting the patch.
    # Cover both the new task and established core-skill regression cases.
    updated_trials_matrix = [
        # The new task converges successfully.
        TrialResult(
            trial_id="tr_new_task_ok", task_id="task_heavy_forecasting", run_id="r_new", 
            status="completed", step_count=2, duration_ms=200.0, grader_results=[]
        ),
        # Established skill 1: rolling backtest passes.
        TrialResult(
            trial_id="tr_backtest_skill_ok", task_id="skill_rolling_backtest_task_01", run_id="r_s1", 
            status="completed", step_count=3, duration_ms=300.0, grader_results=[]
        ),
        # Established skill 2: seasonal diagnostics pass.
        TrialResult(
            trial_id="tr_seasonal_skill_ok", task_id="skill_seasonal_diagnostics_task_01", run_id="r_s2", 
            status="completed", step_count=2, duration_ms=150.0, grader_results=[]
        )
    ]
    
    # 5. Run the skill-retention regression gate.
    guard = SkillsRegressionGuard()
    guard_report = guard.verify_skills_integrity(updated_trials_matrix)
    
    # Confirm the candidate fixed the bug without regressing established skills.
    assert guard_report["integrity_passed"] is True
    assert guard_report["details"]["skill_rolling_backtest"]["actual_rate"] == 1.0
