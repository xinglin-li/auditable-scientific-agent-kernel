# tests/test_eval_graders.py
import pytest
from agent_runtime.evals.models import TrialResult, GraderResult, EvalTask
from agent_runtime.evals.graders import TrajectoryGrader, PolicyGrader, OutcomeGrader, EfficiencyGrader
from agent_runtime.evals.regression import RegressionReporter

@pytest.fixture
def mock_task():
    return EvalTask(
        task_id="task_arima_001",
        name="ARIMA analysis",
        user_input="Run forecast",
        expected_outcome={"file_exists": "pyproject.toml"}, # Use an existing file for a deterministic passing outcome.
        limits={"max_steps": 5}
    )

def test_four_dimension_graders_logic(mock_task):
    """Verify deterministic graders against representative behavior traces."""
    
    # Build a trial that submits a job without approval.
    bad_trial = TrialResult(
        trial_id="tr_01", task_id=mock_task.task_id, run_id="r_01", status="completed",
        step_count=5, duration_ms=120.0,
        trace_events=[
            {"event": "context_assembled", "step": 0},
            {"event": "job_submitted_to_queue", "step": 1} # Violation: human_decision_approved is missing.
        ]
    )
    
    t_grader = TrajectoryGrader()
    p_grader = PolicyGrader()
    o_grader = OutcomeGrader()
    e_grader = EfficiencyGrader()
    
    # 1. The trajectory grader must detect unauthorized execution.
    res_t = t_grader.grade(bad_trial, mock_task)
    assert res_t.passed is False
    assert res_t.score == 0.0
    
    # 2. The efficiency grader must reject a trial at its step limit.
    res_e = e_grader.grade(bad_trial, mock_task)
    assert res_e.passed is False
    
    # 3. The policy grader must accept a clean trace.
    res_p = p_grader.grade(bad_trial, mock_task)
    assert res_p.passed is True

def test_regression_reporter_safety_gate():
    """Verify that the regression scorecard blocks a policy regression."""
    
    # 1. Build a fully compliant baseline suite.
    base_trial = TrialResult(
        trial_id="b_1", task_id="t", run_id="r", status="completed", step_count=2, duration_ms=500.0,
        grader_results=[
            GraderResult(grader_name="trajectory_sequence_grader", passed=True, score=1.0),
            GraderResult(grader_name="security_policy_grader", passed=True, score=1.0)
        ]
    )
    
    # 2. Build a faster candidate suite with one trial that bypasses approval.
    cand_trial_fast_but_leaked = TrialResult(
        trial_id="c_1", task_id="t", run_id="r2", status="completed", step_count=1, duration_ms=100.0,
        grader_results=[
            GraderResult(grader_name="trajectory_sequence_grader", passed=False, score=0.0), # Regression.
            GraderResult(grader_name="security_policy_grader", passed=True, score=1.0)
        ]
    )
    
    reporter = RegressionReporter()
    scorecard = reporter.generate_report(
        baseline_trials=[base_trial],
        candidate_trials=[cand_trial_fast_but_leaked]
    )
    
    # 3. Any hard-metric regression must reject the candidate despite efficiency gains.
    assert scorecard.regression_detected is True
    assert scorecard.is_improvement is False
    assert "Safety compliance regression" in scorecard.reason
