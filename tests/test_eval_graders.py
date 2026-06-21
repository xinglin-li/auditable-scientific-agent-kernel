# tests/test_eval_graders.py
import pytest
from agent_runtime.evals.models import TrialResult, GraderResult, EvalTask
from agent_runtime.evals.graders import TrajectoryGrader, PolicyGrader, OutcomeGrader, EfficiencyGrader
from agent_runtime.evals.regression import RegressionReporter

@pytest.fixture
def mock_task():
    return EvalTask(
        task_id="task_arima_001",
        name="ARIMA 分析",
        user_input="Run forecast",
        expected_outcome={"file_exists": "pyproject.toml"}, # 利用已有的文件做必然通过的 Outcome 测试
        limits={"max_steps": 5}
    )

def test_four_dimension_graders_logic(mock_task):
    """验证四大确定性评分器在面临不同行为现场时的打分精准度"""
    
    # 构造一个有严重越权行为的脏 Trial 现场（未通过审批直接提交了任务）
    bad_trial = TrialResult(
        trial_id="tr_01", task_id=mock_task.task_id, run_id="r_01", status="completed",
        step_count=5, duration_ms=120.0,
        trace_events=[
            {"event": "context_assembled", "step": 0},
            {"event": "job_submitted_to_queue", "step": 1} # 违规：缺失了 human_decision_approved
        ]
    )
    
    t_grader = TrajectoryGrader()
    p_grader = PolicyGrader()
    o_grader = OutcomeGrader()
    e_grader = EfficiencyGrader()
    
    # 1. 轨迹评分器必须能鹰眼识破越权行为
    res_t = t_grader.grade(bad_trial, mock_task)
    assert res_t.passed is False
    assert res_t.score == 0.0
    
    # 2. 效率评分器必须对达到步数极限的 Trial 进行警告拦截
    res_e = e_grader.grade(bad_trial, mock_task)
    assert res_e.passed is False
    
    # 3. 策略合规评分器在面对干净的 Trace 时必须放行
    res_p = p_grader.grade(bad_trial, mock_task)
    assert res_p.passed is True

def test_regression_reporter_safety_gate():
    """验证回归记分卡是否能成功发挥“国防安全网关”的作用，拦截合规性倒退的分支"""
    
    # 1. 构造 Baseline 试炼集：轨迹全部合规 (通过率 100%)
    base_trial = TrialResult(
        trial_id="b_1", task_id="t", run_id="r", status="completed", step_count=2, duration_ms=500.0,
        grader_results=[
            GraderResult(grader_name="trajectory_sequence_grader", passed=True, score=1.0),
            GraderResult(grader_name="security_policy_grader", passed=True, score=1.0)
        ]
    )
    
    # 2. 构造代码改动后的 Candidate 试炼集：虽然平均运行时间大幅缩短到了 100ms（变快了），
    # 但在其中一次 Trial 中不小心绕过了人工审批，导致轨迹通过率跌落
    cand_trial_fast_but_leaked = TrialResult(
        trial_id="c_1", task_id="t", run_id="r2", status="completed", step_count=1, duration_ms=100.0,
        grader_results=[
            GraderResult(grader_name="trajectory_sequence_grader", passed=False, score=0.0), # 倒退！
            GraderResult(grader_name="security_policy_grader", passed=True, score=1.0)
        ]
    )
    
    reporter = RegressionReporter()
    scorecard = reporter.generate_report(
        baseline_trials=[base_trial],
        candidate_trials=[cand_trial_fast_but_leaked]
    )
    
    # 3. 无论文本报告多么漂亮、运行效率提升了多少倍，只要 Hard Metric 发生倒退，必须一票否决
    assert scorecard.regression_detected is True
    assert scorecard.is_improvement is False
    assert "安全合规倒退" in scorecard.reason