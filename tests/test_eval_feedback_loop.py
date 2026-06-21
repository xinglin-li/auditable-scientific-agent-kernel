# tests/test_eval_feedback_loop.py
import pytest
from agent_runtime.evals.models import TrialResult, GraderResult
from agent_runtime.evals.harness import EvalHarness
from agent_runtime.evals.failure_episode import FAILURE_MAX_STEPS_EXCEEDED
from agent_runtime.evals.optimizer import ReflexionOptimizer
from agent_runtime.evals.skills_tracker import SkillsRegressionGuard

def test_automated_reflexion_and_skills_regression_lifecycle():
    # 1. 模拟一个因为死循环引发剧烈崩溃的老旧 Trial
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
            {"event_type": "tool_called", "step": 5, "payload": {"tool": "fit_arima"}} # 陷入死循环死穴
        ]
    )
    
    # 2. 触发 Day 1 的冷凝器，固化为标准失败片段
    episode = EvalHarness.extract_failure_episode(broken_trial)
    assert episode is not None
    assert episode.failure_type == FAILURE_MAX_STEPS_EXCEEDED
    
    # 3. 交付 Day 3 离线优化器，进行无污染高浓度根因诊断
    optimizer = ReflexionOptimizer()
    patch = optimizer.diagnose_and_suggest(episode)
    
    # 验证优化器是否根据分类字典产出了精准的防御性规约指令
    assert "防死循环策略" in patch.injected_instruction
    assert "诊断报告" in episode.root_cause_hypothesis
    
    # 4. 模拟注入补丁后，Candidate 智能体在全新一轮测试中的表现
    # 此时我们同时模拟两类测试用例：新任务通过了；老核心技能测试（skill_rolling_backtest）也包含在内
    updated_trials_matrix = [
        # 新任务顺利收敛并通过
        TrialResult(
            trial_id="tr_new_task_ok", task_id="task_heavy_forecasting", run_id="r_new", 
            status="completed", step_count=2, duration_ms=200.0, grader_results=[]
        ),
        # 核心老技能 1：滚动回测测试用例顺利通过
        TrialResult(
            trial_id="tr_backtest_skill_ok", task_id="skill_rolling_backtest_task_01", run_id="r_s1", 
            status="completed", step_count=3, duration_ms=300.0, grader_results=[]
        ),
        # 核心老技能 2：季节性诊断测试用例顺利通过
        TrialResult(
            trial_id="tr_seasonal_skill_ok", task_id="skill_seasonal_diagnostics_task_01", run_id="r_s2", 
            status="completed", step_count=2, duration_ms=150.0, grader_results=[]
        )
    ]
    
    # 5. 启动技能防遗忘网关进行终极审计
    guard = SkillsRegressionGuard()
    guard_report = guard.verify_skills_integrity(updated_trials_matrix)
    
    # 验证结果：新智能体既修了 Bug，又 100% 完美的保留了历史量化投研核心长板
    assert guard_report["integrity_passed"] is True
    assert guard_report["details"]["skill_rolling_backtest"]["actual_rate"] == 1.0