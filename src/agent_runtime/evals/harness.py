# src/agent_runtime/evals/harness.py
import time
import uuid
from typing import List, Dict, Any, Optional
from agent_runtime.runtime.loop import AgentRuntime
from agent_runtime.evals.models import EvalTask, TrialResult, GraderResult
from agent_runtime.evals.failure_episode import (
    FailureEpisode, FAILURE_MAX_STEPS_EXCEEDED, FAILURE_TOOL_ARGUMENT, FAILURE_UNKNOWN
)

class EvalHarness:
    """隔离执行、多 Trial 驱动与多维度质量度量的核心发动机"""
    
    def __init__(self, runtime_factory):
        """
        传入一个 runtime_factory 闭包，用于在每个 Trial 开始前，
        重新实例化干净、无状态污染的 AgentRuntime 与内存仓。
        """
        self.runtime_factory = runtime_factory

    def execute_trial(self, task: EvalTask, trial_idx: int) -> TrialResult:
        """运行单次绝对隔离的 Trial"""
        trial_id = f"trial_{task.task_id}_{int(time.time())}_{trial_idx}"
        
        # 1. 动态拉起彻底隔离的干净 Runtime 实例，阻止跨 Trial 的记忆串扰
        runtime: AgentRuntime = self.runtime_factory()
        
        # 2. 注入 Task 声明的运行限制限制
        if "max_steps" in task.limits:
            runtime.max_steps = task.limits["max_steps"]

        start_time = time.perf_counter()
        
        try:
            # 3. 交付确定性控制环流转执行
            state = runtime.run(task.user_input)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # 4. 提取基线时序 Trace 实体并转化为标准 DTO 字典列表
            flat_traces = [e.model_dump() for e in state.trace_events]
            
            return TrialResult(
                trial_id=trial_id,
                task_id=task.task_id,
                run_id=state.run_id,
                status=state.status,
                final_answer=state.final_answer,
                trace_events=flat_traces,
                duration_results=[],
                duration_ms=round(duration_ms, 2),
                step_count=state.step_count
            )
            
        except Exception as e:
            # 即使发生运行时崩溃，也绝对不让 Harness 散架，而是优雅捕获并记录为 failed 状态
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TrialResult(
                trial_id=trial_id,
                task_id=task.task_id,
                run_id=f"failed_run_{uuid.uuid4().hex[:8]}",
                status="failed",
                final_answer=f"Fatal Execution Error: {str(e)}",
                trace_events=[{"event_type": "fatal_crash", "step": 0, "payload": {"msg": str(e)}}],
                duration_ms=round(duration_ms, 2),
                step_count=0
            )

    def run_task_suite(self, tasks: List[EvalTask], num_trials: int = 3) -> List[TrialResult]:
        """驱动完整的数据集，收集并进行多维打分打标"""
        all_results = []
        
        for task in tasks:
            for i in range(num_trials):
                result = self.execute_trial(task, i)
                
                # ---- Day 1 确定性 Outcome 核心检验与自动打标逻辑 ----
                grader_results = []
                
                # 规则 1：检查执行状态
                status_passed = result.status == "completed"
                grader_results.append(GraderResult(
                    grader_name="status_completed_grader",
                    passed=status_passed,
                    score=1.0 if status_passed else 0.0,
                    reason=f"Agent 状态为: {result.status}"
                ))
                
                # 规则 2：环境 Outcome 实体检验 (例如校验预期生成的数据文件是否存在)
                if "file_exists" in task.expected_outcome:
                    from pathlib import Path
                    target_file = Path(task.expected_outcome["file_exists"])
                    file_ok = target_file.exists()
                    grader_results.append(GraderResult(
                        grader_name="file_outcome_grader",
                        passed=file_ok,
                        score=1.0 if file_ok else 0.0,
                        reason=f"目标文件 {target_file} 存在状态: {file_ok}"
                    ))
                
                result.grader_results = grader_results
                all_results.append(result)
                
        return all_results

    @staticmethod
    def extract_failure_episode(trial: TrialResult) -> Optional[FailureEpisode]:
        """故障现场捕获器：若检测到 Trial 未通过，自动冷凝为 FailureEpisode"""
        # 判断是否通过了所有硬性评分
        is_success = all(g.passed for g in trial.grader_results) if trial.grader_results else trial.status == "completed"
        if is_success:
            return None
            
        # 根据 Trace 事件和状态进行确定性 Failure Taxonomy 分类映射
        ftype = FAILURE_UNKNOWN
        err_msg = trial.final_answer or "未知错误"
        
        if trial.status == "max_steps_exceeded":
            ftype = FAILURE_MAX_STEPS_EXCEEDED
        else:
            # 扫描时序 Trace，诊断定位具体的崩溃节点
            for event in trial.trace_events:
                if event.get("event_type") == "tool_validation_failed":
                    ftype = FAILURE_TOOL_ARGUMENT
                    err_msg = f"工具校验拦截: {event.get('payload', {})}"
                    break
        
        # 抽取最精简的时序摘要作为 Reflexion 的特征向量
        summary_events = [f"[{e.get('step')}] {e.get('event_type') or e.get('event')}" for e in trial.trace_events]
        
        return FailureEpisode(
            task_id=trial.task_id,
            run_id=trial.run_id,
            trial_id=trial.trial_id,
            failure_type=ftype,
            error_message=err_msg,
            failed_step=trial.step_count,
            trace_summary=summary_events[:15], # 截取前 15 步防止爆 Token
            root_cause_hypothesis=f"Agent 在状态为 {trial.status} 时终止，最后记录步数为 {trial.step_count}。"
        )
