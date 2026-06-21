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
    """Run isolated trials and collect multidimensional quality measurements."""
    
    def __init__(self, runtime_factory):
        """
        Accept a runtime factory that creates a clean AgentRuntime and memory
        store before every trial.
        """
        self.runtime_factory = runtime_factory

    def execute_trial(self, task: EvalTask, trial_idx: int) -> TrialResult:
        """Run one fully isolated trial."""
        trial_id = f"trial_{task.task_id}_{int(time.time())}_{trial_idx}"
        
        # 1. Create a clean runtime to prevent memory leakage across trials.
        runtime: AgentRuntime = self.runtime_factory()
        
        # 2. Apply the execution limits declared by the task.
        if "max_steps" in task.limits:
            runtime.max_steps = task.limits["max_steps"]

        start_time = time.perf_counter()
        
        try:
            # 3. Execute the deterministic control loop.
            state = runtime.run(task.user_input)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # 4. Convert trace events into the standard DTO representation.
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
            # Capture runtime crashes as failed trials without terminating the harness.
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
        """Execute a full dataset and apply all configured graders."""
        all_results = []
        
        for task in tasks:
            for i in range(num_trials):
                result = self.execute_trial(task, i)
                
                # ---- Deterministic outcome checks and automatic labeling ----
                grader_results = []
                
                # Rule 1: verify execution status.
                status_passed = result.status == "completed"
                grader_results.append(GraderResult(
                    grader_name="status_completed_grader",
                    passed=status_passed,
                    score=1.0 if status_passed else 0.0,
                    reason=f"Agent status: {result.status}"
                ))
                
                # Rule 2: verify external outcomes, such as expected files.
                if "file_exists" in task.expected_outcome:
                    from pathlib import Path
                    target_file = Path(task.expected_outcome["file_exists"])
                    file_ok = target_file.exists()
                    grader_results.append(GraderResult(
                        grader_name="file_outcome_grader",
                        passed=file_ok,
                        score=1.0 if file_ok else 0.0,
                        reason=f"Target file {target_file} exists: {file_ok}"
                    ))
                
                result.grader_results = grader_results
                all_results.append(result)
                
        return all_results

    @staticmethod
    def extract_failure_episode(trial: TrialResult) -> Optional[FailureEpisode]:
        """Condense a failed trial into a FailureEpisode."""
        # Determine whether every hard grader passed.
        is_success = all(g.passed for g in trial.grader_results) if trial.grader_results else trial.status == "completed"
        if is_success:
            return None
            
        # Map trace events and status to the deterministic failure taxonomy.
        ftype = FAILURE_UNKNOWN
        err_msg = trial.final_answer or "Unknown error"
        
        if trial.status == "max_steps_exceeded":
            ftype = FAILURE_MAX_STEPS_EXCEEDED
        else:
            # Scan the trace to locate the failing node.
            for event in trial.trace_events:
                if event.get("event_type") == "tool_validation_failed":
                    ftype = FAILURE_TOOL_ARGUMENT
                    err_msg = f"Tool validation blocked the call: {event.get('payload', {})}"
                    break
        
        # Extract a compact event summary for Reflexion.
        summary_events = [f"[{e.get('step')}] {e.get('event_type') or e.get('event')}" for e in trial.trace_events]
        
        return FailureEpisode(
            task_id=trial.task_id,
            run_id=trial.run_id,
            trial_id=trial.trial_id,
            failure_type=ftype,
            error_message=err_msg,
            failed_step=trial.step_count,
            trace_summary=summary_events[:15], # Keep 15 steps to bound token usage.
            root_cause_hypothesis=f"The agent terminated with status {trial.status} at step {trial.step_count}."
        )
