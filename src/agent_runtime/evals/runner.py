# src/agent_runtime/evals/runner.py
import asyncio
import json
import time
import logging
from typing import AsyncGenerator, List, Any, Dict
from agent_runtime.evals.models import EvalTask, TrialResult, GraderResult
from agent_runtime.evals.graders import TrajectoryGrader, PolicyGrader, OutcomeGrader, EfficiencyGrader

logger = logging.getLogger("eval-runner")

class DynamicSemaphore:
    """
    Variable-capacity semaphore that avoids replacing the synchronization
    object while requests are active. asyncio.Condition keeps resizing atomic.
    """
    def __init__(self, initial_value: int):
        self.value = max(1, initial_value)
        self.allocated = 0                  # Number of currently occupied slots.
        self._cond = asyncio.Condition()

    async def __aenter__(self):
        async with self._cond:
            # Wait while allocation is at or above the current capacity.
            while self.allocated >= self.value:
                await self._cond.wait()
            self.allocated += 1
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._cond:
            self.allocated -= 1
            # Release the slot atomically and wake waiters on every exit path.
            self._cond.notify_all()

    async def resize(self, new_value: int):
        """
        Resize the concurrency limit without replacing this instance. Lowering
        capacity naturally blocks new requests until allocation falls below it.
        """
        async with self._cond:
            old_value = self.value
            self.value = max(1, new_value)
            logger.info(f"[Semaphore resize] concurrency changed from {old_value} to {self.value} | allocated slots: {self.allocated}")
            # Notify waiters after either expansion or contraction.
            self._cond.notify_all()


class AsyncEvalRunner:
    """Evaluation runner with adaptive concurrency and SSE streaming."""
    
    def __init__(self, runtime_factory, initial_concurrency: int = 4):
        self.runtime_factory = runtime_factory
        self.current_concurrency = initial_concurrency
        
        # Keep one resizable semaphore instance for the runner lifetime.
        self.semaphore = DynamicSemaphore(initial_concurrency)
        self.graders = [TrajectoryGrader(), PolicyGrader(), OutcomeGrader(), EfficiencyGrader()]

    async def run_single_trial_async(self, task: EvalTask, trial_idx: int) -> TrialResult:
        """Run one isolated trial with adaptive backoff for HTTP 429 failures."""
        # Every coroutine passes through the same synchronization barrier.
        async with self.semaphore:
            trial_id = f"trial_{task.task_id}_{int(time.time())}_{trial_idx}"
            runtime = self.runtime_factory()
            if "max_steps" in task.limits:
                runtime.max_steps = task.limits["max_steps"]
                
            start_time = time.perf_counter()
            
            try:
                state = await asyncio.to_thread(runtime.run, task.user_input)
                
                # Reduce capacity atomically after a rate-limit failure.
                if state.status == "failed" and "429" in (state.final_answer or ""):
                    if self.current_concurrency > 1:
                        self.current_concurrency -= 1
                        # Resize safely without invalidating the active context manager.
                        await self.semaphore.resize(self.current_concurrency)
                        logger.warning(f"Model API returned HTTP 429; reduced concurrency to {self.current_concurrency}")
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                flat_traces = [e.model_dump() for e in state.trace_events]
                
                trial_res = TrialResult(
                    trial_id=trial_id, task_id=task.task_id, run_id=state.run_id,
                    status=state.status, final_answer=state.final_answer,
                    trace_events=flat_traces, duration_ms=round(duration_ms, 2),
                    step_count=state.step_count
                )
                
                grader_results = []
                for grader in self.graders:
                    grader_results.append(grader.grade(trial_res, task))
                trial_res.grader_results = grader_results
                
                return trial_res
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return TrialResult(
                    trial_id=trial_id, task_id=task.task_id, run_id="failed_async",
                    status="failed", final_answer=str(e), duration_ms=round(duration_ms, 2), step_count=0
                )

    async def stream_suite_evaluation(self, tasks: List[EvalTask], num_trials: int = 2) -> AsyncGenerator[str, None]:
        """Stream concurrent evaluation progress as standard SSE events."""
        futures = []
        for task in tasks:
            for i in range(num_trials):
                futures.append(self.run_single_trial_async(task, i))
                
        for next_future in asyncio.as_completed(futures):
            trial_result = await next_future
            all_passed = all(g.passed for g in trial_result.grader_results) if trial_result.grader_results else False
            
            sse_payload = {
                "task_id": trial_result.task_id,
                "trial_id": trial_result.trial_id,
                "status": trial_result.status,
                "all_passed": all_passed,
                "step_count": trial_result.step_count,
                "duration_ms": trial_result.duration_ms,
                "final_answer": trial_result.final_answer
            }
            yield f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n"
            
        yield "data: [DONE]\n\n"
