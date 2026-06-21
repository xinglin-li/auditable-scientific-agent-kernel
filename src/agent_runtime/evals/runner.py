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
    工业级可变容量信号量（消除引用切换隐患）
    利用 asyncio.Condition 保证在高并发窗口期动态扩缩容的绝对原子性与协程安全
    """
    def __init__(self, initial_value: int):
        self.value = max(1, initial_value)
        self.allocated = 0                  # 当前已被占用的槽位数
        self._cond = asyncio.Condition()

    async def __aenter__(self):
        async with self._cond:
            # 如果当前分配的槽位已经达到或超过了动态调整后的上限，协程进入挂起等待队列
            while self.allocated >= self.value:
                await self._cond.wait()
            self.allocated += 1
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._cond:
            self.allocated -= 1
            # 无论是正常退出还是异常崩溃，都必须原子性释放槽位并唤醒等待队列
            self._cond.notify_all()

    async def resize(self, new_value: int):
        """
        动态调整并发上限（核心修复点）
        由于不更动实例引用，降低容量会导致新请求自然挂起，直到老请求退出使 allocated 降到新上限以下
        """
        async with self._cond:
            old_value = self.value
            self.value = max(1, new_value)
            logger.info(f"[信号量动态扩缩容] 并发窗口由 {old_value} 调整为 {self.value} | 当前在载槽位: {self.allocated}")
            # 扩容时需要唤醒挂起的协程；缩容时虽然新协程继续等待，但也进行安全通知
            self._cond.notify_all()


class AsyncEvalRunner:
    """具备自适应并发窗口控制与 SSE 实时流式导出的工业级评测引擎"""
    
    def __init__(self, runtime_factory, initial_concurrency: int = 4):
        self.runtime_factory = runtime_factory
        self.current_concurrency = initial_concurrency
        
        # 修复点：使用自定义的、不改变引用的动态信号量
        self.semaphore = DynamicSemaphore(initial_concurrency)
        self.graders = [TrajectoryGrader(), PolicyGrader(), OutcomeGrader(), EfficiencyGrader()]

    async def run_single_trial_async(self, task: EvalTask, trial_idx: int) -> TrialResult:
        """在并发调度内安全、隔离地执行单次独立试炼，具备 429 自适应退避能力"""
        # 此时所有协程始终通过同一个锁屏障进行管控
        async with self.semaphore:
            trial_id = f"trial_{task.task_id}_{int(time.time())}_{trial_idx}"
            runtime = self.runtime_factory()
            if "max_steps" in task.limits:
                runtime.max_steps = task.limits["max_steps"]
                
            start_time = time.perf_counter()
            
            try:
                state = await asyncio.to_thread(runtime.run, task.user_input)
                
                # 弹性自适应拦截：如果检测到因 429 频控导致的失败，执行原子性缩容
                if state.status == "failed" and "429" in (state.final_answer or ""):
                    if self.current_concurrency > 1:
                        self.current_concurrency -= 1
                        # 修复点：await 安全调整内部配额，不破坏上下文管理器状态
                        await self.semaphore.resize(self.current_concurrency)
                        logger.warning(f"触发大模型429频控！动态自适应收缩并发信号量上限至: {self.current_concurrency}")
                
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
        """将并发调度的中间状态与评测指标流式转化为标准 SSE 协议数据流"""
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