# src/agent_runtime/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uuid
import asyncio

from agent_runtime.runtime.state import AgentState
from agent_runtime.runtime.loop import AgentRuntime
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.models import AgentMessage

# Strongly typed evaluation components added for Day 4.
from agent_runtime.evals.models import EvalTask
from agent_runtime.evals.runner import AsyncEvalRunner

app = FastAPI(title="AI Agent Runtime Service", version="1.0.0")

# Preserve the existing in-memory run store.
RUNS_DATABASE: Dict[str, AgentState] = {}
ACTIVE_TASKS: Dict[str, asyncio.Task] = {}

def get_configured_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(AddNumbersTool())
    return reg

class CreateRunRequest(BaseModel):
    user_input: str = Field(..., description="The query prompt for the agent to execute.")
    max_steps: Optional[int] = Field(5, description="Upper bound of agent loops.")

class RunSummaryResponse(BaseModel):
    run_id: str
    status: str
    step_count: int
    final_answer: Optional[str] = None

# ==================== REST routes (backward compatible) ====================

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "agent-runtime-core"}

async def background_agent_worker(run_id: str, user_input: str, max_steps: int):
    fake_responses = [AgentMessage(role="assistant", content="Processed via FastAPI Gateway seamlessly.")]
    provider = FakeProvider(fake_responses)
    runtime = AgentRuntime(provider=provider, tool_registry=get_configured_registry(), max_steps=max_steps)
    try:
        await asyncio.sleep(0.01)
        final_state = runtime.run(user_input)
        final_state.run_id = run_id
        RUNS_DATABASE[run_id] = final_state
    except asyncio.CancelledError:
        if run_id in RUNS_DATABASE:
            RUNS_DATABASE[run_id].status = "cancelled"
    except Exception:
        if run_id in RUNS_DATABASE:
            RUNS_DATABASE[run_id].status = "failed"

@app.post("/runs", response_model=RunSummaryResponse, status_code=status.HTTP_201_CREATED)
async def create_run(payload: CreateRunRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    initial_state = AgentState(run_id=run_id, messages=[], step_count=0, status="running")
    RUNS_DATABASE[run_id] = initial_state
    task = asyncio.create_task(background_agent_worker(run_id, payload.user_input, payload.max_steps))
    ACTIVE_TASKS[run_id] = task
    return RunSummaryResponse(run_id=run_id, status="running", step_count=0)

@app.get("/runs/{run_id}", response_model=RunSummaryResponse)
def get_run_status(run_id: str):
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail=f"Run database records for ID '{run_id}' not found.")
    state = RUNS_DATABASE[run_id]
    return RunSummaryResponse(run_id=state.run_id, status=state.status, step_count=state.step_count, final_answer=state.final_answer)

# ==================== Asynchronous streaming evaluation route ====================

class EvalSuiteExecutionRequest(BaseModel):
    """Request contract for executing a batch evaluation suite."""
    tasks: List[EvalTask] = Field(..., min_length=1, description="Golden evaluation tasks")
    num_trials: int = Field(2, ge=1, description="Number of repetitions per task")
    initial_concurrency: int = Field(3, ge=1, description="Initial concurrency limit")

def api_runtime_factory_wrapper():
    """Create a clean, isolated runtime for each evaluation trial."""
    # Simulate a generic evaluation response.
    responses = [AgentMessage(role="assistant", content="ARIMA model specs calculated. Passed.")]
    return AgentRuntime(provider=FakeProvider(responses), tool_registry=get_configured_registry())

@app.post("/api/v1/evals/run")
async def run_eval_suite_stream(payload: EvalSuiteExecutionRequest) -> StreamingResponse:
    """
    Run an evaluation dataset concurrently and stream audit events to the
    client using the standard SSE protocol.
    """
    runner = AsyncEvalRunner(
        runtime_factory=api_runtime_factory_wrapper,
        initial_concurrency=payload.initial_concurrency
    )
    
    # Use the SSE media type so clients can consume the response incrementally.
    return StreamingResponse(
        runner.stream_suite_evaluation(payload.tasks, num_trials=payload.num_trials),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
