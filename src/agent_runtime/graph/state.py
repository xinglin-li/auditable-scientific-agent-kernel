# src/agent_runtime/graph/state.py
from typing import Any, Literal, Optional, List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel

from agent_runtime.graph.reducers import (
    reduce_append,
    reduce_unique_str_list,
    reduce_artifact_refs
)

class MacroAgentState(TypedDict, total=False):
    # -- Identity and static context fields (default override semantics) --
    thread_id: str
    run_id: str
    user_query: str
    parsed_request: dict
    context_bundle: dict
    selected_skills: List[str]
    
    # -- Strongly typed state for auditable scientific modeling --
    model_spec: Any                 # Strongly typed ModelSpec IR instance.
    execution_plan: Any             # Strongly typed ExecutionPlan IR instance.
    # Current active diagnostics use replacement semantics so a successful
    # re-validation can clear errors consumed by the repair node.
    diagnostics: List[dict]
    repair_history: Annotated[List[dict], reduce_append]  # Audit trail of bounded repair diffs.
    repair_attempts: int            # Current repair-attempt count.
    
    # -- Analysis plan and exclusive state-machine position (latest edit overrides) --
    analysis_plan: dict
    plan_status: Literal["not_started", "drafted", "approved", "executing", "done"]
    approval_status: Literal["not_required", "pending", "approved", "rejected", "edited"]
    
    # -- Deduplicated external-system pointers and references (metadata only, no large payloads) --
    active_job_ids: Annotated[List[str], reduce_unique_str_list]
    completed_job_results: Annotated[List[dict], reduce_artifact_refs]
    artifact_refs: Annotated[List[dict], reduce_artifact_refs]
    
    # -- Deterministic append-only audit history and error accumulation --
    trace_events: Annotated[List[dict], reduce_append]
    errors: Annotated[List[dict], reduce_append]
    messages: Annotated[List[Any], reduce_append]  # Preserve compatibility with standard LLM interactions.
    
    # -- Final user-facing response (override semantics) --
    final_answer: Optional[str]
    status: Literal[
        "created",
        "running",
        "waiting_for_approval",
        "waiting_for_job",
        "completed",
        "failed",
        "cancelled"
    ]
