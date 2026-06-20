# durable-agent-orchestration-lab

A test-driven macro research agent that combines LangGraph durable execution, human-in-the-loop approval, asynchronous jobs, crash recovery, MCP tool bridging, and deterministic security controls.

This repository builds on [context-rag-memory-skills](https://github.com/xinglin-li/context-rag-memory-skills). The earlier project established the context, retrieval, memory, skills, tool-validation, and structured ReAct layers. This iteration adds a durable orchestration layer for long-running macro research workflows.

The core premise remains unchanged: the model may propose work, but deterministic runtime code owns state transitions, approvals, side effects, persistence, retries, and security boundaries.

## Project Status

This is an educational runtime lab and prototype, not a production agent platform.

It is designed to make durable agent mechanics visible and testable:

- LangGraph state machine with explicit conditional routing
- SQLite-backed LangGraph checkpoints keyed by `thread_id`
- human approval through `interrupt()` and `Command(resume=...)`
- approve, reject, and edit/replan control paths
- asynchronous job submission with idempotency keys
- non-blocking job polling across graph invocations
- worker retries, terminal failures, and cooperative cancellation
- crash/restart recovery from persisted checkpoints
- append-only audit events and deduplicating reducers
- artifact references in graph state instead of large result payloads
- framework-free MCP stdio client over JSON-RPC 2.0
- dynamic MCP-to-native-tool bridge
- governed capability registry, RBAC, schema checks, and HITL policy routing
- inherited context, RAG, memory, skills, tools, tracing, and FastAPI runtime layers

Requires Python 3.11+.

## What Changed From The Previous Project

[context-rag-memory-skills](https://github.com/xinglin-li/context-rag-memory-skills) focused on preparing reliable model context and controlling a bounded ReAct loop:

- context assembly, pruning, and token-budget reporting
- BM25 and vector hybrid retrieval
- citation construction and retrieved-content trust boundaries
- working, episodic, semantic, and long-term memory
- progressive skill discovery and activation
- typed tool calls, validation, traces, and bounded execution

This project keeps those modules and adds a durable execution plane:

- **LangGraph Orchestration** — a typed `MacroAgentState` and explicit node/edge routing
- **Durable Checkpoints** — SQLite checkpoint history scoped by `thread_id`
- **Human-in-the-Loop Control** — plans can be approved, rejected, or edited before execution
- **Dynamic Replanning** — edited plans return to the approval boundary for a second decision
- **Asynchronous Job Lifecycle** — queued, running, succeeded, failed, cancel-requested, and cancelled states
- **Idempotent Submission** — repeated graph execution returns the existing job for the same idempotency key
- **Pause Instead of Busy Polling** — pending jobs end the current invocation and remain checkpointed
- **Crash Recovery** — a new service instance can reload graph state and observe work completed while it was offline
- **Artifact Indirection** — checkpoints retain `artifact_id` and URI metadata, not large CSV payloads
- **MCP Stdio Client** — JSON-RPC `tools/list` and `tools/call` over subprocess pipes
- **MCP Tool Bridge** — remote MCP metadata is wrapped as a local asynchronous callable
- **Capability Governance** — registration, risk labels, role checks, argument-schema checks, token checks, and origin allowlists
- **Auditable Reducers** — append-only traces plus deduplicated job and artifact collections

## Architecture

```text
==================================================================================================
                         DURABLE MACRO RESEARCH AGENT
==================================================================================================

  [ User Request ]
         |
         v
  +-------------------------+
  | Parse Request           |
  | intent / target / trace |
  +------------+------------+
               |
               v
  +-------------------------+       SQLite checkpoint history
  | Assemble Context & Plan |<--------------------------------+
  +------------+------------+                                 |
               |                                              |
               v                                              |
  +-------------------------+                                 |
  | Conditional Router      |                                 |
  | approval / submit /     |                                 |
  | poll / finalize / stop  |                                 |
  +---+-----------------+---+                                 |
      |                 |                                     |
      | approval needed | execution ready                     |
      v                 v                                     |
  +----------------+  +-------------------------+             |
  | Human Review   |  | MCP Policy Gateway      |             |
  | interrupt()    |  | registry / RBAC /       |             |
  | approve/edit/  |  | schema / risk controls  |             |
  | reject         |  +------------+------------+             |
  +-------+--------+               |                          |
          |                        v                          |
          |             +-------------------------+           |
          |             | Idempotent Job Store    |           |
          |             | JobRecord + status      |           |
          |             +------------+------------+           |
          |                          |                        |
          |              pause while queued/running           |
          |                          |                        |
          |                          v                        |
          |             +-------------------------+           |
          |             | Async Worker            |           |
          |             | retry / cancellation /  |           |
          |             | artifact production     |           |
          |             +------------+------------+           |
          |                          |                        |
          +--------------------------+------------------------+
                                     |
                                     v
                         +-------------------------+
                         | Poll Restored Job State |
                         | persist artifact refs   |
                         +------------+------------+
                                      |
                                      v
                         +-------------------------+
                         | Final Report            |
                         | answer + evidence URI   |
                         +-------------------------+

  External MCP path:

  MCP Server subprocess <-> AsyncMcpStdioClient <-> McpToolBridge <-> governed runtime capability
```

## Repository Layout

```text
durable-agent-orchestration-lab/
  pyproject.toml
  README.md
  LICENSE
  skills/
    rolling-backtest/
      SKILL.md
      references/checklist.md
    seasonal-diagnostics/
      SKILL.md
  sample_knowledge/
    time_series/
      arima_basics.md
      backtesting.md
  sample_data/
    series/monthly_sales.csv
  scripts/
    safe_describe_csv.py
    safe_series_summary.py
  src/
    agent_runtime/
      api.py
      config.py
      errors.py
      models.py
      tracing.py
      graph/
        builder.py
        edges.py
        nodes.py
        persistence.py
        reducers.py
        service.py
        state.py
      jobs/
        models.py
        store.py
        worker.py
      mcp/
        bridge.py
        client.py
      mcp_clients/
        models.py
        policy.py
        registry.py
      context/
      memory/
      providers/
      retrieval/
      runtime/
      skills/
      tools/
  tests/
```

## Quick Start

Create and activate a virtual environment, then install the project in editable mode:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest pytest-asyncio
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e .
python -m pip install pytest pytest-asyncio
```

Run the full test suite:

```bash
pytest -q
```

Run the durable workflow and MCP-focused tests:

```bash
pytest tests/test_durable_macro_agent.py tests/test_graph_approval.py tests/test_graph_persistence.py -v
pytest tests/test_job_worker.py tests/test_job_cancellation.py -v
pytest tests/test_mcp_client_bridge.py tests/test_mcp_capability_registry.py tests/test_mcp_approval_policy.py -v
```

Run the inherited FastAPI service:

```bash
uvicorn agent_runtime.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Durable Workflow Example

The end-to-end test demonstrates the intended lifecycle:

```python
import asyncio

from agent_runtime.graph.nodes import job_store
from agent_runtime.graph.service import MacroAgentGraphService
from agent_runtime.jobs.worker import AsyncMacroJobWorker


async def main() -> None:
    db_url = "file:macro_demo?mode=memory&cache=shared"
    thread_id = "macro-thread-001"

    service = MacroAgentGraphService(db_path=db_url)
    worker = AsyncMacroJobWorker(job_store)
    await worker.start()

    try:
        # The graph pauses at human approval.
        service.run_workflow(
            thread_id=thread_id,
            run_id="run-001",
            user_query="Run full rolling ARIMA for sales",
        )

        # Editing causes replanning and a second approval pause.
        service.resume_workflow(
            thread_id=thread_id,
            review_action={
                "action": "edited",
                "updated_plan": {
                    "task": "mcp_arima_forecast",
                    "horizon": 3,
                },
            },
        )

        # Approval submits the asynchronous job and checkpoints waiting state.
        waiting_state = service.resume_workflow(
            thread_id=thread_id,
            review_action={"action": "approved"},
        )
        print(waiting_state["status"])

        await asyncio.sleep(0.08)

        # A later invocation reloads the checkpoint, observes completion,
        # and finalizes with an artifact reference.
        final_state = service.run_workflow(
            thread_id=thread_id,
            run_id="run-002",
            user_query="Resume",
        )
        print(final_state["final_answer"])
    finally:
        await worker.stop()
        service.close()


asyncio.run(main())
```

## Core Concepts

### 1. Typed Durable State

`MacroAgentState` separates state by responsibility:

| State group | Representative fields | Merge behavior |
| --- | --- | --- |
| Identity and request | `thread_id`, `run_id`, `user_query`, `parsed_request` | latest value |
| Planning and control | `analysis_plan`, `plan_status`, `approval_status`, `status` | latest value |
| External work | `active_job_ids`, `completed_job_results`, `artifact_refs` | deduplicating reducers |
| Audit data | `trace_events`, `errors`, `messages` | append-only reducer |
| Delivery | `final_answer` | latest value |

Large analytical outputs do not belong in graph state. The graph stores references such as:

```python
{
    "artifact_id": "art_job_async_123",
    "uri": "storage://macro/processed_job_async_123.csv",
}
```

This keeps checkpoints small enough to inspect, replay, and persist.

### 2. Explicit Graph Routing

The graph is assembled from deterministic nodes:

```text
START
  -> parse_request
  -> assemble_context
  -> human_approval | submit_mcp_job | poll_job_status | finalize | END
```

The router evaluates, in order:

1. hard step limits
2. terminal failure
3. direct-answer intent
4. approval state
5. active-versus-completed job set difference
6. final delivery readiness

Pending jobs return `pause` and end the current invocation. This prevents a graph call from busy-polling while long-running work is still executing.

### 3. Human-in-the-Loop Approval

`human_approval_node` calls LangGraph `interrupt()` with the proposed plan. A caller resumes the same thread with:

```python
Command(resume={"action": "approved"})
```

Supported decisions:

- `approved` — continue toward guarded execution
- `rejected` — skip tool execution and finalize as failed
- `edited` — replace the plan and return to the approval boundary

The edit path intentionally requires a second decision after replanning.

### 4. Checkpointing and Recovery

`CheckpointPersistenceManager` owns the lifecycle of LangGraph's SQLite checkpointer. Every workflow uses a `thread_id` as its durable identity.

Checkpoint history exposes:

- checkpoint IDs
- state values at each transition
- the next scheduled nodes
- an auditable sequence of plan, approval, job, and finalization events

A replacement `MacroAgentGraphService` can connect to the same checkpoint database and continue the thread. The graph then polls the existing job instead of blindly submitting the side effect again.

### 5. Idempotent Asynchronous Jobs

`JobRecord` models the worker lifecycle:

```text
queued -> running -> succeeded
                  -> queued -> ... -> failed
                  -> cancel_requested -> cancelled
```

Each submission carries an idempotency key derived from the graph thread and step. If the same key is submitted again, the job store returns the existing `JobRecord`.

The worker:

- claims queued work
- increments attempt count
- runs asynchronous macro computation
- requeues transient failures while retry budget remains
- records terminal failures after retry exhaustion
- checks for cancellation between compute-intensive phases
- returns an artifact reference on success

### 6. MCP Client and Tool Bridge

`AsyncMcpStdioClient` communicates with an MCP server subprocess using newline-delimited JSON-RPC 2.0 frames.

Implemented primitives:

- `tools/list`
- `tools/call`

The client maps request IDs to pending futures, resolves responses in a background read loop, propagates reader failures, and terminates the subprocess during shutdown.

`McpToolBridge` converts discovered MCP tool metadata into a native async callable while retaining the remote tool name, description, and input schema.

### 7. Capability Governance

External tools are registered as `RegisteredCapability` records with:

- server and capability identity
- capability kind
- input schema
- risk level
- approval requirement
- allowed roles
- transport type

`McpApprovalPolicyGate` evaluates tool calls deterministically:

1. reject unregistered capabilities
2. enforce role-based access control
3. reject arguments outside the declared schema
4. route high-risk operations to HITL approval
5. allow only requests that pass every check

The registry also demonstrates bearer-token validation and origin allowlisting for a remote HTTP transport boundary.

### 8. Reducers and Auditability

Reducers encode merge semantics instead of relying on incidental list behavior:

- `reduce_append` preserves the complete event and error history
- `reduce_unique_str_list` deduplicates active job IDs while preserving order
- `reduce_artifact_refs` upserts structured records by `artifact_id` or `job_id`

This makes state evolution inspectable and protects artifact metadata from duplicate node execution.

### 9. Inherited Context, Retrieval, Memory, and Skills

The previous project remains available under the same runtime package:

- context deduplication, pruning, prioritization, and budget reports
- BM25 and deterministic vector retrieval with reciprocal-rank fusion
- citation maps and retrieved-content trust levels
- working, episodic, semantic, and long-term memory models
- SQLite memory with namespace isolation
- progressive skill discovery and activation
- typed tools, allowlists, retries, async execution, and tracing
- structured ReAct trajectory models

These modules form the context and reasoning plane; the new graph, jobs, and MCP packages form the durable orchestration plane.

## Test Matrix

| Area | Coverage |
| --- | --- |
| Graph routing | direct answer, job loop, set-difference progress, hard step limit |
| Human approval | approve, reject, edit/replan, second approval, side-effect guard |
| Persistence | checkpoint history, restart recovery, thread isolation |
| Reducers | append-only audit events, unique job IDs, artifact upsert |
| Job lifecycle | idempotent submission, retry exhaustion, cancellation |
| Durable integration | edit → approve → async job → pause → restart → recover |
| MCP stdio | subprocess startup, `tools/list`, `tools/call`, shutdown |
| MCP bridge | remote metadata converted into local async callables |
| Capability registry | governed registration, bearer token, origin allowlist |
| Approval policy | unknown tools, RBAC, schema injection, high-risk HITL routing |
| Runtime loop | direct answer, tool calls, retries, max steps, structured trajectory |
| Retrieval | tokenizer, chunking, BM25, vector index, RRF, citations |
| Context | deduplication, pruning, ordering, budget, lost-in-the-middle warning |
| Memory | working state, condensation, policy, namespace isolation, SQLite CRUD |
| Skills | metadata discovery, activation, fuzzy selection, progressive disclosure |
| API | health check, run submission, polling, trace fetch, 404 boundaries |

Run one test file:

```bash
pytest tests/test_graph_approval.py -v
```

Run one test:

```bash
pytest tests/test_durable_macro_agent.py::test_durable_macro_research_agent_grand_lifecycle_and_crash_recovery -v
```

## Security Notice

This project demonstrates application-level controls:

- explicit human approval boundaries
- capability registration and risk labels
- role-based authorization
- argument-schema validation
- bearer-token and origin checks
- idempotency keys around side effects
- approved subprocess commands
- context trust levels and memory-write filtering
- bounded graph routing and structured audit events

It is not a secure sandbox or a production authorization system.

Do not run untrusted MCP servers or scripts through this project. Do not expose the FastAPI service directly to the public internet. Subprocess separation is not an isolation boundary. Production execution of untrusted code requires operating-system controls such as containers, separate users, filesystem restrictions, CPU and memory quotas, syscall filtering, and network egress policy.

## Known Limitations

- The macro computation is simulated; it does not run a production ARIMA pipeline.
- The current job-store implementation is process-local and in-memory despite its prototype class name.
- Checkpoint durability uses SQLite, but production deployment still needs connection lifecycle and concurrency hardening.
- The demo recovery path models service replacement while the worker and job store remain available; it is not a full machine-power-loss simulation.
- MCP support covers the stdio primitives needed by the tests, not the complete protocol surface.
- The HTTP transport checks are policy demonstrations, not a complete MCP HTTP server implementation.
- The graph uses module-level infrastructure objects to simulate singleton dependency injection.
- Artifact URIs are metadata examples; no production object store is connected.
- Context token estimation and deterministic fake embeddings remain test-oriented.
- Provider integration still uses a fake provider rather than a production model adapter.
- Security controls are application-level demonstrations, not formal isolation or a security proof.

## Future Work

Potential next steps:

- replace the in-memory job store with transactional SQLite or PostgreSQL storage
- add leases, heartbeats, and compare-and-swap job transitions for multiple workers
- separate graph, worker, and MCP server into independently restartable processes
- persist artifact metadata in an object-store-backed repository
- add exponential-backoff timing and dead-letter handling
- implement durable cancellation and worker recovery after process loss
- support MCP Streamable HTTP in addition to stdio
- add richer JSON Schema validation and capability versioning
- connect real macro data sources and forecasting models
- integrate retrieved evidence and selected skills directly into graph planning
- add production provider adapters and streaming progress events
- add OpenTelemetry traces and operational metrics
- build golden-dataset evaluations for recovery, policy, and macro-analysis quality

## Design Summary

The model is probabilistic. Durable execution is deterministic.

This project extends the previous context-and-memory runtime with explicit orchestration boundaries: typed graph state, checkpointed transitions, human decisions, idempotent jobs, non-blocking waits, worker retries, cancellation, artifact references, MCP process communication, and policy-governed capabilities.

The engineering rule is simple: let the model propose research work, but let deterministic runtime code decide when it may run, how side effects are deduplicated, what survives a restart, and what evidence reaches the final report.

## License

MIT License. See [LICENSE](LICENSE).
