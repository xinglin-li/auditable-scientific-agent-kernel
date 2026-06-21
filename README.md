# Auditable Scientific Agent Kernel

A local-first, test-driven kernel for scientific agents that converts natural-language research requests into validated model specifications, deterministic execution plans, auditable repairs, and evidence-linked reports.

This Week 4 project builds on [xinglin-li/durable-agent-orchestration-lab](https://github.com/xinglin-li/durable-agent-orchestration-lab). The inherited runtime, retrieval, memory, persistence, jobs, and MCP components remain in the repository, while the active graph now focuses on scientific validation and evaluation.

> Let the model propose a candidate. Let deterministic software validate, compile, execute, repair, and grade it.

## Current Status

Implemented:

- strongly typed `ModelSpec` intermediate representation
- ARIMA, ETS, and seasonal-naive model contracts
- schema parsing and domain validation
- static checks for future leakage and unsafe paths
- deterministic `ModelSpec -> ExecutionPlan` compilation
- allowlisted scientific operation executor
- structured numerical diagnostics
- bounded ARIMA and sample-window repair policies
- LangGraph scientific workflow with explicit terminal conditions
- isolated evaluation trials and JSONL evaluation datasets
- trajectory, policy, outcome, and efficiency graders
- baseline-versus-candidate regression scorecards
- compact `FailureEpisode` records and offline Reflexion suggestions
- core-skill regression protection
- concurrent evaluation runner with adaptive rate-limit backoff
- FastAPI run endpoints and SSE evaluation streaming

The project currently uses deterministic fake providers and simulated scientific execution. No real LLM API is connected yet.

## Architecture

```text
User request
    |
    v
generate_spec
    |
    v
validate_spec -----------------------------+
    |                                      |
    | valid                                | repairable error
    v                                      v
compile_and_execute <---------------- repair_spec
    |                                      |
    | success                              | repaired spec
    v                                      +----> validate_spec
finalize_report
    |
    v
Auditable result

Non-repairable policy violation ----------> failed report
```

The main boundary is `ModelSpec`. Natural-language output cannot execute directly. It must pass through typed parsing, domain validation, static analysis, deterministic compilation, and an approved handler registry.

## Scientific Pipeline

### 1. Candidate specification

`ModelSpecParser` converts structured model output into a Pydantic-validated candidate. Fields that affect authorization remain controlled by system policy.

Supported model contracts:

- `seasonal_naive`
- `ets`
- `arima`

### 2. Validation and static analysis

The validation layer checks:

- dataset availability
- declared versus actual frequency
- ETS seasonal configuration
- ARIMA order constraints and complexity
- sample size versus backtest window
- future-information leakage
- path-traversal indicators

Diagnostics are structured records with a stage, severity, error code, evidence, and repairability decision.

### 3. Deterministic compilation and execution

Validated specifications are compiled into dependency-ordered `ExecutionPlan` steps. The executor dispatches only operations registered in its allowlist, preventing arbitrary model-generated code execution.

### 4. Bounded repair

Repairable failures can enter a limited repair cycle. Current repair strategies include:

- reducing non-convergent ARIMA models to `ARIMA(1, d, 1)`
- contracting an oversized initial backtest window

Every repair produces a structured diff. Future leakage and other non-repairable policy violations terminate the workflow instead of being silently modified.

## Evaluation System

The evaluation layer keeps behavior, outcomes, and safety checks separate.

| Component | Responsibility |
| --- | --- |
| `EvalTask` | Versioned input, expected outcome, trajectory rules, and limits |
| `TrialResult` | Isolated execution result with trace and timing data |
| `EvalHarness` | Runs clean trials and captures failures without crashing the suite |
| `TrajectoryGrader` | Checks required event ordering and approval boundaries |
| `PolicyGrader` | Detects runtime safety-gate violations |
| `OutcomeGrader` | Verifies external state such as generated files |
| `EfficiencyGrader` | Enforces step budgets |
| `RegressionReporter` | Compares baseline and candidate hard metrics |
| `FailureEpisode` | Stores a compact toxic slice for offline diagnosis |
| `ReflexionOptimizer` | Produces candidate prompt-policy patches offline |
| `SkillsRegressionGuard` | Prevents fixes from regressing established skills |

Hard safety metrics cannot be hidden by improvements in speed or prose quality. A candidate fails regression review if a protected metric declines or misses its required pass rate.

## FastAPI Interface

The service exposes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/runs` | Start an agent run and return a run ID |
| `GET` | `/runs/{run_id}` | Read the latest stored run snapshot |
| `POST` | `/api/v1/evals/run` | Execute evaluation tasks and stream SSE events |

The API run store is currently in memory. The SSE endpoint streams trial progress and completion events from `AsyncEvalRunner`.

Interactive API documentation is available at `/docs` while the service is running.

## Repository Layout

```text
auditable-scientific-agent-kernel/
├── src/agent_runtime/
│   ├── api.py                 # FastAPI and SSE endpoints
│   ├── graph/                 # Scientific LangGraph state, nodes, and routing
│   ├── scientific/            # ModelSpec, validators, compiler, executor, repairs
│   ├── evals/                 # Tasks, harness, graders, regression, failure analysis
│   ├── runtime/               # Structured agent control loop
│   ├── context/               # Context assembly and pruning
│   ├── retrieval/             # BM25, vector, hybrid retrieval, and citations
│   ├── memory/                # Working and episodic memory
│   ├── skills/                # Skill loading and activation
│   ├── jobs/                  # Durable job records and workers
│   └── mcp/                   # MCP client and tool bridge
├── tests/
├── scripts/
├── sample_data/
├── sample_knowledge/
└── pyproject.toml
```

## Quick Start

Requires Python 3.11 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest pytest-asyncio
```

Run the current scientific and evaluation tests:

```powershell
pytest tests/test_scientific_compiler.py `
       tests/test_scientific_agent_integration.py `
       tests/test_modelspec_domain_validation.py `
       tests/test_eval_dataset.py `
       tests/test_eval_graders.py `
       tests/test_eval_feedback_loop.py `
       tests/test_eval_api.py -q
```

Run the API:

```powershell
uvicorn agent_runtime.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Start a run:

```powershell
$body = @{
  user_input = "Explain ARIMA stationarity constraints."
  max_steps = 3
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/runs `
  -ContentType "application/json" `
  -Body $body
```

## Testing Notes

The focused scientific, evaluation, and MCP tests pass independently. Some inherited graph tests still target the previous HITL and MCP-job topology and are not compatible with the current scientific graph builder. They need to be migrated or the two workflows need to be composed explicitly before the entire legacy test suite can pass again.

This incompatibility should not be hidden by increasing LangGraph's recursion limit. Graph loops must terminate through explicit state transitions and bounded repair policies.

## Known Limitations

- `FakeProvider` is used instead of a real LLM provider.
- Scientific execution and dataset metadata are deterministic simulations.
- API run state is process-local and disappears on restart.
- The API background worker still wraps a synchronous runtime.
- The inherited HITL/MCP job graph is not currently composed with the scientific graph.
- Artifact URIs are evidence references, not a connected object store.
- Repair policies cover only a small set of known diagnostics.
- This is an educational control-plane prototype, not a secure sandbox.

## Next Steps

1. Add a provider interface backed by a real LLM API.
2. Separate scientific and legacy orchestration graphs, then compose them deliberately.
3. Persist API run snapshots and intermediate state in SQLite or PostgreSQL.
4. Add real statistical handlers and artifact storage.
5. Add security red-team datasets and trust-envelope enforcement.
6. Version evaluation baselines and generated reports.

## Design Principle

Scientific agents should not be trusted because their final answer sounds plausible. They should be trusted only to the extent that their specifications, trajectories, side effects, numerical results, repairs, and evidence can be inspected and reproduced.

## License

MIT License. See [LICENSE](LICENSE).
