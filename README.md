# auditable-scientific-agent-kernel

A local-first, test-driven kernel for building scientific agents that are measurable, policy-constrained, and able to compile natural-language research requests into validated, executable, and auditable model specifications.

This Week 4 project starts from [xinglin-li/durable-agent-orchestration-lab](https://github.com/xinglin-li/durable-agent-orchestration-lab). The template already provides the Agent runtime, context and retrieval stack, memory and skills, LangGraph persistence, human approval, asynchronous jobs, and MCP integration. This repository adds the evaluation, security, and scientific compilation layers on top of that foundation.

The governing rule is:

> Let the model propose a candidate specification. Let deterministic software validate, authorize, compile, execute, and evaluate it.

## Project Status

**Week 4 kickoff — work in progress.**

The repository currently contains the inherited Week 1–3 baseline. The Week 4 packages, datasets, graders, policy controls, ModelSpec compiler, and scientific integration workflow will be implemented incrementally.

Current baseline:

- bounded structured Agent runtime
- context assembly, hybrid retrieval, citations, memory, and skills
- LangGraph state, routing, checkpoints, and state history
- human-in-the-loop approve, reject, and edit flows
- asynchronous jobs, retries, cancellation, and idempotency
- MCP stdio client, tool bridge, capability registry, and approval policy
- FastAPI entry point and pytest suite

Week 4 target:

- [ ] trace-first local evaluation harness
- [ ] versioned eval datasets and isolated trials
- [ ] deterministic, trajectory, outcome, execution, policy, and efficiency graders
- [ ] bounded rubric-based LLM judge for subjective dimensions only
- [ ] baseline-versus-candidate regression reports
- [ ] hard, soft, and informational metric scorecards
- [ ] failure taxonomy and auditable `FailureEpisode` records
- [ ] trust envelopes, runtime guardrails, and security policy engine
- [ ] prompt-injection and data-exfiltration red-team regression suite
- [ ] versioned `ModelSpec` intermediate representation
- [ ] schema validation, domain validation, and static analysis
- [ ] deterministic `ModelSpec -> ExecutionPlan` compiler
- [ ] approved scientific handler registry and executor
- [ ] diagnostics, bounded repair, and human escalation
- [ ] artifact provenance and evidence-grounded reporting
- [ ] integrated Scientific Agent Kernel workflow

Requires Python 3.11+.

## Why This Project Exists

Agent quality cannot be reduced to whether the final answer sounds convincing. An agent may produce a polished report while using an unauthorized tool, leaking private data, violating temporal constraints, or failing to create the claimed artifact.

This project treats quality as several separate dimensions:

| Dimension | Question |
| --- | --- |
| Component quality | Did each tool, retriever, validator, and compiler stage behave correctly? |
| Trajectory quality | Did the agent follow an acceptable sequence of steps? |
| Outcome quality | Did the external environment contain the claimed result? |
| Policy compliance | Did execution respect trust, permission, approval, and leakage rules? |
| Efficiency | Did the agent stay within step, tool-call, latency, and token budgets? |

The project also rejects unrestricted code generation as a scientific execution strategy. Natural-language research requests must cross a stable intermediate representation before they can affect the execution environment.

```text
Natural-language research request
                |
                v
      Candidate ModelSpec
                |
                v
 Schema + domain + static validation
                |
                v
      Human approval if required
                |
                v
     Deterministic compiler
                |
                v
         ExecutionPlan
                |
                v
   Approved scientific handlers
                |
                v
 Diagnostics + bounded repair
                |
                v
 Evidence-grounded report + evals
```

## What This Adds to the Template

The [durable-agent-orchestration-lab](https://github.com/xinglin-li/durable-agent-orchestration-lab) answers:

> How can an agent pause, resume, survive service replacement, request approval, and execute long-running external work safely?

This repository answers three additional questions:

1. **Evaluation:** How do we know whether the agent improved or regressed?
2. **Security:** How do we constrain the influence of untrusted content and unauthorized actions?
3. **Scientific compilation:** How do we translate a research request into a model plan that can be validated before execution?

The first supported model families will be deliberately narrow:

- `seasonal_naive`
- `ets`
- `arima`

The architecture should later support dynamic regression, state-space models, MIDAS, VAR/VECM, stochastic volatility, and ensembles without weakening the validation boundary.

## Target Architecture

```text
==================================================================================================
                         AUDITABLE SCIENTIFIC AGENT KERNEL
==================================================================================================

  User / API
      |
      v
  LangGraph Scientific Workflow <---------------- SQLite checkpoints / thread history
      |
      +-- Context Engine
      |     +-- RAG and citations
      |     +-- memory
      |     +-- skills
      |
      +-- Security Boundary
      |     +-- trust envelope
      |     +-- input / output / tool guardrails
      |     +-- policy engine and approvals
      |     +-- audit log
      |     +-- red-team cases
      |
      +-- ModelSpec Pipeline
      |     +-- candidate parser
      |     +-- schema validator
      |     +-- domain validator
      |     +-- static analyzer
      |     +-- deterministic compiler
      |     +-- ExecutionPlan
      |
      +-- Durable Execution
      |     +-- MCP client / scientific server
      |     +-- approved handler registry
      |     +-- queue / worker / retry / cancellation
      |     +-- diagnostics and bounded repair
      |     +-- artifacts and provenance
      |
      +-- Eval Harness
            +-- datasets and isolated trials
            +-- traces and environment outcomes
            +-- deterministic graders
            +-- trajectory and execution graders
            +-- policy and efficiency graders
            +-- bounded LLM judge
            +-- regression scorecard
            +-- failure episodes and reports
```

## Evaluation Model

The local evaluation harness will use six explicit concepts:

| Concept | Meaning |
| --- | --- |
| `EvalTask` | A versioned test case with expected outcomes, trace rules, tools, artifacts, and limits |
| `Trial` | One isolated execution of an eval task |
| `Trace` | Ordered model, tool, guardrail, approval, handoff, and state-transition events |
| `Outcome` | The externally verifiable result of the run |
| `Grader` | A deterministic or bounded rubric-based evaluator |
| `EvalHarness` | The runner that executes trials, invokes graders, aggregates metrics, and emits reports |

### Grader boundaries

Use code whenever the property is deterministic.

| Grader | Appropriate checks |
| --- | --- |
| Deterministic | schema validity, exact fields, counts, identifiers |
| Trajectory | required order, forbidden events, approval before execution |
| Execution-based | artifact existence, schema, recomputed metrics, reproducibility |
| Outcome | external state matches the claimed result |
| Policy | permissions, trust boundaries, leakage, approval compliance |
| Efficiency | step count, tool calls, latency, token usage |
| Bounded LLM judge | clarity, completeness, professionalism, risk explanation |

The LLM judge must not decide permissions, schema validity, leakage, tool authorization, code-execution success, or numerical correctness.

### Regression scorecard

Metrics remain separated instead of being hidden inside one weighted score:

| Metric class | Behavior |
| --- | --- |
| Hard | Must not regress; some must remain at 100% |
| Soft | May trade off within declared thresholds |
| Informational | Supports diagnosis but does not directly decide pass/fail |

A candidate cannot be called an improvement when a hard safety or correctness metric regresses, even if cost or prose quality improves.

## Security Model

The security layer assumes that retrieved documents, emails, web pages, MCP resources, tool outputs, and user-provided files may contain hostile instructions.

Planned controls:

- explicit provenance and trust level on data crossing node boundaries
- structured extraction before untrusted content reaches privileged logic
- least-privilege capability and handler access
- human approval for high-risk actions
- separate input, output, and tool-action guardrails
- structured `GuardrailViolation` records
- append-only security audit events
- direct and indirect prompt-injection tests
- fake-approval, unauthorized-tool, path-traversal, and argument-injection tests
- data-exfiltration and private-data-leakage checks
- reproducible `RedTeamCase -> EvalTask -> regression` flow

Guardrails and evaluators have different jobs:

- a **guardrail** blocks dangerous behavior during execution
- a **validator** rejects an invalid request, specification, plan, or result
- an **evaluator** measures behavior during development or after execution
- a **human reviewer** resolves ambiguous or high-risk boundaries

## ModelSpec as an Intermediate Representation

`ModelSpec` is the stable boundary between probabilistic language interpretation and deterministic scientific execution. It plays a role similar to an AST, compiler IR, or query plan.

A structurally valid specification may still be scientifically invalid. The pipeline therefore separates:

1. **Syntax validation** — can the candidate output be parsed?
2. **Schema validation** — are required fields and types valid?
3. **Domain validation** — are model choices meaningful for the data and task?
4. **Static analysis** — is there leakage, insufficient sample size, or a constraint conflict?
5. **Execution validation** — did the approved handler execute successfully?
6. **Evidence validation** — does the final report agree with produced artifacts?

The LLM parser may propose a candidate `ModelSpec`; it cannot grant permissions, bypass validation, select an unapproved handler, or mark its own output as correct.

## Bounded Repair

Diagnostics may trigger a limited repair cycle when policy explicitly allows it.

```text
Execution failure
      |
      v
Structured diagnostic
      |
      v
Repair policy decision
      +-- disallowed / high risk -> human escalation
      |
      +-- allowed -> candidate repair
                         |
                         v
                 full re-validation
                         |
                         v
                 bounded re-execution
```

Every repair attempt must be recorded. Leakage, permission failures, and high-risk specification changes must not be silently repaired.

## Repository Layout

The current repository is the Week 1–3 template. Week 4 will add the following modules as they are implemented:

```text
auditable-scientific-agent-kernel/
  pyproject.toml
  README.md
  LICENSE
  src/
    agent_runtime/
      api.py
      graph/                  # inherited durable workflow; extended in Week 4
      jobs/                   # inherited durable job execution
      mcp/                    # inherited MCP client and bridge
      context/                # inherited context control
      retrieval/              # inherited hybrid retrieval
      memory/                 # inherited memory system
      skills/                 # inherited progressive skills
      runtime/                # inherited structured runtime
      tools/                  # inherited validated tools
      evals/                  # planned: harness, graders, regression, reports
      security/               # planned: trust, policy, guardrails, audit, red team
      scientific/             # planned: ModelSpec, compiler, executor, diagnostics
  evals/                      # planned: tasks, baselines, and generated reports
  sample_data/
    series/                   # inherited sample data
    scientific/               # planned scientific fixtures
    adversarial/              # planned hostile-content fixtures
  skills/
  tests/
  docs/                       # planned design and threat-model documents
```

Planned Week 4 package boundaries:

```text
src/agent_runtime/evals/
  models.py
  dataset.py
  harness.py
  aggregation.py
  regression.py
  scorecard.py
  failure_episode.py
  graders/

src/agent_runtime/security/
  models.py
  trust.py
  policy.py
  guardrails.py
  approvals.py
  audit.py
  red_team.py

src/agent_runtime/scientific/
  modelspec.py
  parser.py
  validators.py
  static_analysis.py
  compiler.py
  execution_plan.py
  executor.py
  diagnostics.py
  repairs.py
  evidence.py
  report.py
```

These files should be created incrementally, not scaffolded before their contracts and tests are needed.

## Quick Start

Create and activate a virtual environment:

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

Run the inherited baseline suite:

```bash
pytest -q
```

Run the inherited durable and MCP integration tests:

```bash
pytest tests/test_durable_macro_agent.py tests/test_graph_approval.py tests/test_graph_persistence.py -v
pytest tests/test_mcp_client_bridge.py tests/test_mcp_capability_registry.py tests/test_mcp_approval_policy.py -v
```

Run the API:

```bash
uvicorn agent_runtime.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Week 4 eval commands will be documented after the local harness and datasets exist.

## Week 4 Implementation Plan

### Day 1 — Trace-first eval harness

- define task, trial, trace, outcome, and grader contracts
- implement dataset loading and isolated local trials
- establish the failure taxonomy

### Day 2 — Graders and regression

- add deterministic, trajectory, outcome, execution, policy, and efficiency graders
- establish baseline-versus-candidate comparison
- implement the multidimensional quality scorecard

### Day 3 — Improvement loop

- convert failed traces into candidate eval tasks
- add skills regression coverage
- create `FailureEpisode` as an offline learning artifact

### Day 4 — Security boundary

- introduce trust envelopes and structured extraction
- add guardrails, policy decisions, violations, and audit events
- turn red-team attacks into repeatable regression cases

### Day 5 — ModelSpec

- define the versioned scientific intermediate representation
- add parser boundaries, schema validation, and domain validation
- support `seasonal_naive`, `ets`, and `arima`

### Day 6 — Compiler and bounded repair

- add static leakage and sample-size analysis
- compile validated specifications into dependency-ordered execution plans
- execute approved handlers and emit diagnostics
- implement bounded repair and human escalation

### Day 7 — Integrated kernel

- connect context, security, ModelSpec, durable execution, evidence, and evals
- add scientific, prompt-injection, red-team, and regression integration tests
- document the architecture, threat model, eval strategy, and repair policy

## Definition of Done

Week 4 is complete only when the repository can demonstrate all of the following:

- the same eval task can run through isolated, repeatable trials
- traces and external outcomes are evaluated separately
- deterministic checks take precedence over LLM judgment
- hard metric regression blocks an “improved” result
- failed traces can become reviewed eval tasks and failure episodes
- hostile retrieved content cannot become privileged instructions
- security violations are blocked and recorded in the audit trail
- red-team cases run as a reproducible regression suite
- a natural-language request becomes a versioned candidate `ModelSpec`
- schema, domain, leakage, and sample-size checks run before compilation
- only approved handlers can execute an `ExecutionPlan`
- artifacts carry provenance and reports are checked against evidence
- repair is bounded, revalidated, auditable, and escalated when necessary
- durable approval and job recovery continue to work

## Non-Goals

This iteration does not attempt to provide:

- arbitrary model-generated code execution
- a complete statistical modeling platform
- complex multi-agent collaboration
- production OAuth, cloud deployment, or Kubernetes
- a production message queue or multi-worker scheduler
- unrestricted online self-modification of prompts, skills, or policies
- complete MIDAS, state-space, VAR/VECM, or stochastic-volatility implementations
- a large-scale public benchmark

`FailureEpisode` and Reflexion-style analysis are offline improvement inputs. They must not automatically rewrite production prompts, skills, guardrails, or code.

## Security Notice

This project is an educational prototype. Its policy checks, prompt-injection tests, subprocess controls, and approval flows are application-level defenses, not a secure sandbox or formal security proof.

Do not execute untrusted scripts or MCP servers. Do not expose the API directly to the public internet. Production handling of untrusted code or sensitive data requires operating-system isolation, secret management, network egress controls, resource quotas, hardened authentication, and independent security review.

## Known Limitations at Kickoff

- Week 4 eval, security, and scientific packages are not implemented yet.
- The inherited provider is a deterministic fake used for tests.
- The inherited vector index uses deterministic fake embeddings.
- The macro job computation is simulated.
- The inherited job-store prototype is process-local.
- Artifact URIs are metadata examples rather than a connected object store.
- Context token estimation is approximate.
- MCP coverage is intentionally limited to the protocol surface exercised by the template tests.
- The current security controls do not eliminate prompt-injection risk.

This section will be updated as Week 4 implementation progresses.

## Next Stage

The completed kernel will become the foundation for **Chronos Research Agent**: an auditable agent for time-series investigation, specification, forecasting, model comparison, and decision support.

The kernel remains responsible for the stable control plane:

```text
Research objective
  -> ModelSpec
  -> validation
  -> compilation
  -> controlled execution
  -> diagnostics
  -> bounded repair
  -> evidence-grounded report
  -> regression evaluation
```

## Design Summary

Scientific agents need more than plausible language and successful tool calls. They need measurable trajectories, verifiable outcomes, explicit trust boundaries, stable intermediate representations, deterministic compilation, controlled execution, and evidence-backed reports.

This repository is the bridge from a general durable Agent runtime to an auditable scientific system.

## License

MIT License. See [LICENSE](LICENSE).
