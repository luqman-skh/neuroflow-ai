# NeuroFlow AI
## Agentic Experimental Data Analysis Platform

> **Project type:** AI Engineering / Agentic AI prototype  
> **Domain:** Neuroscience research and memory-task experiments  
> **Primary goal:** Let researchers describe an analysis request in natural language, review an LLM-generated execution plan, approve or modify it, and receive results from approved analysis tools.

---

## 1. Project overview

NeuroFlow AI is a human-in-the-loop agentic research analysis application for experimental data produced by memory-task applications, such as Unity3D-based desktop and mobile experiments.

A researcher enters a question such as:

> Compare memory performance between the control and experimental groups. Also investigate whether response time is associated with error rate.

The application uses an LLM to interpret the request and create a structured analysis plan. The plan identifies the analytical operations to run, the parameters and filters to use, and why each operation is relevant.

The researcher reviews the proposed plan, edits or removes individual operations if needed, and explicitly approves execution. The application then invokes Python analysis tools, displays their outputs and charts, and asks the LLM to synthesise the results into a cautious, evidence-grounded summary.

**The first version is a workflow/orchestration prototype, not a real neuroscience analysis product.** The statistical and ML functions are placeholders. They return deterministic illustrative outputs and charts; they do not analyse real participant data or make genuine scientific findings. The UI and all generated reports must clearly label these outputs as mock/demo results.

### Core workflow

1. Researcher submits a natural-language research question.
2. LLM identifies the research objectives, relevant variables, groups, and constraints.
3. LLM proposes a structured analysis plan using an allowlisted catalogue of analysis tools.
4. Application validates the plan and its parameters.
5. Researcher reviews, edits, approves, or rejects the plan.
6. Only after approval, the application executes the approved tools.
7. Results are validated and displayed.
8. LLM produces an evidence-grounded synthesis using only returned tool outputs.
9. The session and execution details are saved for reproducibility.

---

## 2. Product principles and boundaries

### Must-have principles

- **Human approval before execution:** no analysis tool may run merely because the LLM proposed it. The researcher must approve the plan first.
- **Deterministic tools:** Python functions perform calculations or return mock outputs. The LLM must not invent numerical results.
- **Structured contracts:** use validated Pydantic models for tool inputs, plans, execution records, and results.
- **Allowlisted tools:** the model may select only registered analysis tools. It must not generate arbitrary Python or SQL for execution.
- **Transparent execution:** show the researcher each selected tool, its purpose, parameters, filters, status, and returned result.
- **Traceability:** preserve the original request, proposed and approved plans, edits, tool calls, outputs, and final synthesis.
- **Honest demo outputs:** clearly mark placeholder results and charts as illustrative. Never imply they are real participant findings.
- **Reproducible workflow:** persist LangGraph state so an interrupted approval workflow can resume.
- **Separation of concerns:** keep the UI, graph orchestration, LLM integration, tool implementations, schemas, and persistence separate.

### Explicitly out of scope for the initial prototype

- Connecting to the real Unity3D application or real participant records.
- Implementing validated neuroscience/statistical algorithms.
- Training actual ML models.
- Diagnosing participants or inferring neurological conditions.
- Autonomous execution without researcher approval.
- RAG over research papers or protocols.
- MCP integration.
- Multi-agent collaboration.
- Authentication, role management, and production deployment infrastructure.

Do not add these features unless a later milestone explicitly requires them.

---

## 3. Proposed technology stack

Use a Python-first stack:

| Area | Technology | Purpose |
|---|---|---|
| Language | Python | Application and analysis tools |
| Agent orchestration | LangGraph | Stateful workflow, routing, approval interrupt, resume |
| LLM integration | LangChain chat model interface or supported LangGraph model integration | Provider-independent model calls |
| Local model option | Ollama | Local development |
| Hosted model option | Configurable API provider | Alternative model backend |
| Structured validation | Pydantic | Typed plans, parameters, and results |
| Frontend | Gradio Blocks | Research chat, plan review, approval, results |
| Charts | Plotly | Interactive statistical visualisations |
| Persistence | SQLite | Session history and local application data |
| Graph checkpointing | LangGraph SQLite checkpointer, where supported by installed package versions | Persist graph state across interrupts |
| Testing | pytest | Unit and integration tests |
| Configuration | pydantic-settings or python-dotenv | Environment-based configuration |
| Packaging | Docker (later milestone) | Reproducible runtime |

Keep the LLM provider configurable. Do not hardcode provider credentials or model names throughout the application. Read them from environment variables and provide a clear `.env.example`.

Prefer current stable package versions that are mutually compatible. Pin dependencies once the working environment is established.

---

## 4. High-level architecture

```text
                    ┌──────────────────────────┐
                    │     Gradio Blocks UI     │
                    │                          │
                    │ Research question        │
                    │ Plan review / editing    │
                    │ Approve / reject         │
                    │ Execution log            │
                    │ Results and charts       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       LangGraph          │
                    │                          │
                    │ 1. Understand request    │
                    │ 2. Generate plan         │
                    │ 3. Validate plan         │
                    │ 4. Interrupt for approval│
                    │ 5. Execute approved plan │
                    │ 6. Evaluate results      │
                    │ 7. Synthesise report     │
                    └───────┬──────────┬───────┘
                            │          │
                  ┌─────────▼───┐  ┌───▼────────────────┐
                  │ Pydantic    │  │ Analysis tool      │
                  │ schemas     │  │ registry           │
                  └─────────────┘  │                    │
                                   │ Mock Python tools  │
                                   │ - performance      │
                                   │ - learning curve   │
                                   │ - response time    │
                                   │ - group comparison │
                                   │ - correlations     │
                                   │ - task difficulty  │
                                   │ - clustering       │
                                   │ - prediction       │
                                   │ - data validation  │
                                   │ - anomalies        │
                                   └─────────┬──────────┘
                                             │
                                             ▼
                                   ┌────────────────────┐
                                   │ Structured results │
                                   │ and Plotly data    │
                                   └────────────────────┘

                   ┌─────────────────────────────┐
                   │ SQLite                      │
                   │ Sessions, plans, approvals, │
                   │ execution history, graph    │
                   │ checkpoints                 │
                   └─────────────────────────────┘
```

### Suggested graph nodes

Use one primary LangGraph graph. Do not create multiple agents just for the sake of having a multi-agent architecture.

1. `understand_request`: interpret the research question and extract intent.
2. `generate_analysis_plan`: select relevant registered tools and produce a structured plan.
3. `validate_analysis_plan`: validate tool names, required arguments, parameter ranges, and dependencies.
4. `human_approval`: pause with a LangGraph interrupt and wait for researcher input.
5. `execute_analysis_plan`: execute only the approved and validated operations.
6. `evaluate_execution_results`: inspect tool statuses, required outputs, and failures.
7. `synthesize_results`: produce a summary grounded in the returned results.
8. `persist_session`: save the completed session and its audit details.

Use conditional edges for:
- Invalid plan → regenerate or request clarification.
- Rejected plan → return to planning with researcher feedback.
- Approved plan → execute.
- Tool failure → record failure and follow a bounded error-handling path.
- Incomplete results or justified follow-up → optionally propose another analysis round.

Keep the first version's iterative loop bounded. Do not let the LLM run analyses indefinitely.

### Human approval and checkpointing

Use LangGraph's interrupt/resume pattern for the approval gate. Configure a persistent checkpointer and a stable `thread_id` per research session.

The graph must pause after plan validation and before tool execution. The UI should display the pending plan and resume the same graph thread after the researcher approves, edits, or rejects it.

Do not implement approval as an LLM-generated boolean. Approval must be an explicit UI action and a trusted application input.

If the installed LangGraph version uses a different API from examples in this document, follow the installed version's official API while preserving the required behaviour.

---

## 5. Analysis tool catalogue

Create ten placeholder analysis functions. Each function should have a clear docstring, typed parameters, deterministic mock output, and a consistent result contract.

The functions should return structured Python objects, not preformatted prose strings. The LLM can explain the returned outputs later, but it must not fabricate or modify the numerical values.

### 5.1 Participant performance

#### `analyze_memory_performance`

Purpose:
- Summarise participant memory-task performance.
- Return illustrative accuracy, recall score, completion time, and distribution summaries.

Example parameters:
- `metric`: selected outcome metric.
- `task_type`: task filter or `"all"`.
- `group`: optional experimental group.
- `participant_ids`: optional list of participant IDs.

Example output:
- Summary statistics.
- Group or task breakdown, where requested.
- Chart specification/data.
- Filters actually applied.

#### `analyze_learning_curve`

Purpose:
- Illustrate performance progression over repeated trials or sessions.

Example parameters:
- `metric`: e.g. accuracy or response time.
- `group`: optional group filter.
- `trial_start`: first trial.
- `trial_end`: last trial.

Example output:
- Values by trial.
- Trend chart data.
- Short descriptive interpretation.

#### `analyze_response_time`

Purpose:
- Illustrate response-time distributions and variability.

Example parameters:
- `task_type`.
- `group`.
- `participant_ids`.
- Optional trial range.

Example output:
- Mean/median response time.
- Variability and quantiles.
- Distribution chart data.
- Outlier summary (mock).

### 5.2 Group and task analysis

#### `compare_experimental_groups`

Purpose:
- Illustrate a comparison of an outcome between two experimental groups.

Example parameters:
- `metric`.
- `group_a`.
- `group_b`.
- `task_type`.
- Optional participant filter.

Example output:
- Group summaries.
- Illustrative test name and statistics.
- Effect-size field and confidence-interval fields.
- Group comparison chart data.

The mock output must be labelled illustrative. Do not imply that a statistical test was genuinely performed.

#### `analyze_performance_correlations`

Purpose:
- Illustrate associations between performance variables.

Example parameters:
- `x_variable`.
- `y_variable`.
- `group` or optional task filter.
- `correlation_method` (e.g. Pearson or Spearman).

Example output:
- Correlation coefficient.
- Illustrative p-value.
- Scatterplot data.
- Correlation matrix data if useful.

#### `analyze_task_difficulty`

Purpose:
- Illustrate differences in performance across memory tasks.

Example parameters:
- `metric`.
- `group`.
- `sort_order`.

Example output:
- Task-level summaries.
- Ranked task performance.
- Bar chart or heatmap data.

### 5.3 ML-based behavioural analysis

#### `cluster_participant_behaviour`

Purpose:
- Return illustrative participant behavioural clusters based on features such as response time, error rate, retry frequency, and accuracy.

Example parameters:
- `features`.
- `n_clusters`.
- `group` (optional).

Example output:
- Cluster labels and sizes.
- Cluster profile summaries.
- 2D scatter/PCA-style chart data.
- Explicit note that the clusters are mock outputs and no model was trained.

#### `predict_task_performance`

Purpose:
- Demonstrate the interface of a supervised prediction pipeline for a predefined outcome.

Example parameters:
- `target`.
- `features`.
- `task_type`.

Example output:
- Illustrative classification metrics.
- Confusion-matrix data.
- Feature-importance data.
- Explicit note that no model was trained or evaluated.

### 5.4 Data quality and monitoring

#### `validate_experiment_data`

Purpose:
- Return a mock data-quality report.

Example parameters:
- `task_type`.
- `group`.
- Optional session or participant filters.

Example output:
- Missing-record count.
- Duplicate-event count.
- Invalid timestamp count.
- Incomplete-session count.
- Validation status and warnings.

This tool should be treated as a required pre-analysis step in the proposed workflow, unless the researcher explicitly uses a demo mode that skips it. Make the default workflow include it.

#### `detect_anomalous_sessions`

Purpose:
- Return illustrative sessions flagged for researcher review.

Example parameters:
- `sensitivity`.
- `task_type`.
- `group`.

Example output:
- Example session IDs.
- Mock anomaly scores.
- Reasons for flagging.
- A clear warning that flagged sessions are not necessarily invalid.

### Tool implementation requirements

- All ten functions must be callable independently in tests.
- Use deterministic outputs so repeated calls with the same arguments return the same mock values.
- Echo the actual input parameters and filters in the result.
- Validate arguments before returning a result.
- Return a common `AnalysisResult` model or a clearly defined common result envelope.
- Include a `mock_data: true` or equivalent field in every result.
- Include a limitations/warnings field in every result.
- Do not make network calls from these placeholder tools.
- Do not execute generated code.
- Keep chart data structured and serialisable; the UI should render charts from it.

---

## 6. Data and schema design

The initial prototype should have a small synthetic experiment context, even though the analysis functions return mock outputs.

Create a demo data dictionary that describes the variables available to the agent. It can be a JSON or Python object and should include:

| Field | Description | Example |
|---|---|---|
| `participant_id` | Pseudonymous participant identifier | `P001` |
| `group` | Experimental condition | `control` / `experimental` |
| `task_id` | Memory task identifier | `memory_recall_01` |
| `task_type` | Task category | `recall`, `recognition`, `spatial_memory` |
| `trial_number` | Trial sequence number | `1` |
| `response_correct` | Whether the response was correct | `true` |
| `response_time_ms` | Response time in milliseconds | `1850` |
| `attempt_count` | Number of attempts | `2` |
| `session_completed` | Whether the session completed | `true` |
| `event_timestamp` | Event timestamp | ISO 8601 timestamp |
| `session_id` | Session identifier | `S001` |

These are proposed demo fields, not a claim about the exact schema of the original DZNE application. Keep the data dictionary configurable and make the agent aware only of fields that are actually defined.

### Suggested Pydantic models

At minimum, define models for:

- `ResearchRequest`
- `ExperimentContext`
- `AnalysisPlan`
- `AnalysisStep`
- `ToolParameters` or per-tool argument models
- `AnalysisResult`
- `ExecutionRecord`
- `ApprovalDecision`
- `ResearchSession`
- `FinalReport`

An analysis step should contain fields similar to:

```python
class AnalysisStep(BaseModel):
    step_id: str
    tool_name: str
    purpose: str
    parameters: dict
    rationale: str
    dependencies: list[str] = []
```

Use dedicated Pydantic argument models for individual tools where practical. Avoid relying on unrestricted dictionaries as the only validation layer.

A common result envelope could contain:

```python
class AnalysisResult(BaseModel):
    analysis_id: str
    analysis_name: str
    status: str
    mock_data: bool = True
    filters_applied: dict
    statistics: dict
    chart_data: dict | list | None = None
    interpretation: str
    limitations: list[str] = []
    warnings: list[str] = []
```

Adapt field types where necessary to support Plotly chart specifications cleanly. Ensure every result can be serialised to JSON.

---

## 7. LLM behaviour and planning rules

The LLM should:

- Understand the researcher's natural-language objective.
- Identify relevant variables from the configured experiment data dictionary.
- Select one or more tools from the registered catalogue.
- Generate valid tool parameters and filters.
- Explain why each tool is relevant.
- Identify ambiguities and ask clarifying questions when necessary.
- Avoid proposing analyses unsupported by the available variables.
- Avoid unnecessary analyses that do not address the research question.
- Produce a concise synthesis grounded in tool outputs.

The LLM must not:

- Invent available dataset fields.
- Invent participant counts or numerical results.
- Claim that mock statistical calculations were performed.
- Execute tools before approval.
- Select tools outside the registry.
- Generate arbitrary Python or SQL for execution.
- Interpret association as causation.
- Claim that an ML model was trained when the tool only returns mock metrics.
- present a mock result as a real scientific finding.

### Planning example

Researcher request:

> Compare memory accuracy between control and experimental groups and investigate whether response time is associated with error rate.

A valid plan might contain:

1. `validate_experiment_data` — check data-quality status before analysis.
2. `analyze_memory_performance` — establish descriptive performance summaries.
3. `compare_experimental_groups` — compare memory accuracy between the two specified groups.
4. `analyze_performance_correlations` — examine response time and error rate.

The plan must include the actual parameters, filters, rationale, and execution order. The researcher must see this plan before any analysis runs.

---

## 8. Frontend requirements

Use `gr.Blocks` rather than a minimal chat-only interface. Keep the first UI functional and clear rather than spending time on elaborate styling.

### Tab 1: Research workspace

Include:
- Project/application heading.
- Demo experiment selector or visible active experiment context.
- Research question textbox.
- Submit button.
- Conversation/history area.
- Clarification questions from the LLM.
- Current workflow status.

### Tab 2: Analysis plan and approval

Include:
- Generated analysis steps.
- Tool/function name for each step.
- Purpose and rationale.
- Editable parameters and filters.
- Ability to remove an individual step.
- Approve-all action.
- Reject-plan action with a feedback textbox.
- Clear status showing that execution has not started while approval is pending.

If a researcher edits the plan, validate the edited plan again before execution.

### Tab 3: Results

Include:
- Execution status for each tool.
- Execution log.
- Filters/parameters actually passed to each tool.
- Metrics and statistics.
- Plotly charts.
- LLM-generated synthesis.
- Limitations and mock-data warnings.
- Exportable report, if straightforward.

### Interaction requirements

- The UI must preserve the graph `thread_id` for the active research session.
- Approval must resume the paused graph, not start a new independent graph.
- Rejection must route back to planning with researcher feedback.
- Show loading and error states.
- Do not show a successful result before the tool has returned one.
- Clearly distinguish proposed, approved, running, completed, and failed operations.

---

## 9. Persistence and reproducibility

Use SQLite for the initial application.

Persist enough information to reconstruct a research session:

- Session/thread ID.
- Creation and completion timestamps.
- Original researcher request.
- Experiment context/data-dictionary version.
- Generated plan.
- Researcher-edited and approved plan.
- Approval/rejection decision and feedback.
- Tool names and parameters.
- Execution statuses and structured outputs.
- Final synthesis/report.
- Model/provider identifier and prompt version, where available.

Use LangGraph checkpointing for graph state and application-level persistence for session history. Avoid duplicating large data unnecessarily; store serialised results or references to them.

The application should be able to list previous sessions and reopen their plans and results. Full replay is optional for the first version.

---

## 10. Testing and quality requirements

Use pytest. Tests are required for core behaviour, not an optional final polish.

### Unit tests

- Every mock analysis tool returns a valid result.
- Every tool echoes the parameters it received.
- Outputs are deterministic.
- Invalid parameters are rejected.
- Results are JSON serialisable.
- Every result is marked as mock/demo output.

### Planner and validation tests

- A group-comparison question proposes a relevant group-comparison tool.
- A learning-progression question proposes a learning-curve tool.
- Ambiguous requests trigger clarification where necessary.
- Unknown variables are rejected or clarified.
- Unknown tool names are rejected.
- Invalid parameter values are rejected.
- The plan contains a rationale for each step.

Avoid tests that depend on exact wording from a live LLM. Where possible, test the structured schema and graph behaviour with a deterministic fake model.

### Graph workflow tests

- The graph pauses before tool execution.
- No tool executes before approval.
- Approval resumes the same thread and executes only approved steps.
- Editing a plan causes it to be revalidated.
- Rejection returns to planning with feedback.
- Tool errors are recorded and do not silently appear as successful results.
- Checkpointed state can resume after an interrupt.
- Iteration limits are respected.

### UI smoke tests

Verify that:
- A request can be submitted.
- A generated plan is displayed.
- Approval and rejection controls work.
- Results and charts render.
- Errors are visible to the user.

### Demo evaluation set

Create a small set of representative requests:

1. Compare memory accuracy between two groups.
2. Examine learning progression over trials.
3. Investigate response time and error rate association.
4. Identify tasks with the lowest accuracy.
5. Find unusual or incomplete sessions.
6. Cluster participants by behavioural features.
7. Predict a predefined task outcome.
8. Ask an ambiguous question that requires clarification.
9. Ask about a variable that is not in the data dictionary.
10. Request an unsupported analysis and verify that the system refuses or asks for a suitable alternative.

---

## 11. Development roadmap

Implement the project in the milestones below. Complete and verify each milestone before moving to the next. Do not attempt to implement the entire application in one step.

### Milestone 1 — Statistical tool library and data contracts

**Objective:** Build the complete placeholder analysis layer independently of the LLM, graph, and frontend.

Tasks:

1. Inspect the existing repository before changing anything.
2. Establish a clean Python project structure and dependency/configuration approach.
3. Define the demo experiment context and data dictionary.
4. Define Pydantic input models for all ten analysis tools.
5. Define a common `AnalysisResult` output contract.
6. Implement the ten deterministic placeholder functions listed in Section 5.
7. Ensure each function returns realistic-looking structured statistics, chart data, filters, interpretation, and limitations.
8. Mark every output explicitly as mock/demo data.
9. Create a tool registry mapping safe tool names to functions, schemas, and descriptions.
10. Add unit tests for every function, input validation, deterministic output, and JSON serialisation.
11. Add a small standalone script or example showing how to call tools and inspect their outputs.

**Acceptance criteria:**

- All ten tools can be called independently.
- Every tool has validated parameters.
- Every tool returns the common result envelope.
- Outputs are deterministic and explicitly marked as mock data.
- Actual parameters and filters are echoed in the output.
- Chart data is serialisable and suitable for Plotly rendering.
- Tests pass.
- No LLM, LangGraph, or Gradio integration is required in this milestone.

**Stop after Milestone 1.** Report what was created, how to run the tests, any design decisions, and any remaining issues. Do not start Milestone 2 until explicitly instructed.

### Milestone 2 — LLM integration and structured analysis planning

**Objective:** Translate natural-language research questions into validated analysis plans.

Tasks:

1. Add configurable LLM provider/model integration.
2. Implement research request and experiment-context prompts.
3. Expose the tool catalogue and data dictionary to the planner.
4. Generate structured Pydantic `AnalysisPlan` output.
5. Include tool name, rationale, parameters, filters, dependencies, and execution order in each step.
6. Add clarification handling for ambiguous requests.
7. Validate plans against the registered tool catalogue and tool-specific argument schemas.
8. Add deterministic fake-model tests and a small set of live-model integration tests if credentials are configured.

**Acceptance criteria:**

- Representative questions generate structurally valid plans.
- The planner uses only registered tools.
- Unknown variables and invalid arguments are rejected or clarified.
- Each proposed tool has an understandable rationale.
- No analysis function executes in this milestone.

### Milestone 3 — LangGraph orchestration and human approval

**Objective:** Build the stateful agent workflow with a real approval gate.

Tasks:

1. Define typed graph state.
2. Implement request understanding and planning nodes.
3. Implement plan validation.
4. Implement the human approval interrupt.
5. Add approve, reject, and edit/resubmit paths.
6. Execute only approved, validated plan steps.
7. Implement result evaluation and error handling.
8. Implement result synthesis grounded in tool outputs.
9. Add persistent checkpointing and stable thread IDs.
10. Add graph tests for pause, approval, rejection, resume, and failures.

**Acceptance criteria:**

- The graph pauses before tool execution.
- No tool runs without explicit researcher approval.
- The graph resumes the same thread after approval.
- Edited plans are revalidated.
- Rejected plans can be regenerated using researcher feedback.
- Tool errors are represented honestly.
- The final synthesis cannot invent tool statistics.

### Milestone 4 — Gradio frontend

**Objective:** Provide a usable researcher-facing interface.

Tasks:

1. Build a `gr.Blocks` application with Research Workspace, Analysis Plan, and Results sections/tabs.
2. Connect the research input to the graph.
3. Display generated plans and rationales.
4. Provide editable parameters and the ability to remove steps.
5. Implement approve, reject, and edit/resubmit controls.
6. Display workflow status and execution logs.
7. Render structured metrics and Plotly charts.
8. Display the LLM synthesis, warnings, and mock-data labels.
9. Handle loading, validation, and failure states.
10. Preserve the active graph thread ID across UI interactions.

**Acceptance criteria:**

- A researcher can complete the end-to-end workflow through the UI.
- Approval and rejection work correctly.
- The UI displays the actual tool calls and parameters.
- Results and charts render from tool outputs.
- Mock results are clearly labelled.

### Milestone 5 — Advanced agentic behaviour and session history

**Objective:** Add bounded iterative analysis and reproducibility features.

Tasks:

1. Allow the evaluator to identify incomplete results or justified follow-up questions.
2. Let the planner propose supplementary analyses based on intermediate outputs.
3. Require researcher approval for every supplementary execution round.
4. Add a strict iteration limit and stopping conditions.
5. Persist sessions, plans, approvals, tool executions, results, and final reports.
6. Add a session-history view.
7. Add audit-friendly execution records.
8. Record model/provider and prompt version where practical.

**Acceptance criteria:**

- Follow-up analysis is proposed only when justified.
- No additional tools execute without approval.
- The iteration limit is enforced.
- Previous sessions can be reopened and inspected.
- The analysis history is sufficiently detailed to understand what happened.

### Milestone 6 — Testing, documentation, and deployment

**Objective:** Make the prototype demonstrable and straightforward to run.

Tasks:

1. Expand unit, graph integration, and UI smoke tests.
2. Create the ten-question demo evaluation set described in Section 10.
3. Document expected behaviour and known limitations.
4. Add `.env.example`.
5. Add Dockerfile and, if useful, Docker Compose.
6. Document local and containerised startup.
7. Add architecture diagrams and screenshots to the README.
8. Document the mock-data boundary and the path to replacing mock tools with real analysis implementations.

**Acceptance criteria:**

- Tests pass in a clean environment.
- Another developer can run the project using the README.
- The demo can be completed from request through report.
- No secrets are committed.
- The limitations of the prototype are explicit.

---

## 12. Suggested repository structure

Use this as a target structure, but adapt it to the repository if a sensible structure already exists.

```text
neuroflow-ai/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   └── neuroflow/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── experiment.py
│       │   ├── analysis.py
│       │   ├── planning.py
│       │   └── session.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   └── experiment_context.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── performance.py
│       │   ├── group_analysis.py
│       │   ├── behavioural_ml.py
│       │   ├── data_quality.py
│       │   └── registry.py
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── state.py
│       │   ├── prompts.py
│       │   ├── planner.py
│       │   ├── nodes.py
│       │   ├── routing.py
│       │   └── graph.py
│       │
│       ├── persistence/
│       │   ├── __init__.py
│       │   ├── database.py
│       │   └── sessions.py
│       │
│       └── ui/
│           ├── __init__.py
│           ├── app.py
│           └── components.py
│
├── tests/
│   ├── unit/
│   │   ├── test_tools.py
│   │   ├── test_schemas.py
│   │   └── test_registry.py
│   ├── integration/
│   │   └── test_graph.py
│   └── evaluation/
│       └── test_planning_cases.py
│
└── scripts/
    └── demo_tools.py
```

Do not create empty files and directories just to match this tree. Add modules when they have a clear purpose.

---

## 13. Coding-agent operating instructions

The coding agent implementing this project must follow these rules.

### Before starting any milestone

1. Inspect the repository, existing files, and current Git status.
2. Read this README fully.
3. Identify existing conventions and avoid overwriting useful work.
4. Check the current Python version and installed tooling.
5. Present a short implementation plan for the requested milestone.

### While implementing

- Work only on the currently requested milestone.
- Do not silently skip acceptance criteria.
- Prefer small, cohesive modules and clear interfaces.
- Use type hints and docstrings for public functions.
- Keep configuration and secrets outside source code.
- Avoid unnecessary abstractions and dependencies.
- Do not add real statistical calculations in place of the requested placeholders during Milestone 1.
- Do not build the next milestone early.
- Run relevant tests after changes and fix failures.
- Do not claim that tests pass unless they were actually run.
- Update this README only when implementation decisions materially change the documented design; explain any deviation.
- If a requirement is ambiguous or conflicts with the existing repository, explain the issue and propose the smallest sensible resolution rather than guessing.

### At the end of each milestone

Provide:

1. Summary of files created or changed.
2. Important implementation decisions.
3. Commands to install, run, and test the work.
4. Test results, including failures.
5. Acceptance-criteria checklist.
6. Known limitations and outstanding issues.
7. Confirmation that work stopped at the requested milestone.

Do not automatically continue to the next milestone.

---

## 14. Getting started

Once this README is in the repository, start the coding agent with:

> Read the README.md in full and follow its coding-agent operating instructions. Inspect the existing repository and Git status first. Start **Milestone 1 only: Statistical tool library and data contracts**. Implement the ten deterministic placeholder analysis tools, their Pydantic schemas, the demo experiment context, the tool registry, and comprehensive unit tests. Do not implement LangGraph, LLM integration, Gradio, or any later milestone yet. Run the tests, verify every Milestone 1 acceptance criterion, and report the results. Stop after Milestone 1.

---

## 15. Definition of success

The completed prototype should let a researcher ask a question in natural language, inspect and modify the proposed analytical workflow, explicitly approve it, observe the selected tools and their parameters being executed, and receive structured illustrative results and a cautious synthesis.

The engineering value of NeuroFlow AI is not the mock statistics. It is the reliable, transparent, stateful orchestration of analytical tools under researcher supervision.
