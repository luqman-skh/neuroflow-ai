"""Graph nodes implementing the NeuroFlow analysis lifecycle."""

import uuid
from datetime import datetime, timezone
from typing import Any
from langgraph.types import interrupt

from neuroflow.agent.planner import AnalysisPlanner
from neuroflow.agent.state import AgentState
from neuroflow.agent.validator import validate_analysis_plan
from neuroflow.config import get_settings
from neuroflow.data.experiment_context import get_default_experiment_context
from neuroflow.persistence.sessions import get_session_repository
from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.experiment import ExperimentContext
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep, ResearchRequest
from neuroflow.schemas.session import (
    ApprovalDecision,
    ExecutionRecord,
    FinalReport,
    ResearchSession,
)
from neuroflow.tools.registry import ToolRegistry, get_tool_registry


class NodeContext:
    """Dependency container for graph nodes."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        context: ExperimentContext | None = None,
        planner: AnalysisPlanner | None = None,
    ) -> None:
        self.registry = registry or get_tool_registry()
        self.context = context or get_default_experiment_context()
        self.planner = planner or AnalysisPlanner(registry=self.registry, context=self.context)


# Global default node dependencies
_default_node_context = NodeContext()


def get_node_context() -> NodeContext:
    """Return default node dependencies."""
    return _default_node_context


def set_node_context(ctx: NodeContext) -> None:
    """Set the active node dependencies for testing or custom configuration."""
    global _default_node_context
    _default_node_context = ctx


# --- 1. Understand Request Node ---


def understand_request_node(state: AgentState) -> dict[str, Any]:
    """Interpret the incoming research query and initialize the session state."""
    raw_req = state.get("request")
    session_id = state.get("session_id") or f"sess_{uuid.uuid4().hex[:8]}"
    settings = get_settings()

    if isinstance(raw_req, str):
        req = ResearchRequest(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            query=raw_req.strip(),
            experiment_id=_default_node_context.context.experiment_id,
        )
    elif isinstance(raw_req, dict):
        req = ResearchRequest(**raw_req)
    elif isinstance(raw_req, ResearchRequest):
        req = raw_req
    else:
        req = ResearchRequest(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            query="Analyze experiment data",
            experiment_id=_default_node_context.context.experiment_id,
        )

    return {
        "session_id": session_id,
        "request": req,
        "status": "request_understood",
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "results": state.get("results", {}),
        "execution_records": state.get("execution_records", []),
        "validation_errors": [],
        "approval_status": "pending",
        "follow_up_proposed": False,
        "follow_up_rationale": None,
        "supplementary_rounds": state.get("supplementary_rounds", 0),
        "max_supplementary_rounds": state.get("max_supplementary_rounds", 0),
        "model_provider": state.get("model_provider") or settings.llm_provider,
        "model_name": state.get("model_name") or settings.llm_model,
    }


# --- 2. Generate Analysis Plan Node ---


def generate_analysis_plan_node(state: AgentState) -> dict[str, Any]:
    """Select registered tools and generate a structured analysis plan."""
    planner = _default_node_context.planner
    req = state["request"]
    feedback = state.get("feedback")
    iteration = state.get("iteration_count", 0) + 1

    query_to_plan = req.query
    if feedback:
        query_to_plan = f"{req.query}. Researcher revision feedback: {feedback}"

    planning_res = planner.plan(query_to_plan)

    if planning_res.status == "clarification_needed" and planning_res.clarification:
        return {
            "clarification": planning_res.clarification,
            "status": "clarification_needed",
            "iteration_count": iteration,
        }

    if planning_res.status == "rejected":
        return {
            "validation_errors": planning_res.validation_errors,
            "status": "plan_rejected",
            "iteration_count": iteration,
        }

    return {
        "plan": planning_res.plan,
        "feedback": None,  # clear consumed feedback
        "status": "plan_generated",
        "iteration_count": iteration,
        "validation_errors": [],
    }


# --- 3. Validate Analysis Plan Node ---


def validate_analysis_plan_node(state: AgentState) -> dict[str, Any]:
    """Validate tool names, required arguments, parameter ranges, and step dependencies."""
    plan = state.get("plan")
    if not plan:
        return {
            "validation_errors": ["No analysis plan was found to validate."],
            "status": "invalid_plan",
        }

    is_valid, errors = validate_analysis_plan(
        plan=plan,
        registry=_default_node_context.registry,
        context=_default_node_context.context,
    )

    if not is_valid:
        return {
            "validation_errors": errors,
            "status": "invalid_plan",
        }

    return {
        "validation_errors": [],
        "status": "awaiting_approval",
    }


# --- 4. Human Approval Node (Approval Gate Interrupt) ---


def human_approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval gate. Pauses execution via interrupt until researcher decides."""
    plan = state.get("plan")
    session_id = state.get("session_id", "unknown_session")
    is_followup = state.get("follow_up_proposed", False)

    msg = (
        f"Supplementary follow-up round proposed: {state.get('follow_up_rationale')}. "
        f"Researcher approval is strictly required before execution."
        if is_followup
        else (
            "An analysis plan has been proposed and validated. "
            "Researcher approval is strictly required before any tool executes."
        )
    )

    interrupt_payload = {
        "type": "human_approval_required",
        "session_id": session_id,
        "is_supplementary_round": is_followup,
        "follow_up_rationale": state.get("follow_up_rationale"),
        "plan": plan.model_dump() if plan else None,
        "summary": plan.summary if plan else "",
        "message": msg,
    }

    raw_decision = interrupt(interrupt_payload)

    # Process resumed decision from researcher
    if isinstance(raw_decision, dict):
        decision = ApprovalDecision(**raw_decision)
    elif isinstance(raw_decision, ApprovalDecision):
        decision = raw_decision
    elif isinstance(raw_decision, bool):
        decision = ApprovalDecision(
            plan_id=plan.plan_id if plan else "plan",
            approved=raw_decision,
        )
    else:
        raise ValueError(f"Invalid approval decision type received: {type(raw_decision).__name__}")

    # Check for plan modifications by researcher
    if decision.approved:
        if decision.modified_steps:
            updated_plan = plan.model_copy(deep=True)
            updated_plan.steps = decision.modified_steps
            return {
                "approval_decision": decision,
                "approval_status": "modified",
                "plan": updated_plan,
                "follow_up_proposed": False,
                "status": "plan_modified",
            }
        return {
            "approval_decision": decision,
            "approval_status": "approved",
            "follow_up_proposed": False,
            "status": "approved",
        }

    return {
        "approval_decision": decision,
        "approval_status": "rejected",
        "feedback": decision.feedback,
        "follow_up_proposed": False,
        "status": "rejected",
    }


# --- 5. Execute Analysis Plan Node ---


def execute_analysis_plan_node(state: AgentState) -> dict[str, Any]:
    """Execute ONLY approved and validated analysis plan steps."""
    # Strict execution barrier: verify approval status
    if state.get("approval_status") != "approved":
        raise PermissionError(
            "Execution blocked: human researcher approval must be granted before tools can execute."
        )

    plan = state.get("plan")
    if not plan or not plan.steps:
        return {"status": "execution_failed", "validation_errors": ["No approved steps to execute."]}

    registry = _default_node_context.registry
    new_records: list[ExecutionRecord] = []
    results: dict[str, AnalysisResult] = dict(state.get("results", {}))

    for step in plan.steps:
        record_id = f"exec_{uuid.uuid4().hex[:8]}"
        try:
            result = registry.execute_tool(step.tool_name, step.parameters)
            results[step.step_id] = result
            record = ExecutionRecord(
                record_id=record_id,
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters_passed=step.parameters,
                status=result.status,
                result=result,
                error_message=None,
                executed_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            record = ExecutionRecord(
                record_id=record_id,
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters_passed=step.parameters,
                status="error",
                result=None,
                error_message=str(e),
                executed_at=datetime.now(timezone.utc),
            )
        new_records.append(record)

    return {
        "execution_records": new_records,
        "results": results,
        "status": "executed",
    }


# --- 6. Evaluate Execution Results Node (Advanced Iterative Agentic Behaviour) ---


def evaluate_execution_results_node(state: AgentState) -> dict[str, Any]:
    """Inspect execution statuses, identify incomplete results, and propose justified follow-ups."""
    records = state.get("execution_records", [])
    results = state.get("results", {})
    session_id = state.get("session_id", "sess")
    supp_rounds = state.get("supplementary_rounds", 0)
    max_supp_rounds = state.get("max_supplementary_rounds", 1)

    executed_tools = {r.tool_name for r in records}

    # Strict stopping condition check
    if supp_rounds >= max_supp_rounds:
        return {
            "follow_up_proposed": False,
            "status": "evaluation_completed_max_rounds",
        }

    # Evaluate intermediate results to determine if a follow-up is justified:
    follow_up_step: AnalysisStep | None = None
    follow_up_rationale: str | None = None

    # Criterion 1: Data validation found incomplete sessions or missing records, but anomalous scan has not run
    if "validate_experiment_data" in executed_tools and "detect_anomalous_sessions" not in executed_tools:
        val_res = next((r for r in results.values() if r.analysis_name == "validate_experiment_data"), None)
        if val_res and val_res.statistics.get("incomplete_session_count", 0) > 0:
            follow_up_step = AnalysisStep(
                step_id="step_followup_1",
                tool_name="detect_anomalous_sessions",
                purpose="Scan flagged incomplete sessions with medium sensitivity to locate outlier runs.",
                parameters={"sensitivity": "medium", "task_type": "all"},
                rationale="Data validation identified incomplete sessions. A targeted anomaly audit is justified.",
            )
            follow_up_rationale = (
                "Initial data audit detected incomplete participant sessions. "
                "Running 'detect_anomalous_sessions' is recommended to isolate affected runs."
            )

    # Criterion 2: Group comparison found substantial group difference, but correlation was not run
    elif "compare_experimental_groups" in executed_tools and "analyze_performance_correlations" not in executed_tools:
        grp_res = next((r for r in results.values() if r.analysis_name == "compare_experimental_groups"), None)
        if grp_res and grp_res.statistics.get("effect_size_cohens_d", 0) > 0.6:
            follow_up_step = AnalysisStep(
                step_id="step_followup_1",
                tool_name="analyze_performance_correlations",
                purpose="Investigate speed-accuracy correlation to explain the group performance difference.",
                parameters={"x_variable": "response_time_ms", "y_variable": "error_rate"},
                rationale="Significant group difference detected. Correlating latency with errors provides mechanistic insight.",
            )
            follow_up_rationale = (
                f"Group comparison detected a large effect size (Cohen's d = {grp_res.statistics.get('effect_size_cohens_d')}). "
                f"Supplementary correlation between response time and error rate is recommended."
            )

    if follow_up_step and follow_up_rationale:
        follow_up_plan = AnalysisPlan(
            plan_id=f"plan_supp_{uuid.uuid4().hex[:6]}",
            request_id=state["request"].request_id,
            summary=f"Supplementary Follow-Up: {follow_up_rationale}",
            steps=[follow_up_step],
        )

        return {
            "plan": follow_up_plan,
            "follow_up_proposed": True,
            "follow_up_rationale": follow_up_rationale,
            "approval_status": "pending",
            "supplementary_rounds": supp_rounds + 1,
            "status": "follow_up_proposed",
        }

    return {
        "follow_up_proposed": False,
        "status": "evaluation_completed",
    }


# --- 7. Synthesize Results Node ---


def synthesize_results_node(state: AgentState) -> dict[str, Any]:
    """Produce a narrative summary grounded strictly in the returned tool outputs."""
    results = state.get("results", {})
    session_id = state.get("session_id", "sess")
    req = state.get("request")
    records = state.get("execution_records", [])

    synthesis_lines = [
        f"### Research Analysis Synthesis",
        f"**Research Request:** {req.query if req else 'Analysis Run'}",
        f"**Executed Operations:** {len(records)} tool steps completed across {state.get('supplementary_rounds', 0) + 1} round(s).",
        "",
    ]

    all_limitations: list[str] = [
        "All analytical findings are derived from deterministic placeholder tools and synthetic study data."
    ]

    for step_id, res in results.items():
        synthesis_lines.append(f"#### {res.analysis_name} (Run ID: {res.analysis_id})")
        synthesis_lines.append(f"- **Interpretation:** {res.interpretation}")
        if res.statistics:
            highlight_keys = [k for k in res.statistics.keys() if k not in ["trials", "values", "cluster_profiles", "ranking"]][:4]
            highlights = ", ".join(f"{k}: {res.statistics[k]}" for k in highlight_keys)
            if highlights:
                synthesis_lines.append(f"- **Key Statistics:** {highlights}")
        if res.warnings:
            synthesis_lines.append(f"- **Warnings:** {'; '.join(res.warnings)}")
        synthesis_lines.append("")

        for lim in res.limitations:
            if lim not in all_limitations:
                all_limitations.append(lim)

    report = FinalReport(
        report_id=f"rep_{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        synthesis="\n".join(synthesis_lines),
        limitations=all_limitations,
    )

    return {
        "report": report,
        "status": "completed",
    }


# --- 8. Persist Session Node ---


def persist_session_node(state: AgentState) -> dict[str, Any]:
    """Audit and save the full research session to SQLite persistence."""
    repo = get_session_repository()

    req = state.get("request")
    session_id = state.get("session_id") or f"sess_{uuid.uuid4().hex[:8]}"

    session_obj = ResearchSession(
        session_id=session_id,
        request=req,
        plan=state.get("plan"),
        approval=state.get("approval_decision"),
        execution_records=state.get("execution_records", []),
        final_report=state.get("report"),
    )

    repo.save_session(
        session=session_obj,
        model_provider=state.get("model_provider"),
        model_name=state.get("model_name"),
        results=state.get("results"),
        status=state.get("status", "completed"),
    )

    return {
        "status": "session_persisted",
    }
