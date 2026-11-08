"""Graph nodes implementing the NeuroFlow analysis lifecycle."""

import uuid
from datetime import datetime, timezone
from typing import Any
from langgraph.types import interrupt

from neuroflow.agent.planner import AnalysisPlanner
from neuroflow.agent.state import AgentState
from neuroflow.agent.validator import validate_analysis_plan
from neuroflow.data.experiment_context import get_default_experiment_context
from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.experiment import ExperimentContext
from neuroflow.schemas.planning import AnalysisPlan, ResearchRequest
from neuroflow.schemas.session import ApprovalDecision, ExecutionRecord, FinalReport
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

    # Yield control to the researcher via LangGraph interrupt
    interrupt_payload = {
        "type": "human_approval_required",
        "session_id": session_id,
        "plan": plan.model_dump() if plan else None,
        "summary": plan.summary if plan else "",
        "message": (
            "An analysis plan has been proposed and validated. "
            "Researcher approval is strictly required before any tool executes."
        ),
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
                "status": "plan_modified",
            }
        return {
            "approval_decision": decision,
            "approval_status": "approved",
            "status": "approved",
        }

    return {
        "approval_decision": decision,
        "approval_status": "rejected",
        "feedback": decision.feedback,
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


# --- 6. Evaluate Execution Results Node ---


def evaluate_execution_results_node(state: AgentState) -> dict[str, Any]:
    """Inspect execution statuses, required outputs, and error states."""
    records = state.get("execution_records", [])
    results = state.get("results", {})

    total_steps = len(records)
    failed_steps = [r for r in records if r.status == "error"]
    warning_steps = [r for r in records if r.status == "warning"]

    if failed_steps:
        status = "evaluation_completed_with_errors"
    elif warning_steps:
        status = "evaluation_completed_with_warnings"
    else:
        status = "evaluation_completed"

    return {
        "status": status,
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
        f"**Executed Operations:** {len(records)} tool steps completed.",
        "",
    ]

    all_limitations: list[str] = [
        "All analytical findings are derived from deterministic placeholder tools and synthetic study data."
    ]

    for step_id, res in results.items():
        synthesis_lines.append(f"#### {res.analysis_name} (Run ID: {res.analysis_id})")
        synthesis_lines.append(f"- **Interpretation:** {res.interpretation}")
        if res.statistics:
            # Pick a few key statistics to highlight
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
    """Audit and finalize session history."""
    return {
        "status": "session_persisted",
    }
