"""Conditional routing logic for the NeuroFlow state graph."""

from typing import Literal
from neuroflow.agent.state import AgentState


def route_after_planning(
    state: AgentState,
) -> Literal["validate_analysis_plan", "generate_analysis_plan", "__end__"]:
    """Route after planning: check for clarification requests, replanning, or validation."""
    status = state.get("status")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if status == "clarification_needed":
        return "__end__"

    if status == "plan_rejected":
        if iteration >= max_iter:
            return "__end__"
        return "generate_analysis_plan"

    return "validate_analysis_plan"


def route_after_validation(
    state: AgentState,
) -> Literal["human_approval", "generate_analysis_plan", "__end__"]:
    """Route after validation: invalid plans trigger replanning; valid plans pause for approval."""
    status = state.get("status")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if status == "invalid_plan":
        if iteration >= max_iter:
            return "__end__"
        return "generate_analysis_plan"

    return "human_approval"


def route_after_approval(
    state: AgentState,
) -> Literal["validate_analysis_plan", "execute_analysis_plan", "generate_analysis_plan", "__end__"]:
    """Route after human approval: routes to execution, re-validation, or replanning based on researcher input."""
    status = state.get("approval_status")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if status == "modified":
        # Modified plans must be revalidated before execution
        return "validate_analysis_plan"

    if status == "approved":
        return "execute_analysis_plan"

    if status == "rejected":
        # If researcher provided feedback and iteration limit allows, return to planning
        if state.get("feedback") and iteration < max_iter:
            return "generate_analysis_plan"
        return "__end__"

    return "__end__"
