"""LangGraph workflow assembly with human-in-the-loop approval gate and checkpointing."""

import uuid
from typing import Any
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from neuroflow.agent.nodes import (
    NodeContext,
    evaluate_execution_results_node,
    execute_analysis_plan_node,
    generate_analysis_plan_node,
    human_approval_node,
    persist_session_node,
    set_node_context,
    synthesize_results_node,
    understand_request_node,
    validate_analysis_plan_node,
)
from neuroflow.agent.routing import (
    route_after_approval,
    route_after_planning,
    route_after_validation,
)
from neuroflow.agent.state import AgentState
from neuroflow.schemas.session import ApprovalDecision


def create_research_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    node_context: NodeContext | None = None,
) -> CompiledStateGraph:
    """Construct and compile the primary research analysis workflow graph."""
    if node_context:
        set_node_context(node_context)

    builder = StateGraph(AgentState)

    # 1. Add all workflow nodes
    builder.add_node("understand_request", understand_request_node)
    builder.add_node("generate_analysis_plan", generate_analysis_plan_node)
    builder.add_node("validate_analysis_plan", validate_analysis_plan_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_analysis_plan", execute_analysis_plan_node)
    builder.add_node("evaluate_execution_results", evaluate_execution_results_node)
    builder.add_node("synthesize_results", synthesize_results_node)
    builder.add_node("persist_session", persist_session_node)

    # 2. Add edges and conditional routing
    builder.add_edge(START, "understand_request")
    builder.add_edge("understand_request", "generate_analysis_plan")

    builder.add_conditional_edges(
        "generate_analysis_plan",
        route_after_planning,
        {
            "validate_analysis_plan": "validate_analysis_plan",
            "generate_analysis_plan": "generate_analysis_plan",
            "__end__": END,
        },
    )

    builder.add_conditional_edges(
        "validate_analysis_plan",
        route_after_validation,
        {
            "human_approval": "human_approval",
            "generate_analysis_plan": "generate_analysis_plan",
            "__end__": END,
        },
    )

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "validate_analysis_plan": "validate_analysis_plan",
            "execute_analysis_plan": "execute_analysis_plan",
            "generate_analysis_plan": "generate_analysis_plan",
            "__end__": END,
        },
    )

    builder.add_edge("execute_analysis_plan", "evaluate_execution_results")
    builder.add_edge("evaluate_execution_results", "synthesize_results")
    builder.add_edge("synthesize_results", "persist_session")
    builder.add_edge("persist_session", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)


def run_research_session(
    graph: CompiledStateGraph,
    query: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Initiate a research session thread on the compiled graph until approval pause or completion."""
    tid = thread_id or f"session_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid}}

    initial_state: AgentState = {
        "session_id": tid,
        "request": query,  # type: ignore
        "approval_status": "pending",
        "iteration_count": 0,
        "max_iterations": 3,
    }

    result = graph.invoke(initial_state, config=config)
    return result


def resume_research_session(
    graph: CompiledStateGraph,
    thread_id: str,
    decision: ApprovalDecision | dict[str, Any] | bool,
) -> dict[str, Any]:
    """Resume an interrupted research session thread with the researcher's approval decision."""
    config = {"configurable": {"thread_id": thread_id}}
    command = Command(resume=decision)
    result = graph.invoke(command, config=config)
    return result
