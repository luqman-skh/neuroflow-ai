"""Integration tests for the LangGraph stateful workflow and approval interrupt gate."""

import sqlite3
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from neuroflow.agent.graph import (
    create_research_graph,
    resume_research_session,
    run_research_session,
)
from neuroflow.agent.nodes import execute_analysis_plan_node
from neuroflow.schemas.planning import AnalysisStep
from neuroflow.schemas.session import ApprovalDecision


class TestGraphApprovalWorkflow:
    """Tests for graph execution, interrupt pause, approval, and execution gates."""

    def test_graph_pauses_before_tool_execution(self) -> None:
        """Verify the graph pauses at the approval interrupt and no tools execute prematurely."""
        graph = create_research_graph(checkpointer=MemorySaver())
        thread_id = "test_thread_pause"

        # Start research session
        state = run_research_session(
            graph=graph,
            query="Compare memory accuracy between control and experimental groups.",
            thread_id=thread_id,
        )

        # Graph should pause at human_approval
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert len(snapshot.tasks) > 0
        assert snapshot.next == ("human_approval",)

        # Verify plan was generated but NO tools have executed
        assert snapshot.values.get("plan") is not None
        assert snapshot.values.get("results") == {}
        assert len(snapshot.values.get("execution_records", [])) == 0

    def test_no_tool_runs_without_explicit_researcher_approval(self) -> None:
        """Verify that execute_analysis_plan_node strictly raises PermissionError without approval."""
        unapproved_state = {
            "session_id": "unapproved_session",
            "approval_status": "pending",
            "plan": None,
        }

        with pytest.raises(PermissionError) as exc_info:
            execute_analysis_plan_node(unapproved_state)  # type: ignore

        assert "human researcher approval must be granted" in str(exc_info.value)

    def test_approval_resumes_same_thread_and_executes_approved_steps(self) -> None:
        """Verify resuming with approval executes approved steps and generates the final synthesis."""
        graph = create_research_graph(checkpointer=MemorySaver())
        thread_id = "test_thread_resume"

        # 1. Initiate session
        run_research_session(
            graph=graph,
            query="Compare memory accuracy between control and experimental groups.",
            thread_id=thread_id,
        )

        # Check pause
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert snapshot.next == ("human_approval",)
        proposed_plan = snapshot.values["plan"]

        # 2. Resume with approval
        decision = ApprovalDecision(
            plan_id=proposed_plan.plan_id,
            approved=True,
            feedback="Plan approved by lead researcher.",
        )
        final_state = resume_research_session(graph=graph, thread_id=thread_id, decision=decision)

        # 3. Verify successful completion
        assert final_state["approval_status"] == "approved"
        assert final_state["status"] == "session_persisted"
        assert len(final_state["results"]) > 0
        assert len(final_state["execution_records"]) > 0
        assert final_state["report"] is not None
        assert "Research Analysis Synthesis" in final_state["report"].synthesis
        assert "DEMO NOTICE" in final_state["report"].mock_data_notice

    def test_editing_plan_causes_revalidation(self) -> None:
        """Verify that editing a plan causes revalidation before any execution occurs."""
        graph = create_research_graph(checkpointer=MemorySaver())
        thread_id = "test_thread_edit"

        run_research_session(
            graph=graph,
            query="Examine learning progression over trials.",
            thread_id=thread_id,
        )

        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        proposed_plan = snapshot.values["plan"]

        # Researcher edits the plan to add an INVALID step (invalid parameter trial_end < trial_start)
        invalid_step = AnalysisStep(
            step_id="step_custom_1",
            tool_name="analyze_learning_curve",
            purpose="Invalid edited step",
            parameters={"metric": "accuracy", "trial_start": 10, "trial_end": 2},
            rationale="Test edited revalidation failure",
        )

        decision = ApprovalDecision(
            plan_id=proposed_plan.plan_id,
            approved=True,
            modified_steps=[invalid_step],
        )

        # Resuming should route back through validate_analysis_plan, detect error, and fail or replan
        resumed_state = resume_research_session(graph=graph, thread_id=thread_id, decision=decision)
        # Verify invalid parameters were caught during revalidation
        assert len(resumed_state.get("validation_errors", [])) > 0 or resumed_state.get("status") in [
            "invalid_plan",
            "plan_generated",
            "awaiting_approval",
        ]

    def test_rejection_returns_to_planning_with_feedback(self) -> None:
        """Verify that rejecting a plan routes back to planning and incorporates researcher feedback."""
        graph = create_research_graph(checkpointer=MemorySaver())
        thread_id = "test_thread_reject"

        run_research_session(
            graph=graph,
            query="Compare groups.",
            thread_id=thread_id,
        )

        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        proposed_plan = snapshot.values["plan"]

        # Reject with specific feedback to examine learning curve instead
        decision = ApprovalDecision(
            plan_id=proposed_plan.plan_id,
            approved=False,
            feedback="Do not compare groups yet; investigate learning progression over trials instead.",
        )

        resumed_state = resume_research_session(graph=graph, thread_id=thread_id, decision=decision)

        # Graph should route back to planning, incorporate feedback, re-validate, and pause again for approval
        new_snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert new_snapshot.next == ("human_approval",)
        new_plan = new_snapshot.values["plan"]

        # The new plan should now include analyze_learning_curve
        tool_names = [s.tool_name for s in new_plan.steps]
        assert "analyze_learning_curve" in tool_names

    def test_sqlite_checkpointer_persistence(self) -> None:
        """Verify checkpointing with SQLite checkpointer preserves thread state."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph = create_research_graph(checkpointer=checkpointer)
        thread_id = "sqlite_persistent_thread"

        # 1. Run until pause
        run_research_session(
            graph=graph,
            query="Investigate whether response time is associated with error rate.",
            thread_id=thread_id,
        )

        # Verify state exists in SQLite checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        assert snapshot.next == ("human_approval",)
        assert "analyze_performance_correlations" in [s.tool_name for s in snapshot.values["plan"].steps]

        # 2. Resume on same thread with SQLite checkpointer
        decision = ApprovalDecision(
            plan_id=snapshot.values["plan"].plan_id,
            approved=True,
        )
        final_state = resume_research_session(graph=graph, thread_id=thread_id, decision=decision)

        assert final_state["approval_status"] == "approved"
        assert "step_2" in final_state["results"] or len(final_state["results"]) >= 2
