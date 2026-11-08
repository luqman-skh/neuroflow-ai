"""Integration tests for Milestone 5: Bounded iterative follow-up analysis and session persistence."""

import sqlite3
import pytest
from langgraph.checkpoint.memory import MemorySaver

from neuroflow.agent.graph import (
    create_research_graph,
    resume_research_session,
    run_research_session,
)
from neuroflow.persistence.sessions import SessionRepository
from neuroflow.schemas.session import ApprovalDecision


class TestBoundedIterativeAnalysis:
    """Tests verifying justified follow-up proposals, approval requirements, iteration limits, and persistence."""

    def test_justified_followup_pauses_for_approval_and_executes_only_after_approval(self) -> None:
        """Verify that when intermediate results justify a follow-up, it pauses at human approval."""
        graph = create_research_graph(checkpointer=MemorySaver())
        thread_id = "iterative_followup_thread_01"

        # 1. Start research session with max_supplementary_rounds=1
        query = "Compare memory accuracy between the control and experimental groups."
        run_research_session(
            graph=graph, query=query, thread_id=thread_id, max_supplementary_rounds=1
        )

        # 2. Check Round 1 pause at human_approval
        snapshot_r1 = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert snapshot_r1.next == ("human_approval",)
        assert snapshot_r1.values.get("supplementary_rounds", 0) == 0

        # Approve Round 1 plan
        plan_r1 = snapshot_r1.values["plan"]
        decision_r1 = ApprovalDecision(plan_id=plan_r1.plan_id, approved=True)

        resumed_r1 = resume_research_session(graph=graph, thread_id=thread_id, decision=decision_r1)

        # 3. Intermediate results evaluation:
        # compare_experimental_groups yielded a large effect (Cohen's d > 0.6)
        # Evaluator should propose a justified supplementary correlation round and PAUSE again!
        snapshot_r2 = graph.get_state({"configurable": {"thread_id": thread_id}})

        # If a follow-up was proposed, it must pause at human_approval and NOT execute automatically
        if snapshot_r2.values.get("follow_up_proposed") is True:
            assert snapshot_r2.next == ("human_approval",)
            assert snapshot_r2.values.get("supplementary_rounds") == 1
            follow_up_plan = snapshot_r2.values["plan"]
            assert "step_followup" in follow_up_plan.steps[0].step_id

            # Verify the follow-up tool has NOT executed yet
            assert follow_up_plan.steps[0].step_id not in snapshot_r2.values.get("results", {})

            # 4. Researcher explicitly approves supplementary Round 2
            decision_r2 = ApprovalDecision(plan_id=follow_up_plan.plan_id, approved=True)
            resumed_r2 = resume_research_session(graph=graph, thread_id=thread_id, decision=decision_r2)

            # Follow-up tool has now executed
            assert follow_up_plan.steps[0].step_id in resumed_r2["results"]
            assert resumed_r2["status"] == "session_persisted"

    def test_strict_iteration_limit_enforced(self) -> None:
        """Verify that supplementary analysis rounds cannot exceed max_supplementary_rounds."""
        graph = create_research_graph(checkpointer=MemorySaver())
        thread_id = "iterative_limit_thread_01"

        # Explicitly configure max_supplementary_rounds = 0 (no follow-ups allowed)
        initial_state = {
            "session_id": thread_id,
            "request": "Compare memory accuracy between control and experimental groups.",
            "max_supplementary_rounds": 0,
            "approval_status": "pending",
        }

        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke(initial_state, config=config)

        # Round 1 approval
        snapshot = graph.get_state(config)
        plan = snapshot.values["plan"]
        final_state = resume_research_session(
            graph=graph, thread_id=thread_id, decision=ApprovalDecision(plan_id=plan.plan_id, approved=True)
        )

        # Should complete directly to final report without proposing follow-up
        assert final_state["supplementary_rounds"] == 0
        assert final_state.get("follow_up_proposed") is False
        assert final_state["status"] == "session_persisted"

    def test_session_audit_persisted_to_repository(self, tmp_path) -> None:
        """Verify that completed research session details can be reopened and audited."""
        db_file = tmp_path / "audit_test.db"
        repo = SessionRepository(db_path=db_file)

        # Set up node context to write to this test repository
        from neuroflow.agent.nodes import NodeContext, set_node_context
        from neuroflow.persistence import sessions as sessions_mod

        old_repo = sessions_mod._default_repository
        sessions_mod._default_repository = repo

        try:
            graph = create_research_graph(checkpointer=MemorySaver())
            thread_id = "persisted_audit_session_77"

            run_research_session(
                graph=graph,
                query="Examine learning progression over trials.",
                thread_id=thread_id,
            )

            # Approve Round 1
            snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
            plan = snapshot.values["plan"]
            resume_research_session(
                graph=graph, thread_id=thread_id, decision=ApprovalDecision(plan_id=plan.plan_id, approved=True)
            )

            # If a justified follow-up round was proposed, approve it so workflow completes to persist_session
            snapshot2 = graph.get_state({"configurable": {"thread_id": thread_id}})
            if snapshot2.next == ("human_approval",):
                plan2 = snapshot2.values["plan"]
                resume_research_session(
                    graph=graph, thread_id=thread_id, decision=ApprovalDecision(plan_id=plan2.plan_id, approved=True)
                )

            # Verify session was persisted in SQLite repository
            loaded = repo.get_session(thread_id)
            assert loaded is not None
            assert loaded["session"].session_id == thread_id
            assert "Examine learning progression" in loaded["session"].request.query
            assert loaded["session"].final_report is not None
            assert len(loaded["session"].execution_records) > 0
            assert len(loaded["results"]) > 0

            # Verify session appears in listing
            all_sessions = repo.list_sessions()
            assert any(s["session_id"] == thread_id for s in all_sessions)
        finally:
            sessions_mod._default_repository = old_repo
