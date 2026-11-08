"""End-to-end evaluation tests for the complete NeuroFlow AI workflow.

Verifies the full lifecycle: request -> plan -> human approval gate -> tool execution ->
synthesis report -> persistence storage.
"""

import pytest
import uuid
from neuroflow.agent.graph import create_research_graph, run_research_session, resume_research_session
from neuroflow.persistence.sessions import get_session_repository
from neuroflow.schemas.session import ApprovalDecision


class TestEndToEndPipeline:
    """Verifies that representative requests can complete through the full lifecycle."""

    @pytest.fixture
    def graph(self):
        return create_research_graph()

    def test_complete_demo_workflow_from_request_to_persisted_report(self, graph) -> None:
        """Full execution lifecycle for Case 1: Group memory comparison."""
        thread_id = f"e2e_test_{uuid.uuid4().hex[:8]}"
        query = "Compare memory accuracy between the control and experimental groups."

        # Step 1: Initial submission pauses at human approval
        initial_state = run_research_session(graph=graph, query=query, thread_id=thread_id)
        assert initial_state["status"] in ("plan_generated", "awaiting_approval")
        assert initial_state["approval_status"] == "pending"
        assert len(initial_state["results"]) == 0  # No execution yet

        # Verify LangGraph checkpoint is paused at interrupt gate
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert snapshot.next == ("human_approval",)
        plan = snapshot.values["plan"]
        assert plan is not None
        assert len(plan.steps) >= 2

        # Step 2: Human researcher approves the plan
        decision = ApprovalDecision(plan_id=plan.plan_id, approved=True, comments="Proceed with analysis")
        final_state = resume_research_session(graph=graph, thread_id=thread_id, decision=decision)

        # Step 3: Assert execution completed successfully
        assert final_state["status"] in ("completed", "session_persisted")
        assert final_state["approval_status"] == "approved"
        assert len(final_state["results"]) >= 2

        # Step 4: Verify evidence-grounded report and mock disclaimer
        report = final_state.get("report")
        assert report is not None
        assert report.report_id.startswith("rep_")
        assert "Evidence-Grounded Scientific Synthesis" in report.synthesis or "Research Analysis Synthesis" in report.synthesis
        assert "DEMO NOTICE" in report.mock_data_notice
        assert len(report.limitations) > 0

        # Step 5: Verify SQLite persistence
        repo = get_session_repository()
        persisted = repo.get_session(thread_id)
        assert persisted is not None
        assert persisted["session"].session_id == thread_id
        assert persisted["status"] == "completed"
        assert len(persisted["session"].execution_records) >= 2
        assert persisted["session"].final_report.report_id == report.report_id

    def test_rejection_and_modification_workflow(self, graph) -> None:
        """Verifies researcher rejection with feedback terminates or redirects cleanly."""
        thread_id = f"e2e_reject_{uuid.uuid4().hex[:8]}"
        query = "Examine learning progression over trials."

        # Submit request
        run_research_session(graph=graph, query=query, thread_id=thread_id)

        # Reject with feedback
        decision = ApprovalDecision(
            plan_id="any",
            approved=False,
            comments="Please do not proceed with learning curve analysis.",
        )
        res_state = resume_research_session(graph=graph, thread_id=thread_id, decision=decision)
        assert res_state["approval_status"] == "rejected"
        assert len(res_state["results"]) == 0  # No tools executed

    def test_ambiguous_query_lifecycle(self, graph) -> None:
        """Verifies ambiguous queries produce a clarification prompt without entering approval."""
        thread_id = f"e2e_clarify_{uuid.uuid4().hex[:8]}"
        query = "do some analysis"

        state = run_research_session(graph=graph, query=query, thread_id=thread_id)
        assert state["status"] == "clarification_needed"
        assert state.get("clarification") is not None
        assert state["clarification"].needs_clarification is True
        assert len(state["results"]) == 0
