"""Unit and smoke tests for the NeuroFlow Gradio user interface."""

import pytest
import gradio as gr
import plotly.graph_objects as go

from neuroflow.agent.graph import create_research_graph, run_research_session
from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep, ClarificationNeeded
from neuroflow.schemas.session import ExecutionRecord, FinalReport
from neuroflow.ui.app import create_app
from neuroflow.ui.components import (
    convert_chart_data_to_figure,
    format_clarification_markdown,
    format_execution_log,
    format_plan_markdown,
    format_report_markdown,
    format_results_markdown,
)


class TestUIComponents:
    """Tests for UI formatting and chart conversion helper functions."""

    def test_convert_chart_data_to_figure(self) -> None:
        chart_data = {
            "data": [{"type": "bar", "x": ["Group A", "Group B"], "y": [0.75, 0.88]}],
            "layout": {"title": "Accuracy Comparison"},
        }
        fig = convert_chart_data_to_figure(chart_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"

    def test_convert_chart_data_none(self) -> None:
        assert convert_chart_data_to_figure(None) is None
        assert convert_chart_data_to_figure({}) is None

    def test_format_plan_markdown(self) -> None:
        plan = AnalysisPlan(
            plan_id="plan_test_01",
            request_id="req_01",
            summary="Test study plan",
            steps=[
                AnalysisStep(
                    step_id="step_1",
                    tool_name="validate_experiment_data",
                    purpose="Audit data quality",
                    parameters={"task_type": "all"},
                    rationale="Prerequisite data check",
                )
            ],
        )
        md = format_plan_markdown(plan)
        assert "plan_test_01" in md
        assert "validate_experiment_data" in md
        assert "Audit data quality" in md

    def test_format_clarification_markdown(self) -> None:
        clar = ClarificationNeeded(
            needs_clarification=True,
            question="Which cognitive metric should be evaluated?",
            options=["Memory Accuracy", "Response Time Latency"],
            missing_variables=["blood_pressure"],
            suggested_alternatives=["response_time_ms"],
        )
        md = format_clarification_markdown(clar)
        assert "Clarification Needed" in md
        assert "Which cognitive metric" in md
        assert "blood_pressure" in md
        assert "response_time_ms" in md

    def test_format_execution_log(self) -> None:
        rec = ExecutionRecord(
            record_id="rec_01",
            step_id="step_1",
            tool_name="validate_experiment_data",
            parameters_passed={"task_type": "all"},
            status="success",
        )
        md = format_execution_log([rec])
        assert "step_1" in md
        assert "validate_experiment_data" in md
        assert "SUCCESS" in md


class TestUIAppSmoke:
    """Smoke tests for Gradio app creation, blocks assembly, and workflow execution."""

    def test_create_app_structure(self) -> None:
        app = create_app()
        assert isinstance(app, gr.Blocks)
        assert app.title == "NeuroFlow AI - Research Data Analysis Platform"

    def test_app_workflow_handlers(self) -> None:
        """Verify the submission and approval flow handlers behave correctly."""
        # 1. Run submission through underlying graph and formatters
        graph = create_research_graph()
        thread_id = "ui_smoke_session_001"
        query = "Compare memory accuracy between the control and experimental groups."

        res_state = run_research_session(graph=graph, query=query, thread_id=thread_id)
        assert res_state["status"] == "plan_generated" or res_state["approval_status"] == "pending"

        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert snapshot.next == ("human_approval",)
        plan = snapshot.values["plan"]

        plan_md = format_plan_markdown(plan)
        assert plan.plan_id in plan_md

        # 2. Verify results formatting when approved
        from neuroflow.agent.graph import resume_research_session
        from neuroflow.schemas.session import ApprovalDecision

        decision = ApprovalDecision(plan_id=plan.plan_id, approved=True)
        final_state = resume_research_session(graph=graph, thread_id=thread_id, decision=decision)

        assert final_state["approval_status"] == "approved"
        assert len(final_state["results"]) > 0

        res_md = format_results_markdown(final_state["results"])
        assert "Step Output" in res_md

        report_md = format_report_markdown(final_state["report"])
        assert "Evidence-Grounded Scientific Synthesis" in report_md
        assert "DEMO NOTICE" in report_md
