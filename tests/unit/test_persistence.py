"""Unit tests for SQLite session persistence and repository."""

import sqlite3
from datetime import datetime, timezone
import pytest

from neuroflow.persistence.database import init_db
from neuroflow.persistence.sessions import SessionRepository
from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep, ResearchRequest
from neuroflow.schemas.session import (
    ApprovalDecision,
    ExecutionRecord,
    FinalReport,
    ResearchSession,
)


@pytest.fixture
def repo(tmp_path):
    """Fixture providing a SessionRepository backed by a temporary SQLite file."""
    db_file = tmp_path / "test_neuroflow.db"
    return SessionRepository(db_path=db_file)


def test_save_and_get_session(repo):
    """Verify storing and retrieving a complete research session."""
    req = ResearchRequest(
        request_id="req_001",
        query="Compare accuracy between control and experimental cohorts.",
    )
    plan = AnalysisPlan(
        plan_id="plan_001",
        request_id="req_001",
        summary="Test comparison plan",
        steps=[
            AnalysisStep(
                step_id="step_1",
                tool_name="validate_experiment_data",
                purpose="Pre-analysis validation",
                parameters={"task_type": "all"},
                rationale="Audit completeness",
            ),
            AnalysisStep(
                step_id="step_2",
                tool_name="compare_experimental_groups",
                purpose="Group comparison",
                parameters={"metric": "accuracy", "group_a": "control", "group_b": "experimental"},
                rationale="Hypothesis test",
                dependencies=["step_1"],
            ),
        ],
    )
    decision = ApprovalDecision(plan_id="plan_001", approved=True, feedback="Approved.")
    record = ExecutionRecord(
        record_id="exec_001",
        step_id="step_1",
        tool_name="validate_experiment_data",
        parameters_passed={"task_type": "all"},
        status="success",
    )
    report = FinalReport(
        report_id="rep_001",
        session_id="sess_001",
        synthesis="Test synthesis narrative.",
        limitations=["Synthetic data."],
    )

    session = ResearchSession(
        session_id="sess_001",
        request=req,
        plan=plan,
        approval=decision,
        execution_records=[record],
        final_report=report,
    )

    result_envelope = AnalysisResult(
        analysis_id="val_01",
        analysis_name="validate_experiment_data",
        status="success",
        mock_data=True,
        filters_applied={"task_type": "all"},
        statistics={"valid_records_count": 1800},
        interpretation="Data is valid.",
    )

    repo.save_session(
        session=session,
        model_provider="fake",
        model_name="fake-planner-model",
        results={"step_1": result_envelope},
        status="completed",
    )

    # Retrieve and verify
    loaded = repo.get_session("sess_001")
    assert loaded is not None
    assert loaded["session"].session_id == "sess_001"
    assert loaded["session"].request.query == "Compare accuracy between control and experimental cohorts."
    assert loaded["session"].plan.plan_id == "plan_001"
    assert len(loaded["session"].plan.steps) == 2
    assert loaded["session"].approval.approved is True
    assert len(loaded["session"].execution_records) == 1
    assert loaded["session"].final_report.report_id == "rep_001"
    assert loaded["model_provider"] == "fake"
    assert "step_1" in loaded["results"]
    assert loaded["results"]["step_1"].statistics["valid_records_count"] == 1800


def test_list_and_search_sessions(repo):
    """Verify session listing and keyword search."""
    for i, q in enumerate(
        ["Compare groups", "Examine learning curves", "Search memory task latency"]
    ):
        req = ResearchRequest(request_id=f"req_{i}", query=q)
        sess = ResearchSession(session_id=f"sess_{i}", request=req)
        repo.save_session(sess, status="completed")

    summaries = repo.list_sessions(limit=10)
    assert len(summaries) == 3

    search_hits = repo.search_sessions("learning")
    assert len(search_hits) == 1
    assert search_hits[0]["session_id"] == "sess_1"


def test_delete_session(repo):
    """Verify session deletion."""
    req = ResearchRequest(request_id="req_del", query="Delete test")
    sess = ResearchSession(session_id="sess_del", request=req)
    repo.save_session(sess)

    assert repo.get_session("sess_del") is not None
    deleted = repo.delete_session("sess_del")
    assert deleted is True
    assert repo.get_session("sess_del") is None
