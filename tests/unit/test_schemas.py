"""Unit tests for Pydantic data contracts, schemas, and experiment context."""

import json
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from neuroflow.data.experiment_context import (
    DEFAULT_EXPERIMENT_CONTEXT,
    DEMO_DATA_DICTIONARY,
    get_default_experiment_context,
)
from neuroflow.schemas.analysis import (
    AnalysisResult,
    ClusterBehaviourParams,
    DetectAnomaliesParams,
    GroupComparisonParams,
    LearningCurveParams,
    MemoryPerformanceParams,
    PerformanceCorrelationsParams,
    PredictPerformanceParams,
    ResponseTimeParams,
    TaskDifficultyParams,
    ValidateExperimentDataParams,
)
from neuroflow.schemas.experiment import ExperimentContext, VariableDefinition
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep, ResearchRequest
from neuroflow.schemas.session import (
    ApprovalDecision,
    ExecutionRecord,
    FinalReport,
    ResearchSession,
)


class TestExperimentContextSchemas:
    """Tests for ExperimentContext and VariableDefinition."""

    def test_variable_definition_valid(self) -> None:
        var = VariableDefinition(
            name="accuracy",
            description="Trial accuracy",
            data_type="float",
            example=0.85,
        )
        assert var.name == "accuracy"
        assert var.data_type == "float"

    def test_default_experiment_context(self) -> None:
        ctx = get_default_experiment_context()
        assert ctx.experiment_id == "EXP-DZNE-MEM-001"
        assert "control" in ctx.groups
        assert "experimental" in ctx.groups
        assert ctx.has_variable("participant_id")
        assert ctx.has_variable("response_time_ms")
        assert not ctx.has_variable("blood_pressure")  # unknown variable
        assert ctx.validate_group("control") is True
        assert ctx.validate_group("unknown_group") is False
        assert ctx.validate_task_type("recall") is True
        assert ctx.validate_task_type("all") is True
        assert ctx.validate_task_type("chess") is False

    def test_demo_data_dictionary_completeness(self) -> None:
        expected_fields = [
            "participant_id",
            "group",
            "task_id",
            "task_type",
            "trial_number",
            "response_correct",
            "response_time_ms",
            "attempt_count",
            "session_completed",
            "event_timestamp",
            "session_id",
        ]
        for field in expected_fields:
            assert field in DEMO_DATA_DICTIONARY
            assert DEMO_DATA_DICTIONARY[field].name == field


class TestAnalysisParameterSchemas:
    """Tests for tool parameter validation models."""

    def test_memory_performance_params(self) -> None:
        p = MemoryPerformanceParams(metric="recall_score", task_type="recall")
        assert p.metric == "recall_score"
        assert p.task_type == "recall"

    def test_learning_curve_params_validation(self) -> None:
        p = LearningCurveParams(trial_start=1, trial_end=5)
        assert p.trial_end == 5
        with pytest.raises(ValidationError):
            LearningCurveParams(trial_start=6, trial_end=2)

    def test_response_time_params_validation(self) -> None:
        p = ResponseTimeParams(trial_start=2, trial_end=4)
        assert p.trial_start == 2
        with pytest.raises(ValidationError):
            ResponseTimeParams(trial_start=5, trial_end=1)

    def test_group_comparison_params_validation(self) -> None:
        p = GroupComparisonParams(group_a="group_1", group_b="group_2")
        assert p.group_a == "group_1"
        with pytest.raises(ValidationError):
            GroupComparisonParams(group_a="control", group_b="control")

    def test_correlations_params_validation(self) -> None:
        p = PerformanceCorrelationsParams(
            x_variable="response_time_ms", y_variable="error_rate"
        )
        assert p.x_variable == "response_time_ms"
        with pytest.raises(ValidationError):
            PerformanceCorrelationsParams(
                x_variable="accuracy", y_variable="accuracy"
            )

    def test_cluster_behaviour_params_validation(self) -> None:
        p = ClusterBehaviourParams(n_clusters=5)
        assert p.n_clusters == 5
        with pytest.raises(ValidationError):
            ClusterBehaviourParams(n_clusters=12)

    def test_predict_performance_params_validation(self) -> None:
        p = PredictPerformanceParams(target="fast_response")
        assert p.target == "fast_response"
        with pytest.raises(ValidationError):
            PredictPerformanceParams(target="invalid_target")  # type: ignore


class TestPlanningAndSessionSchemas:
    """Tests for AnalysisPlan, ApprovalDecision, ExecutionRecord, and ResearchSession."""

    def test_research_request_creation(self) -> None:
        req = ResearchRequest(
            request_id="REQ-001",
            query="Compare accuracy between control and experimental groups",
        )
        assert req.query.startswith("Compare")
        assert req.experiment_id == "EXP-DZNE-MEM-001"

    def test_analysis_step_and_plan(self) -> None:
        step = AnalysisStep(
            step_id="step_1",
            tool_name="validate_experiment_data",
            purpose="Validate data before analysis",
            parameters={"task_type": "all"},
            rationale="Ensure dataset completeness",
            dependencies=[],
        )
        plan = AnalysisPlan(
            plan_id="PLAN-001",
            request_id="REQ-001",
            summary="Two-step memory comparison",
            steps=[step],
        )
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "validate_experiment_data"

    def test_session_lifecycle_schemas(self) -> None:
        req = ResearchRequest(request_id="REQ-001", query="Inspect learning curve")
        plan = AnalysisPlan(
            plan_id="PLAN-001",
            request_id="REQ-001",
            summary="Run learning curve analysis",
            steps=[
                AnalysisStep(
                    step_id="s1",
                    tool_name="analyze_learning_curve",
                    purpose="Evaluate trial progression",
                    parameters={"trial_start": 1, "trial_end": 10},
                    rationale="Test for learning effect",
                )
            ],
        )
        decision = ApprovalDecision(
            plan_id="PLAN-001",
            approved=True,
            feedback="Looks good, proceed",
        )
        record = ExecutionRecord(
            record_id="REC-001",
            step_id="s1",
            tool_name="analyze_learning_curve",
            parameters_passed={"trial_start": 1, "trial_end": 10},
            status="success",
        )
        session = ResearchSession(
            session_id="SESS-001",
            request=req,
            plan=plan,
            approval=decision,
            execution_records=[record],
        )

        assert session.session_id == "SESS-001"
        assert session.approval.approved is True
        assert len(session.execution_records) == 1

        # Test full JSON serialization
        session_json = session.model_dump_json(indent=2)
        assert isinstance(session_json, str)
        parsed = json.loads(session_json)
        assert parsed["session_id"] == "SESS-001"
        assert parsed["approval"]["approved"] is True
