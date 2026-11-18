"""Unit tests for the structured analysis planner, LLM factory, and validator."""

import pytest
from langchain_core.messages import HumanMessage

from neuroflow.agent.llm import FakePlannerChatModel, OpenRouterChatModel, get_chat_model
from neuroflow.agent.planner import AnalysisPlanner
from neuroflow.agent.prompts import (
    build_planner_system_prompt,
    format_data_dictionary,
    format_tool_catalogue,
)
from neuroflow.agent.validator import validate_analysis_plan, validate_analysis_step
from neuroflow.config import Settings
from neuroflow.data.experiment_context import get_default_experiment_context
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep, ResearchRequest
from neuroflow.tools.registry import get_tool_registry


class TestLLMFactory:
    """Tests for LLM integration and provider configuration."""

    def test_get_chat_model_default_fake(self) -> None:
        cfg = Settings(LLM_PROVIDER="fake")
        model = get_chat_model(cfg)
        assert isinstance(model, FakePlannerChatModel)

    def test_fake_chat_model_invocation(self) -> None:
        model = FakePlannerChatModel()
        res = model.invoke([HumanMessage(content="Compare memory accuracy between control and experimental groups")])
        assert isinstance(res.content, str)
        assert "compare_experimental_groups" in res.content

    def test_unsupported_provider_raises_value_error(self) -> None:
        # Pydantic validates literal, but let's test invalid string if bypassed or constructed
        with pytest.raises(Exception):
            Settings(LLM_PROVIDER="unsupported_provider")  # type: ignore

    def test_openrouter_factory_and_missing_key(self) -> None:
        cfg = Settings(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="test-key-123")
        model = get_chat_model(cfg)
        assert isinstance(model, OpenRouterChatModel)
        assert model.api_key == "test-key-123"

        cfg_no_key = Settings(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            get_chat_model(cfg_no_key)

    def test_openrouter_invocation_with_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        mock_called = {}

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "id": "gen-1234",
                    "model": "google/gemma-4-26b-a4b-it:free",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"summary": "Test Plan", "steps": []}',
                                "reasoning_details": "Thinking step by step about memory tasks...",
                            }
                        }
                    ],
                    "usage": {"total_tokens": 42},
                }

        def mock_post(url, headers, data, timeout):
            import json
            mock_called["url"] = url
            mock_called["headers"] = headers
            mock_called["payload"] = json.loads(data)
            return MockResponse()

        import requests
        monkeypatch.setattr(requests, "post", mock_post)

        model = OpenRouterChatModel(
            model_name="google/gemma-4-26b-a4b-it:free",
            api_key="sk-test-key",
            enable_reasoning=True,
        )

        messages = [
            SystemMessage(content="You are an analysis planner."),
            HumanMessage(content="Compare memory accuracy between groups."),
            AIMessage(
                content="Preliminary check",
                additional_kwargs={"reasoning_details": "Initial thought chain"},
            ),
            HumanMessage(content="Proceed with full plan."),
        ]

        result = model.invoke(messages)

        assert mock_called["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert mock_called["headers"]["Authorization"] == "Bearer sk-test-key"
        assert mock_called["payload"]["model"] == "google/gemma-4-26b-a4b-it:free"
        assert mock_called["payload"]["reasoning"] == {"enabled": True}

        # Check preserved reasoning_details in assistant message passed back
        msg_list = mock_called["payload"]["messages"]
        assert len(msg_list) == 4
        assert msg_list[0] == {"role": "system", "content": "You are an analysis planner."}
        assert msg_list[2]["role"] == "assistant"
        assert msg_list[2]["content"] == "Preliminary check"
        assert msg_list[2]["reasoning_details"] == "Initial thought chain"

        # Check returned AIMessage preserves reasoning_details
        assert result.content == '{"summary": "Test Plan", "steps": []}'
        assert result.additional_kwargs.get("reasoning_details") == "Thinking step by step about memory tasks..."


class TestPlannerPrompts:
    """Tests for prompt builder and formatting."""

    def test_build_planner_system_prompt(self) -> None:
        ctx = get_default_experiment_context()
        reg = get_tool_registry()
        prompt = build_planner_system_prompt(ctx, reg)

        assert "EXP-DZNE-MEM-001" in prompt
        assert "response_time_ms" in prompt
        assert "analyze_memory_performance" in prompt
        assert "validate_experiment_data" in prompt
        assert "PLANNING RULES AND CONSTRAINTS" in prompt


class TestPlanValidator:
    """Tests for validate_analysis_plan and validate_analysis_step."""

    @pytest.fixture
    def setup_context(self):
        ctx = get_default_experiment_context()
        reg = get_tool_registry()
        return ctx, reg

    def test_valid_plan_passes_validation(self, setup_context) -> None:
        ctx, reg = setup_context
        plan = AnalysisPlan(
            plan_id="plan_1",
            request_id="req_1",
            summary="Valid plan",
            steps=[
                AnalysisStep(
                    step_id="step_1",
                    tool_name="validate_experiment_data",
                    purpose="Validate dataset records",
                    parameters={"task_type": "all"},
                    rationale="Check data quality before running statistical comparisons",
                    dependencies=[],
                ),
                AnalysisStep(
                    step_id="step_2",
                    tool_name="analyze_learning_curve",
                    purpose="Analyze learning progression",
                    parameters={"metric": "accuracy", "trial_start": 1, "trial_end": 10},
                    rationale="Observe cognitive improvement over successive trials",
                    dependencies=["step_1"],
                ),
            ],
        )

        is_valid, errors = validate_analysis_plan(plan, reg, ctx)
        assert is_valid is True
        assert len(errors) == 0

    def test_unknown_tool_rejected(self, setup_context) -> None:
        ctx, reg = setup_context
        plan = AnalysisPlan(
            plan_id="plan_1",
            request_id="req_1",
            summary="Invalid tool plan",
            steps=[
                AnalysisStep(
                    step_id="step_1",
                    tool_name="non_existent_tool",
                    purpose="Do something unauthorized",
                    parameters={},
                    rationale="Test failure",
                )
            ],
        )

        is_valid, errors = validate_analysis_plan(plan, reg, ctx)
        assert is_valid is False
        assert any("is not in the registered catalogue" in e for e in errors)

    def test_invalid_parameters_rejected(self, setup_context) -> None:
        ctx, reg = setup_context
        # trial_end < trial_start
        plan = AnalysisPlan(
            plan_id="plan_1",
            request_id="req_1",
            summary="Invalid parameters plan",
            steps=[
                AnalysisStep(
                    step_id="step_1",
                    tool_name="analyze_learning_curve",
                    purpose="Examine learning",
                    parameters={"metric": "accuracy", "trial_start": 10, "trial_end": 2},
                    rationale="Test parameter validator failure",
                )
            ],
        )

        is_valid, errors = validate_analysis_plan(plan, reg, ctx)
        assert is_valid is False
        assert any("parameter error" in e for e in errors)

    def test_invalid_dependency_rejected(self, setup_context) -> None:
        ctx, reg = setup_context
        # step_1 refers to future step_2
        plan = AnalysisPlan(
            plan_id="plan_1",
            request_id="req_1",
            summary="Forward reference dependency",
            steps=[
                AnalysisStep(
                    step_id="step_1",
                    tool_name="validate_experiment_data",
                    purpose="Validate dataset records",
                    parameters={"task_type": "all"},
                    rationale="Prerequisite data check",
                    dependencies=["step_2"],  # invalid forward ref
                ),
                AnalysisStep(
                    step_id="step_2",
                    tool_name="analyze_memory_performance",
                    purpose="Analyze performance",
                    parameters={"metric": "accuracy"},
                    rationale="Descriptive performance evaluation",
                    dependencies=[],
                ),
            ],
        )

        is_valid, errors = validate_analysis_plan(plan, reg, ctx)
        assert is_valid is False
        assert any("Dependency 'step_2' is invalid" in e for e in errors)

    def test_empty_plan_rejected(self, setup_context) -> None:
        ctx, reg = setup_context
        plan = AnalysisPlan(
            plan_id="plan_1",
            request_id="req_1",
            summary="Empty plan",
            steps=[],
        )
        is_valid, errors = validate_analysis_plan(plan, reg, ctx)
        assert is_valid is False
        assert "must contain at least one" in errors[0]


class TestAnalysisPlanner:
    """Tests for end-to-end AnalysisPlanner logic."""

    def test_planner_generates_valid_plan_without_execution(self) -> None:
        planner = AnalysisPlanner()
        query = "Compare memory accuracy between control and experimental groups"
        res = planner.plan(query)

        assert res.status == "plan_generated"
        assert res.plan is not None
        assert len(res.plan.steps) >= 2
        # Check required pre-analysis validation step
        assert res.plan.steps[0].tool_name == "validate_experiment_data"
        # Check that compare tool is in the steps
        tool_names = [s.tool_name for s in res.plan.steps]
        assert "compare_experimental_groups" in tool_names

        # Verify that all steps have rationales and purposes
        for step in res.plan.steps:
            assert len(step.purpose) > 0
            assert len(step.rationale) > 0

    def test_planner_clarification_on_ambiguous_request(self) -> None:
        planner = AnalysisPlanner()
        res = planner.plan("analyze participants")

        assert res.status == "clarification_needed"
        assert res.clarification is not None
        assert res.clarification.needs_clarification is True
        assert len(res.clarification.options) > 0

    def test_planner_handles_unknown_variables(self) -> None:
        planner = AnalysisPlanner()
        res = planner.plan("Compare blood_pressure and heart_rate between groups")

        assert res.status == "clarification_needed"
        assert res.clarification is not None
        assert len(res.clarification.missing_variables) > 0
        assert "blood_pressure" in res.clarification.missing_variables or "blood pressure" in res.clarification.missing_variables
        assert len(res.clarification.suggested_alternatives) > 0

    def test_planner_handles_unsupported_actions(self) -> None:
        planner = AnalysisPlanner()
        res = planner.plan("Train a deep neural network in PyTorch to classify participants")

        assert res.status == "clarification_needed"
        assert res.clarification is not None
        assert len(res.clarification.unsupported_features) > 0
        assert len(res.clarification.suggested_alternatives) > 0
