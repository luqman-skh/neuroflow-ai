"""Unit tests for the analysis tool registry."""

import pytest
from pydantic import ValidationError

from neuroflow.schemas.analysis import AnalysisResult, MemoryPerformanceParams
from neuroflow.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    create_default_registry,
    default_registry,
    get_tool_registry,
)


class TestToolRegistry:
    """Tests for ToolRegistry registration, validation, and execution."""

    EXPECTED_TEN_TOOLS = [
        "analyze_memory_performance",
        "analyze_learning_curve",
        "analyze_response_time",
        "compare_experimental_groups",
        "analyze_performance_correlations",
        "analyze_task_difficulty",
        "cluster_participant_behaviour",
        "predict_task_performance",
        "validate_experiment_data",
        "detect_anomalous_sessions",
    ]

    def test_default_registry_contains_ten_tools(self) -> None:
        reg = get_tool_registry()
        tool_names = reg.list_tool_names()
        assert len(tool_names) == 10
        for expected in self.EXPECTED_TEN_TOOLS:
            assert expected in tool_names
            assert reg.has_tool(expected) is True

    def test_tool_definition_properties(self) -> None:
        reg = get_tool_registry()
        for name in self.EXPECTED_TEN_TOOLS:
            tool = reg.get_tool(name)
            assert isinstance(tool, ToolDefinition)
            assert tool.name == name
            assert len(tool.display_name) > 0
            assert len(tool.category) > 0
            assert len(tool.description) > 0
            assert callable(tool.function)
            assert tool.parameter_schema is not None

    def test_pre_analysis_tool_flag(self) -> None:
        reg = get_tool_registry()
        val_tool = reg.get_tool("validate_experiment_data")
        assert val_tool.is_pre_analysis is True

        mem_tool = reg.get_tool("analyze_memory_performance")
        assert mem_tool.is_pre_analysis is False

    def test_unknown_tool_lookup(self) -> None:
        reg = get_tool_registry()
        assert reg.has_tool("non_existent_tool") is False
        with pytest.raises(KeyError) as exc_info:
            reg.get_tool("non_existent_tool")
        assert "is not registered" in str(exc_info.value)

    def test_validate_parameters(self) -> None:
        reg = get_tool_registry()
        valid_params = {"metric": "accuracy", "task_type": "recall"}
        validated = reg.validate_parameters("analyze_memory_performance", valid_params)
        assert isinstance(validated, MemoryPerformanceParams)
        assert validated.metric == "accuracy"

        with pytest.raises(ValidationError):
            reg.validate_parameters(
                "analyze_memory_performance", {"metric": "invalid_metric"}
            )

    def test_execute_tool_via_registry(self) -> None:
        reg = get_tool_registry()
        result = reg.execute_tool(
            "analyze_memory_performance", {"metric": "accuracy", "group": "control"}
        )
        assert isinstance(result, AnalysisResult)
        assert result.analysis_name == "analyze_memory_performance"
        assert result.mock_data is True
        assert result.filters_applied["group"] == "control"

    def test_execute_tool_with_pydantic_instance(self) -> None:
        reg = get_tool_registry()
        params = MemoryPerformanceParams(metric="recall_score", group="experimental")
        result = reg.execute_tool("analyze_memory_performance", params)
        assert isinstance(result, AnalysisResult)
        assert result.filters_applied["metric"] == "recall_score"

    def test_execute_unknown_tool_raises_key_error(self) -> None:
        reg = get_tool_registry()
        with pytest.raises(KeyError):
            reg.execute_tool("unknown_tool", {})

    def test_execute_invalid_param_type_raises_type_error(self) -> None:
        reg = get_tool_registry()
        with pytest.raises(TypeError):
            reg.execute_tool("analyze_memory_performance", "not_a_dict_or_basemodel")  # type: ignore
