"""Tool registry mapping safe allowlisted tool names to functions, schemas, and metadata."""

from typing import Any, Callable, Type
from pydantic import BaseModel, Field

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
from neuroflow.tools.behavioural_ml import (
    cluster_participant_behaviour,
    predict_task_performance,
)
from neuroflow.tools.data_quality import (
    detect_anomalous_sessions,
    validate_experiment_data,
)
from neuroflow.tools.group_analysis import (
    analyze_performance_correlations,
    analyze_task_difficulty,
    compare_experimental_groups,
)
from neuroflow.tools.performance import (
    analyze_learning_curve,
    analyze_memory_performance,
    analyze_response_time,
)


class ToolDefinition(BaseModel):
    """Metadata and registration descriptor for an analysis tool."""

    name: str = Field(..., description="Unique allowlisted identifier for the tool")
    display_name: str = Field(..., description="Human-readable title for UI and reports")
    category: str = Field(..., description="Analytical category grouping")
    description: str = Field(..., description="Comprehensive description of the tool's purpose")
    function: Callable[..., AnalysisResult] = Field(
        ..., description="Callable Python function executing the tool"
    )
    parameter_schema: Type[BaseModel] = Field(
        ..., description="Pydantic model class validating input parameters"
    )
    is_pre_analysis: bool = Field(
        default=False,
        description="Flag indicating if this tool should run as a prerequisite data quality check",
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }


class ToolRegistry:
    """Registry maintaining allowlisted analytical tools and validating parameters."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool definition in the catalogue."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition:
        """Retrieve a registered tool by its unique name."""
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            raise KeyError(f"Tool '{name}' is not registered. Available tools: {available}")
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        """Check if a tool name is present in the registry."""
        return name in self._tools

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def list_tool_names(self) -> list[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def validate_parameters(self, tool_name: str, parameters: dict[str, Any]) -> BaseModel:
        """Validate a dictionary of parameters against the tool's Pydantic schema."""
        tool = self.get_tool(tool_name)
        return tool.parameter_schema(**parameters)

    def execute_tool(
        self, tool_name: str, parameters: dict[str, Any] | BaseModel | None = None
    ) -> AnalysisResult:
        """Execute a registered tool by name with validated parameters."""
        tool = self.get_tool(tool_name)
        if parameters is None:
            validated = tool.parameter_schema()
        elif isinstance(parameters, BaseModel):
            validated = parameters
        elif isinstance(parameters, dict):
            validated = tool.parameter_schema(**parameters)
        else:
            raise TypeError(f"Invalid parameters type: {type(parameters).__name__}")

        return tool.function(validated)


def create_default_registry() -> ToolRegistry:
    """Instantiate and populate the default catalogue of all ten deterministic tools."""
    reg = ToolRegistry()

    # 5.1 Participant performance
    reg.register(
        ToolDefinition(
            name="analyze_memory_performance",
            display_name="Memory Performance Analysis",
            category="Participant performance",
            description=(
                "Summarise participant memory-task performance, returning illustrative "
                "accuracy, recall score, completion time, and distribution summaries."
            ),
            function=analyze_memory_performance,
            parameter_schema=MemoryPerformanceParams,
        )
    )

    reg.register(
        ToolDefinition(
            name="analyze_learning_curve",
            display_name="Learning Curve Progression",
            category="Participant performance",
            description="Illustrate performance progression over repeated trials or sessions.",
            function=analyze_learning_curve,
            parameter_schema=LearningCurveParams,
        )
    )

    reg.register(
        ToolDefinition(
            name="analyze_response_time",
            display_name="Response Time Latency Analysis",
            category="Participant performance",
            description="Illustrate response-time distributions, quantiles, and latency variability.",
            function=analyze_response_time,
            parameter_schema=ResponseTimeParams,
        )
    )

    # 5.2 Group and task analysis
    reg.register(
        ToolDefinition(
            name="compare_experimental_groups",
            display_name="Experimental Group Comparison",
            category="Group and task analysis",
            description="Illustrate a comparison of an outcome between two experimental cohorts.",
            function=compare_experimental_groups,
            parameter_schema=GroupComparisonParams,
        )
    )

    reg.register(
        ToolDefinition(
            name="analyze_performance_correlations",
            display_name="Performance Correlations",
            category="Group and task analysis",
            description="Illustrate associations and correlations between performance variables.",
            function=analyze_performance_correlations,
            parameter_schema=PerformanceCorrelationsParams,
        )
    )

    reg.register(
        ToolDefinition(
            name="analyze_task_difficulty",
            display_name="Task Difficulty Ranking",
            category="Group and task analysis",
            description="Illustrate differences in difficulty and performance across memory tasks.",
            function=analyze_task_difficulty,
            parameter_schema=TaskDifficultyParams,
        )
    )

    # 5.3 ML-based behavioural analysis
    reg.register(
        ToolDefinition(
            name="cluster_participant_behaviour",
            display_name="Participant Behavioural Clustering",
            category="ML-based behavioural analysis",
            description=(
                "Return illustrative participant behavioural clusters based on features such as "
                "response time, error rate, retry frequency, and accuracy."
            ),
            function=cluster_participant_behaviour,
            parameter_schema=ClusterBehaviourParams,
        )
    )

    reg.register(
        ToolDefinition(
            name="predict_task_performance",
            display_name="Task Performance Prediction",
            category="ML-based behavioural analysis",
            description="Demonstrate the interface of a supervised prediction pipeline for a predefined outcome.",
            function=predict_task_performance,
            parameter_schema=PredictPerformanceParams,
        )
    )

    # 5.4 Data quality and monitoring
    reg.register(
        ToolDefinition(
            name="validate_experiment_data",
            display_name="Experiment Data Validation",
            category="Data quality and monitoring",
            description=(
                "Return a mock data-quality audit report. Prerequisite step in the analysis workflow."
            ),
            function=validate_experiment_data,
            parameter_schema=ValidateExperimentDataParams,
            is_pre_analysis=True,
        )
    )

    reg.register(
        ToolDefinition(
            name="detect_anomalous_sessions",
            display_name="Session Anomaly Detection",
            category="Data quality and monitoring",
            description="Return illustrative sessions flagged for researcher review.",
            function=detect_anomalous_sessions,
            parameter_schema=DetectAnomaliesParams,
        )
    )

    return reg


# Global default instance
default_registry = create_default_registry()


def get_tool_registry() -> ToolRegistry:
    """Return the global default tool registry."""
    return default_registry
