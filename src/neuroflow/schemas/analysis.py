"""Analysis result contracts and tool parameter schemas."""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class AnalysisResult(BaseModel):
    """Common envelope returned by all NeuroFlow analysis tools."""

    analysis_id: str = Field(..., description="Unique run identifier for this analysis execution")
    analysis_name: str = Field(..., description="Registered name of the analysis tool executed")
    status: Literal["success", "warning", "error"] = Field(
        default="success", description="Execution outcome status"
    )
    mock_data: bool = Field(
        default=True,
        description="Explicit flag indicating outputs are illustrative mock/synthetic results",
    )
    filters_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="Echo of filters and inputs actually applied during execution",
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Numerical and descriptive summary results (deterministic mock data)",
    )
    chart_data: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description="Plotly-compatible chart specification dictionary or trace data",
    )
    interpretation: str = Field(
        ...,
        description="Standardised descriptive interpretation of the output for researchers",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Explicit method and demo limitations to guide interpretation",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Data warnings, quality flags, or boundary notes",
    )
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the analysis completed",
    )


# --- 5.1 Participant performance parameter models ---


class MemoryPerformanceParams(BaseModel):
    """Parameters for analyze_memory_performance."""

    metric: Literal["accuracy", "recall_score", "completion_time_ms", "error_rate"] = Field(
        default="accuracy", description="Outcome performance metric"
    )
    task_type: Literal["all", "recall", "recognition", "spatial_memory"] = Field(
        default="all", description="Task filter"
    )
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental cohort filter"
    )
    participant_ids: list[str] | None = Field(
        default=None, description="Optional list of specific participant identifiers"
    )


class LearningCurveParams(BaseModel):
    """Parameters for analyze_learning_curve."""

    metric: Literal["accuracy", "response_time_ms", "error_rate"] = Field(
        default="accuracy", description="Outcome metric evaluated across trials"
    )
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental cohort filter"
    )
    trial_start: int = Field(default=1, ge=1, description="First trial sequence number")
    trial_end: int = Field(default=10, ge=1, description="Last trial sequence number")

    @model_validator(mode="after")
    def validate_trial_range(self) -> "LearningCurveParams":
        if self.trial_end < self.trial_start:
            raise ValueError(
                f"trial_end ({self.trial_end}) cannot be less than trial_start ({self.trial_start})"
            )
        return self


class ResponseTimeParams(BaseModel):
    """Parameters for analyze_response_time."""

    task_type: Literal["all", "recall", "recognition", "spatial_memory"] = Field(
        default="all", description="Task category filter"
    )
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental cohort filter"
    )
    participant_ids: list[str] | None = Field(
        default=None, description="Optional participant IDs filter"
    )
    trial_start: int | None = Field(
        default=None, ge=1, description="Optional starting trial filter"
    )
    trial_end: int | None = Field(
        default=None, ge=1, description="Optional ending trial filter"
    )

    @model_validator(mode="after")
    def validate_trials(self) -> "ResponseTimeParams":
        if self.trial_start is not None and self.trial_end is not None:
            if self.trial_end < self.trial_start:
                raise ValueError(
                    f"trial_end ({self.trial_end}) cannot be less than trial_start ({self.trial_start})"
                )
        return self


# --- 5.2 Group and task analysis parameter models ---


class GroupComparisonParams(BaseModel):
    """Parameters for compare_experimental_groups."""

    metric: Literal["accuracy", "response_time_ms", "recall_score", "error_rate"] = Field(
        default="accuracy", description="Outcome metric compared between groups"
    )
    group_a: str = Field(default="control", description="First group identifier")
    group_b: str = Field(default="experimental", description="Second group identifier")
    task_type: Literal["all", "recall", "recognition", "spatial_memory"] = Field(
        default="all", description="Task filter"
    )
    participant_ids: list[str] | None = Field(
        default=None, description="Optional list of participant IDs"
    )

    @model_validator(mode="after")
    def validate_distinct_groups(self) -> "GroupComparisonParams":
        if self.group_a.strip().lower() == self.group_b.strip().lower():
            raise ValueError("group_a and group_b must be distinct experimental groups")
        return self


class PerformanceCorrelationsParams(BaseModel):
    """Parameters for analyze_performance_correlations."""

    x_variable: Literal[
        "response_time_ms", "accuracy", "error_rate", "attempt_count", "trial_number"
    ] = Field(..., description="First variable for correlation analysis")
    y_variable: Literal[
        "response_time_ms", "accuracy", "error_rate", "attempt_count", "trial_number"
    ] = Field(..., description="Second variable for correlation analysis")
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental cohort filter"
    )
    task_type: Literal["all", "recall", "recognition", "spatial_memory"] = Field(
        default="all", description="Task category filter"
    )
    correlation_method: Literal["pearson", "spearman"] = Field(
        default="pearson", description="Correlation method"
    )

    @model_validator(mode="after")
    def validate_distinct_variables(self) -> "PerformanceCorrelationsParams":
        if self.x_variable == self.y_variable:
            raise ValueError("x_variable and y_variable must be distinct variables")
        return self


class TaskDifficultyParams(BaseModel):
    """Parameters for analyze_task_difficulty."""

    metric: Literal["accuracy", "response_time_ms", "error_rate", "completion_rate"] = Field(
        default="accuracy", description="Metric used to rank difficulty"
    )
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental cohort filter"
    )
    sort_order: Literal["ascending", "descending"] = Field(
        default="ascending", description="Ordering of difficulty ranking"
    )


# --- 5.3 ML-based behavioural analysis parameter models ---


class ClusterBehaviourParams(BaseModel):
    """Parameters for cluster_participant_behaviour."""

    features: list[str] = Field(
        default_factory=lambda: ["response_time_ms", "error_rate", "accuracy"],
        description="List of feature variables used for clustering",
    )
    n_clusters: int = Field(
        default=3, ge=2, le=10, description="Number of clusters to partition into"
    )
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental cohort filter"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: list[str]) -> list[str]:
        if not features:
            raise ValueError("features list must contain at least one feature")
        valid_features = {
            "response_time_ms",
            "error_rate",
            "accuracy",
            "retry_frequency",
            "attempt_count",
            "completion_time",
        }
        for f in features:
            if f not in valid_features:
                raise ValueError(
                    f"Feature '{f}' is not in allowed behavioural features: {sorted(valid_features)}"
                )
        return features


class PredictPerformanceParams(BaseModel):
    """Parameters for predict_task_performance."""

    target: Literal["high_accuracy", "session_completed", "fast_response"] = Field(
        default="high_accuracy", description="Predefined outcome target classification"
    )
    features: list[str] = Field(
        default_factory=lambda: ["response_time_ms", "attempt_count", "trial_number"],
        description="Predictor features for the classification pipeline",
    )
    task_type: Literal["all", "recall", "recognition", "spatial_memory"] = Field(
        default="all", description="Task filter"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: list[str]) -> list[str]:
        if not features:
            raise ValueError("features list must contain at least one feature")
        return features


# --- 5.4 Data quality and monitoring parameter models ---


class ValidateExperimentDataParams(BaseModel):
    """Parameters for validate_experiment_data."""

    task_type: Literal["all", "recall", "recognition", "spatial_memory"] = Field(
        default="all", description="Task filter for quality audit"
    )
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental group filter"
    )
    participant_ids: list[str] | None = Field(
        default=None, description="Optional participant IDs filter"
    )
    session_ids: list[str] | None = Field(
        default=None, description="Optional session IDs filter"
    )


class DetectAnomaliesParams(BaseModel):
    """Parameters for detect_anomalous_sessions."""

    sensitivity: Literal["low", "medium", "high"] = Field(
        default="medium", description="Anomaly detection threshold sensitivity"
    )
    task_type: Literal["all", "recall", "recognition", "spatial_memory"] = Field(
        default="all", description="Task filter"
    )
    group: Literal["control", "experimental"] | None = Field(
        default=None, description="Optional experimental group filter"
    )
