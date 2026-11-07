"""NeuroFlow analysis tools catalogue and registry."""

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
from neuroflow.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    create_default_registry,
    default_registry,
    get_tool_registry,
)

__all__ = [
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
    "ToolDefinition",
    "ToolRegistry",
    "create_default_registry",
    "default_registry",
    "get_tool_registry",
]
