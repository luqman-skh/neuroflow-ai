"""Pydantic schemas and contracts for NeuroFlow AI."""

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
from neuroflow.schemas.experiment import (
    ExperimentContext,
    VariableDefinition,
)
from neuroflow.schemas.planning import (
    AnalysisPlan,
    AnalysisStep,
    ResearchRequest,
)
from neuroflow.schemas.session import (
    ApprovalDecision,
    ExecutionRecord,
    FinalReport,
    ResearchSession,
)

__all__ = [
    "AnalysisResult",
    "ClusterBehaviourParams",
    "DetectAnomaliesParams",
    "GroupComparisonParams",
    "LearningCurveParams",
    "MemoryPerformanceParams",
    "PerformanceCorrelationsParams",
    "PredictPerformanceParams",
    "ResponseTimeParams",
    "TaskDifficultyParams",
    "ValidateExperimentDataParams",
    "ExperimentContext",
    "VariableDefinition",
    "AnalysisPlan",
    "AnalysisStep",
    "ResearchRequest",
    "ApprovalDecision",
    "ExecutionRecord",
    "FinalReport",
    "ResearchSession",
]
