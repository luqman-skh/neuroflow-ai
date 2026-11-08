"""Typed state definitions for the NeuroFlow LangGraph workflow."""

from typing import Annotated, Any, Literal, TypedDict
from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.planning import AnalysisPlan, ClarificationNeeded, ResearchRequest
from neuroflow.schemas.session import ApprovalDecision, ExecutionRecord, FinalReport


def update_execution_records(
    existing: list[ExecutionRecord], new_records: list[ExecutionRecord]
) -> list[ExecutionRecord]:
    """Reducer combining existing execution records with newly completed ones."""
    return existing + new_records


class AgentState(TypedDict, total=False):
    """Primary state container passed through the LangGraph analysis graph."""

    session_id: str
    request: ResearchRequest
    plan: AnalysisPlan | None
    clarification: ClarificationNeeded | None
    approval_status: Literal["pending", "approved", "rejected", "modified"]
    approval_decision: ApprovalDecision | None
    feedback: str | None
    validation_errors: list[str]
    execution_records: Annotated[list[ExecutionRecord], update_execution_records]
    results: dict[str, AnalysisResult]
    report: FinalReport | None
    status: str
    iteration_count: int
    max_iterations: int
