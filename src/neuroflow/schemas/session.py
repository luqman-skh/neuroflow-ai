"""Schemas for session state, approval decisions, execution records, and reports."""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field

from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep, ResearchRequest


class ApprovalDecision(BaseModel):
    """Researcher human-in-the-loop decision regarding a proposed analysis plan."""

    plan_id: str = Field(..., description="ID of the plan being decided upon")
    approved: bool = Field(
        ..., description="True if researcher approved execution; False if rejected"
    )
    feedback: str | None = Field(
        default=None, description="Optional researcher feedback or revision instructions"
    )
    modified_steps: list[AnalysisStep] | None = Field(
        default=None,
        description="Optional researcher-edited list of steps to replace proposed plan",
    )
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of researcher action",
    )


class ExecutionRecord(BaseModel):
    """Audit record of a single executed tool step."""

    record_id: str = Field(..., description="Unique execution record identifier")
    step_id: str = Field(..., description="Associated step ID from the approved plan")
    tool_name: str = Field(..., description="Name of the executed tool")
    parameters_passed: dict[str, Any] = Field(
        default_factory=dict, description="Exact parameters passed to the tool"
    )
    status: Literal["success", "warning", "error"] = Field(
        ..., description="Execution status outcome"
    )
    result: AnalysisResult | None = Field(
        default=None, description="Returned structured AnalysisResult envelope"
    )
    error_message: str | None = Field(
        default=None, description="Error details if execution encountered an error"
    )
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when execution occurred",
    )


class FinalReport(BaseModel):
    """Grounded synthesis and scientific summary derived from executed tools."""

    report_id: str = Field(..., description="Unique report identifier")
    session_id: str = Field(..., description="Associated research session ID")
    synthesis: str = Field(
        ..., description="Evidence-grounded narrative summary based only on tool outputs"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Aggregated limitations across analyses"
    )
    mock_data_notice: str = Field(
        default=(
            "DEMO NOTICE: All analytical results, statistics, and charts are synthetic "
            "placeholders and do not represent real participant findings."
        ),
        description="Mandatory disclaimer for mock data output",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp",
    )


class ResearchSession(BaseModel):
    """Complete research session state suitable for persistence and auditing."""

    session_id: str = Field(..., description="Stable thread / session identifier")
    request: ResearchRequest = Field(..., description="Initial research request")
    plan: AnalysisPlan | None = Field(
        default=None, description="Proposed or approved analysis plan"
    )
    approval: ApprovalDecision | None = Field(
        default=None, description="Human approval decision record"
    )
    execution_records: list[ExecutionRecord] = Field(
        default_factory=list, description="Audit log of tool execution runs"
    )
    final_report: FinalReport | None = Field(
        default=None, description="Synthesised final report if completed"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
