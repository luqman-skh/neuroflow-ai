"""Schemas for research requests, analysis planning, and plan steps."""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Initial research question and context submitted by a researcher."""

    request_id: str = Field(..., description="Unique request identifier")
    query: str = Field(..., description="Natural-language research question")
    researcher_id: str | None = Field(default=None, description="Optional researcher ID")
    experiment_id: str = Field(
        default="EXP-DZNE-MEM-001", description="Associated experiment ID"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when request was submitted",
    )


class AnalysisStep(BaseModel):
    """A single proposed analysis operation in an analysis plan."""

    step_id: str = Field(..., description="Unique identifier for this step, e.g. step_1")
    tool_name: str = Field(
        ..., description="Allowlisted tool name from the registered catalogue"
    )
    purpose: str = Field(
        ..., description="Short explanation of what this analytical step achieves"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of parameters and filters to pass to the tool",
    )
    rationale: str = Field(
        ..., description="Scientific or methodological justification for this step"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of prior step_ids that this step depends upon",
    )


class AnalysisPlan(BaseModel):
    """Complete structured analysis plan proposed by the system for researcher review."""

    plan_id: str = Field(..., description="Unique plan identifier")
    request_id: str = Field(..., description="Referenced research request ID")
    summary: str = Field(
        ..., description="High-level overview of the proposed analytical strategy"
    )
    steps: list[AnalysisStep] = Field(
        default_factory=list,
        description="Ordered sequence of analytical steps to be executed upon approval",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the plan was generated",
    )


class ClarificationNeeded(BaseModel):
    """Structured response when a user request is ambiguous, invalid, or requires clarification."""

    needs_clarification: bool = Field(
        default=True, description="Indicator that the planner requires more information"
    )
    question: str = Field(
        ..., description="Specific clarifying question posed to the researcher"
    )
    options: list[str] = Field(
        default_factory=list,
        description="Suggested choices or options the researcher can pick from",
    )
    missing_variables: list[str] = Field(
        default_factory=list,
        description="Variables requested that are not present in the experiment data dictionary",
    )
    unsupported_features: list[str] = Field(
        default_factory=list,
        description="Features or analyses requested that are outside the allowed tool catalogue",
    )
    suggested_alternatives: list[str] = Field(
        default_factory=list,
        description="Recommended supported tools or variables that can be used instead",
    )


class PlanningResult(BaseModel):
    """Top-level envelope returned by the AnalysisPlanner."""

    status: Literal["plan_generated", "clarification_needed", "rejected"] = Field(
        ..., description="Outcome of the planning attempt"
    )
    plan: AnalysisPlan | None = Field(
        default=None, description="The validated plan, if successfully generated"
    )
    clarification: ClarificationNeeded | None = Field(
        default=None, description="Clarification details if status is clarification_needed or rejected"
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Any schema or parameter validation errors encountered",
    )
    raw_model_response: str | None = Field(
        default=None, description="Raw output from the LLM for debugging and traceability"
    )
