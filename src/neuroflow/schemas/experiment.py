"""Experiment context and data dictionary schemas."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class VariableDefinition(BaseModel):
    """Schema for a single variable definition in the data dictionary."""

    name: str = Field(..., description="Variable name/column identifier")
    description: str = Field(..., description="Explanation of what this variable represents")
    data_type: Literal["string", "integer", "float", "boolean", "datetime"] = Field(
        ..., description="Data type of the variable"
    )
    example: Any = Field(..., description="Example value for illustration")
    allowed_values: list[str] | None = Field(
        default=None, description="Optional set of categorical allowed values"
    )


class ExperimentContext(BaseModel):
    """Schema representing the active experimental context and schema definition."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    title: str = Field(..., description="Human-readable title of the experiment")
    description: str = Field(..., description="Detailed description of the experiment")
    groups: list[str] = Field(
        default_factory=lambda: ["control", "experimental"],
        description="Available experimental condition groups",
    )
    task_types: list[str] = Field(
        default_factory=lambda: ["recall", "recognition", "spatial_memory"],
        description="Available task types/categories in the experiment",
    )
    variables: dict[str, VariableDefinition] = Field(
        default_factory=dict, description="Dictionary of available dataset variables"
    )
    total_participants: int = Field(default=60, ge=1, description="Total participant count")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional experiment metadata"
    )

    def has_variable(self, variable_name: str) -> bool:
        """Check if a variable exists in the experiment data dictionary."""
        return variable_name in self.variables

    def get_variable_names(self) -> list[str]:
        """Return a list of all variable names."""
        return list(self.variables.keys())

    def validate_group(self, group: str | None) -> bool:
        """Validate if a group is recognized in this experiment."""
        if group is None:
            return True
        return group in self.groups

    def validate_task_type(self, task_type: str | None) -> bool:
        """Validate if a task type is recognized in this experiment."""
        if task_type is None or task_type == "all":
            return True
        return task_type in self.task_types
