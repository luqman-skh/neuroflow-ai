"""Default demo experiment context and data dictionary for NeuroFlow AI."""

from neuroflow.schemas.experiment import ExperimentContext, VariableDefinition

DEMO_DATA_DICTIONARY: dict[str, VariableDefinition] = {
    "participant_id": VariableDefinition(
        name="participant_id",
        description="Pseudonymous participant identifier",
        data_type="string",
        example="P001",
    ),
    "group": VariableDefinition(
        name="group",
        description="Experimental condition or cohort assignment",
        data_type="string",
        example="control",
        allowed_values=["control", "experimental"],
    ),
    "task_id": VariableDefinition(
        name="task_id",
        description="Specific memory task trial identifier",
        data_type="string",
        example="memory_recall_01",
    ),
    "task_type": VariableDefinition(
        name="task_type",
        description="Task category",
        data_type="string",
        example="recall",
        allowed_values=["recall", "recognition", "spatial_memory"],
    ),
    "trial_number": VariableDefinition(
        name="trial_number",
        description="Sequential trial number within the session",
        data_type="integer",
        example=1,
    ),
    "response_correct": VariableDefinition(
        name="response_correct",
        description="Whether the participant response was scored correct",
        data_type="boolean",
        example=True,
    ),
    "response_time_ms": VariableDefinition(
        name="response_time_ms",
        description="Response latency in milliseconds from stimulus onset",
        data_type="float",
        example=1850.0,
    ),
    "attempt_count": VariableDefinition(
        name="attempt_count",
        description="Number of attempts or retries required for the trial",
        data_type="integer",
        example=2,
    ),
    "session_completed": VariableDefinition(
        name="session_completed",
        description="Indicator whether the full experimental session completed successfully",
        data_type="boolean",
        example=True,
    ),
    "event_timestamp": VariableDefinition(
        name="event_timestamp",
        description="ISO 8601 timestamp when the event was logged by the task client",
        data_type="datetime",
        example="2026-10-15T14:22:10Z",
    ),
    "session_id": VariableDefinition(
        name="session_id",
        description="Unique session identifier for an experimental run",
        data_type="string",
        example="S001",
    ),
}

DEFAULT_EXPERIMENT_CONTEXT = ExperimentContext(
    experiment_id="EXP-DZNE-MEM-001",
    title="VR/Desktop Memory & Executive Function Pilot",
    description=(
        "Synthetic pilot study of memory recall, recognition, and spatial navigation "
        "tasks comparing a control cohort to an experimental intervention cohort."
    ),
    groups=["control", "experimental"],
    task_types=["recall", "recognition", "spatial_memory"],
    variables=DEMO_DATA_DICTIONARY,
    total_participants=60,
    metadata={
        "study_lead": "NeuroFlow Research Team (Demo)",
        "protocol_version": "1.0-synthetic",
        "mock_environment": True,
        "sample_size_per_group": {"control": 30, "experimental": 30},
    },
)


def get_default_experiment_context() -> ExperimentContext:
    """Return a fresh copy of the default demo experiment context."""
    return DEFAULT_EXPERIMENT_CONTEXT.model_copy(deep=True)
