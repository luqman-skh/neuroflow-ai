"""Plan validation against registered tools, schemas, and experiment context."""

from typing import Tuple
from pydantic import ValidationError

from neuroflow.schemas.experiment import ExperimentContext
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep
from neuroflow.tools.registry import ToolRegistry


def validate_analysis_step(
    step: AnalysisStep,
    registry: ToolRegistry,
    context: ExperimentContext,
    prior_step_ids: set[str],
) -> list[str]:
    """Validate a single analysis step against registry, schemas, and dependencies.
    
    Returns a list of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # 1. Check tool registration
    if not registry.has_tool(step.tool_name):
        errors.append(
            f"Step '{step.step_id}': Tool '{step.tool_name}' is not in the registered catalogue. "
            f"Available tools: {', '.join(registry.list_tool_names())}"
        )
        return errors  # Cannot validate parameters if tool is unknown

    # 2. Check parameter validity via tool parameter schema
    try:
        registry.validate_parameters(step.tool_name, step.parameters)
    except ValidationError as ve:
        for err in ve.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(
                f"Step '{step.step_id}' ({step.tool_name}) parameter error in '{loc}': {err['msg']}"
            )
    except Exception as e:
        errors.append(
            f"Step '{step.step_id}' ({step.tool_name}) parameter error: {str(e)}"
        )

    # 3. Check group / task_type consistency with experiment context if present in parameters
    group_val = step.parameters.get("group")
    if group_val and not context.validate_group(group_val):
        errors.append(
            f"Step '{step.step_id}': Group '{group_val}' is not recognized in experiment "
            f"'{context.experiment_id}'. Allowed groups: {context.groups}"
        )

    task_type_val = step.parameters.get("task_type")
    if task_type_val and not context.validate_task_type(task_type_val):
        errors.append(
            f"Step '{step.step_id}': Task type '{task_type_val}' is not recognized in experiment "
            f"'{context.experiment_id}'. Allowed task types: {context.task_types}"
        )

    # 4. Check dependencies (must refer only to prior step_ids)
    for dep in step.dependencies:
        if dep not in prior_step_ids:
            errors.append(
                f"Step '{step.step_id}': Dependency '{dep}' is invalid (must be an already declared prior step_id)."
            )

    # 5. Check purpose and rationale completeness
    if not step.purpose or len(step.purpose.strip()) < 5:
        errors.append(f"Step '{step.step_id}': Purpose must be clearly specified.")

    if not step.rationale or len(step.rationale.strip()) < 5:
        errors.append(f"Step '{step.step_id}': Rationale must be clearly specified.")

    return errors


def validate_analysis_plan(
    plan: AnalysisPlan,
    registry: ToolRegistry,
    context: ExperimentContext,
) -> Tuple[bool, list[str]]:
    """Validate a complete analysis plan.
    
    Returns (is_valid, list_of_error_strings).
    """
    all_errors: list[str] = []

    if not plan.steps:
        all_errors.append("Plan must contain at least one analysis step.")
        return False, all_errors

    seen_step_ids: set[str] = set()

    for idx, step in enumerate(plan.steps):
        # Unique step IDs
        if step.step_id in seen_step_ids:
            all_errors.append(f"Duplicate step_id found: '{step.step_id}'")
        
        # Step validation
        step_errors = validate_analysis_step(
            step=step,
            registry=registry,
            context=context,
            prior_step_ids=seen_step_ids,
        )
        all_errors.extend(step_errors)
        seen_step_ids.add(step.step_id)

    is_valid = len(all_errors) == 0
    return is_valid, all_errors
