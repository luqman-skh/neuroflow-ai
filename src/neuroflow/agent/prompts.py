"""System prompts, templates, and formatting for structured analysis planning."""

from typing import Any
from neuroflow.schemas.experiment import ExperimentContext
from neuroflow.tools.registry import ToolRegistry


PLANNER_SYSTEM_PROMPT_TEMPLATE = """You are NeuroFlow AI's expert Research Analysis Planner for experimental neuroscience and memory-task studies.

Your task is to translate a researcher's natural-language objective into a validated, structured analysis plan composed solely of registered Python analysis tools.

### ACTIVE EXPERIMENT CONTEXT:
- Experiment ID: {experiment_id}
- Study Title: {experiment_title}
- Description: {experiment_description}
- Available Cohort Groups: {available_groups}
- Available Task Categories: {available_task_types}
- Total Participant Count: {total_participants}

### DATA DICTIONARY (Available Dataset Variables):
{data_dictionary_formatted}

### REGISTERED ANALYSIS TOOL CATALOGUE:
{tool_catalogue_formatted}

### PLANNING RULES AND CONSTRAINTS:
1. ONLY select tools from the Registered Analysis Tool Catalogue above. Never invent tool names.
2. NEVER generate arbitrary Python or SQL code.
3. Reference ONLY variables explicitly listed in the Data Dictionary.
4. If the researcher's query refers to unknown variables (e.g., blood pressure, EEG, biological markers not in the dataset), DO NOT invent them. Flag that clarification is needed and list the missing variables with suggested alternatives.
5. If the request is too broad, ambiguous, or underspecified, produce a clarification request with multiple choice options.
6. If the request asks for unsupported actions (e.g. training deep neural nets, diagnostic evaluation, arbitrary scripts), refuse or clarify with registered alternatives.
7. DEFAULT WORKFLOW RULE: Unless the researcher explicitly specifies to skip pre-validation, ALWAYS include 'validate_experiment_data' as the first analysis step (step_1) to audit data quality.
8. Each step must include:
   - 'step_id': unique sequential identifier (e.g. 'step_1', 'step_2')
   - 'tool_name': allowlisted tool name
   - 'purpose': concise description of what this step evaluates
   - 'parameters': valid dictionary of parameter values adhering to the tool's schema
   - 'rationale': clear scientific or methodological justification
   - 'dependencies': list of prior step_ids that logically precede this step
9. The workflow must be minimal, relevant, and directly address the research question without unnecessary operations.
10. Do not invent participant counts or mock statistical numbers in the plan. The tools will return those upon execution.

### OUTPUT FORMAT:
You must output a single valid JSON object adhering to ONE of two formats:

Format A (When a valid plan can be proposed):
{{
  "type": "plan",
  "summary": "<High-level analytical summary of the strategy>",
  "steps": [
    {{
      "step_id": "step_1",
      "tool_name": "validate_experiment_data",
      "purpose": "<Purpose>",
      "parameters": {{...}},
      "rationale": "<Methodological rationale>",
      "dependencies": []
    }},
    ...
  ]
}}

Format B (When clarification is required or request is invalid):
{{
  "type": "clarification",
  "question": "<Specific clarifying question>",
  "options": ["<Option 1>", "<Option 2>", ...],
  "missing_variables": ["<unknown_var_1>", ...],
  "unsupported_features": ["<unsupported_feature_1>", ...],
  "suggested_alternatives": ["<tool_or_variable_1>", ...]
}}
"""


def format_data_dictionary(context: ExperimentContext) -> str:
    """Format the experiment context variables into a clean markdown table string."""
    lines = [
        "| Variable | Type | Allowed / Example | Description |",
        "|---|---|---|---|",
    ]
    for var in context.variables.values():
        val_str = f"Allowed: {var.allowed_values}" if var.allowed_values else f"e.g. {var.example}"
        lines.append(f"| `{var.name}` | {var.data_type} | {val_str} | {var.description} |")
    return "\n".join(lines)


def format_tool_catalogue(registry: ToolRegistry) -> str:
    """Format the registered tools into a clean markdown reference for the planner."""
    lines = []
    for tool in registry.list_tools():
        pre = " [Prerequisite data quality check]" if tool.is_pre_analysis else ""
        lines.append(f"#### Tool: `{tool.name}`{pre}")
        lines.append(f"- Display Name: {tool.display_name}")
        lines.append(f"- Category: {tool.category}")
        lines.append(f"- Description: {tool.description}")
        # Parameter schema fields
        param_fields = tool.parameter_schema.model_fields
        field_strs = []
        for fname, finfo in param_fields.items():
            req_str = "required" if finfo.is_required() else f"default={finfo.default}"
            field_strs.append(f"`{fname}` ({req_str}): {finfo.description or ''}")
        lines.append(f"- Parameters:\n  * " + "\n  * ".join(field_strs))
        lines.append("")
    return "\n".join(lines)


def build_planner_system_prompt(
    context: ExperimentContext, registry: ToolRegistry
) -> str:
    """Assemble the complete planner system prompt with active context and tool catalog."""
    return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        experiment_id=context.experiment_id,
        experiment_title=context.title,
        experiment_description=context.description,
        available_groups=", ".join(context.groups),
        available_task_types=", ".join(context.task_types),
        total_participants=context.total_participants,
        data_dictionary_formatted=format_data_dictionary(context),
        tool_catalogue_formatted=format_tool_catalogue(registry),
    )
