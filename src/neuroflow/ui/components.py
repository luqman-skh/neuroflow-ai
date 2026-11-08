"""UI helper components, formatters, and Plotly converters for NeuroFlow AI."""

from typing import Any
import plotly.graph_objects as go

from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.planning import AnalysisPlan, ClarificationNeeded
from neuroflow.schemas.session import ExecutionRecord, FinalReport


def convert_chart_data_to_figure(chart_data: dict[str, Any] | None) -> go.Figure | None:
    """Convert an AnalysisResult chart_data dictionary to a Plotly Figure object."""
    if not chart_data or not isinstance(chart_data, dict):
        return None
    try:
        data = chart_data.get("data", [])
        layout = chart_data.get("layout", {})
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380,
        )
        return fig
    except Exception:
        return None


def format_plan_markdown(plan: AnalysisPlan | None) -> str:
    """Render a human-readable markdown representation of a proposed analysis plan."""
    if not plan:
        return "*No analysis plan has been generated yet.*"

    md = [
        f"### Proposed Analysis Plan: `{plan.plan_id}`",
        f"**Summary:** {plan.summary}",
        "",
        "| Step | Tool Name | Purpose | Rationale | Parameters | Dependencies |",
        "|---|---|---|---|---|---|",
    ]

    for s in plan.steps:
        params_str = f"`{s.parameters}`"
        deps_str = ", ".join(s.dependencies) if s.dependencies else "None"
        md.append(
            f"| **{s.step_id}** | `{s.tool_name}` | {s.purpose} | {s.rationale} | {params_str} | {deps_str} |"
        )

    return "\n".join(md)


def format_clarification_markdown(clarification: ClarificationNeeded | None) -> str:
    """Format clarification details and options for display."""
    if not clarification:
        return ""

    md = [
        "### ⚠️ Clarification Needed Before Planning",
        f"**Question:** {clarification.question}",
        "",
    ]

    if clarification.missing_variables:
        md.append(f"- **Unknown Variables Requested:** {', '.join(clarification.missing_variables)}")
    if clarification.unsupported_features:
        md.append(f"- **Unsupported Actions:** {', '.join(clarification.unsupported_features)}")

    if clarification.options:
        md.append("\n**Suggested Options / Clarifications:**")
        for opt in clarification.options:
            md.append(f"1. {opt}")

    if clarification.suggested_alternatives:
        md.append(f"\n*Recommended supported alternatives:* {', '.join(clarification.suggested_alternatives)}")

    return "\n".join(md)


def format_execution_log(records: list[ExecutionRecord]) -> str:
    """Format execution audit records into a clean log table."""
    if not records:
        return "*No tools have executed yet.*"

    lines = [
        "| Step | Tool | Status | Parameters Passed | Timestamp |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        status_icon = "✅" if r.status == "success" else ("⚠️" if r.status == "warning" else "❌")
        ts = r.executed_at.strftime("%H:%M:%S")
        lines.append(
            f"| `{r.step_id}` | `{r.tool_name}` | {status_icon} **{r.status.upper()}** | `{r.parameters_passed}` | {ts} |"
        )
    return "\n".join(lines)


def format_results_markdown(results: dict[str, AnalysisResult]) -> str:
    """Format structured results and statistical findings into readable markdown."""
    if not results:
        return "*No tool execution outputs available.*"

    md = []
    for step_id, res in results.items():
        md.append(f"### Step Output: `{step_id}` — `{res.analysis_name}`")
        md.append(f"- **Status:** {res.status.upper()} | **Run ID:** `{res.analysis_id}` | **Mock Data Flag:** `{res.mock_data}`")
        md.append(f"- **Interpretation:** {res.interpretation}")

        if res.statistics:
            md.append("\n**Key Summary Statistics:**")
            for k, v in res.statistics.items():
                if k not in ["trials", "values", "cluster_profiles", "ranking"]:
                    md.append(f"  * **{k}:** {v}")

        if res.warnings:
            md.append("\n**⚠️ Warnings:**")
            for w in res.warnings:
                md.append(f"  * {w}")

        if res.limitations:
            md.append("\n*Limitations:* " + " | ".join(res.limitations))

        md.append("\n---\n")

    return "\n".join(md)


def format_report_markdown(report: FinalReport | None) -> str:
    """Format final synthesis and mock notice."""
    if not report:
        return "*Final synthesis report not yet generated.*"

    return f"""### Evidence-Grounded Scientific Synthesis
{report.synthesis}

---
> ⚠️ **{report.mock_data_notice}**
"""
