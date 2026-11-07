"""Group comparison, correlation, and task difficulty analysis tools."""

import random
from typing import Any

from neuroflow.schemas.analysis import (
    AnalysisResult,
    GroupComparisonParams,
    PerformanceCorrelationsParams,
    TaskDifficultyParams,
)
from neuroflow.tools.utils import MOCK_DISCLAIMER_LIMITATION, get_deterministic_seed


def compare_experimental_groups(
    params: GroupComparisonParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Illustrate a comparison of an outcome between two experimental groups."""
    if params is None and kwargs:
        validated = GroupComparisonParams(**kwargs)
    elif isinstance(params, GroupComparisonParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = GroupComparisonParams(**merged)
    else:
        validated = GroupComparisonParams()

    seed = get_deterministic_seed("compare_experimental_groups", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    metric = validated.metric
    group_a = validated.group_a
    group_b = validated.group_b
    task_type = validated.task_type

    # Deterministic mock calculation based on metrics
    if metric == "accuracy":
        mean_a = round(0.68 + rng.uniform(-0.02, 0.02), 4)
        mean_b = round(0.81 + rng.uniform(-0.02, 0.02), 4)
        std_a = round(rng.uniform(0.08, 0.11), 4)
        std_b = round(rng.uniform(0.07, 0.10), 4)
        unit = "ratio"
    elif metric == "recall_score":
        mean_a = round(68.5 + rng.uniform(-2.0, 2.0), 2)
        mean_b = round(82.0 + rng.uniform(-2.0, 2.0), 2)
        std_a = round(rng.uniform(7.0, 11.0), 2)
        std_b = round(rng.uniform(6.0, 10.0), 2)
        unit = "points"
    elif metric == "error_rate":
        mean_a = round(0.32 + rng.uniform(-0.02, 0.02), 4)
        mean_b = round(0.19 + rng.uniform(-0.02, 0.02), 4)
        std_a = round(rng.uniform(0.06, 0.09), 4)
        std_b = round(rng.uniform(0.05, 0.08), 4)
        unit = "ratio"
    else:  # response_time_ms
        mean_a = round(1680.0 + rng.uniform(-40.0, 40.0), 1)
        mean_b = round(1410.0 + rng.uniform(-40.0, 40.0), 1)
        std_a = round(rng.uniform(200.0, 310.0), 1)
        std_b = round(rng.uniform(180.0, 290.0), 1)
        unit = "ms"

    mean_diff = round(mean_b - mean_a, 4)
    pooled_sd = round(((std_a**2 + std_b**2) / 2) ** 0.5, 4)
    cohens_d = round(abs(mean_diff) / max(0.0001, pooled_sd), 3)

    t_stat = round(rng.uniform(2.8, 3.9), 3)
    p_val = round(rng.uniform(0.0008, 0.0050), 5)
    ci_lower = round(mean_diff - 1.96 * (pooled_sd / (30**0.5)), 4)
    ci_upper = round(mean_diff + 1.96 * (pooled_sd / (30**0.5)), 4)

    chart_data = {
        "data": [
            {
                "type": "bar",
                "name": "Group Mean Comparison",
                "x": [group_a, group_b],
                "y": [mean_a, mean_b],
                "error_y": {
                    "type": "data",
                    "array": [std_a, std_b],
                    "visible": True,
                },
                "marker": {"color": ["#1f77b4", "#ff7f0e"]},
            }
        ],
        "layout": {
            "title": f"Group Comparison: {group_a} vs {group_b} on {metric} ({task_type})",
            "xaxis": {"title": "Experimental Group"},
            "yaxis": {"title": f"{metric} [{unit}]"},
        },
    }

    interpretation = (
        f"Comparison on {metric} between '{group_a}' (mean = {mean_a}, SD = {std_a}) and "
        f"'{group_b}' (mean = {mean_b}, SD = {std_b}) yielded an illustrative mean difference of {mean_diff} "
        f"(Cohen's d = {cohens_d}, illustrative Welch's t({56.2}) = {t_stat}, p = {p_val}, 95% CI [{ci_lower}, {ci_upper}]). "
        f"This illustrative difference favors group '{group_b if mean_diff > 0 else group_a}'."
    )

    return AnalysisResult(
        analysis_id=f"grp_comp_{seed:08x}",
        analysis_name="compare_experimental_groups",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "metric": metric,
            "unit": unit,
            "group_a": {
                "name": group_a,
                "n": 30,
                "mean": mean_a,
                "std_dev": std_a,
            },
            "group_b": {
                "name": group_b,
                "n": 30,
                "mean": mean_b,
                "std_dev": std_b,
            },
            "mean_difference": mean_diff,
            "effect_size_cohens_d": cohens_d,
            "illustrative_test_name": "Two-sample Welch's t-test (illustrative mock)",
            "test_statistic_t": t_stat,
            "p_value_illustrative": p_val,
            "confidence_interval_95": [ci_lower, ci_upper],
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Statistical test parameters, p-values, and effect sizes are simulated mock values. "
            "Do not infer actual scientific significance.",
        ],
        warnings=[],
    )


def analyze_performance_correlations(
    params: PerformanceCorrelationsParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Illustrate associations between performance variables."""
    if params is None and kwargs:
        validated = PerformanceCorrelationsParams(**kwargs)
    elif isinstance(params, PerformanceCorrelationsParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = PerformanceCorrelationsParams(**merged)
    else:
        raise ValueError("x_variable and y_variable are required for correlation analysis")

    seed = get_deterministic_seed("analyze_performance_correlations", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    x_var = validated.x_variable
    y_var = validated.y_variable
    group = validated.group
    task_type = validated.task_type
    method = validated.correlation_method

    # Plausible associations based on standard cognitive paradigms:
    # RT & Error Rate -> Positive association (+0.40 to +0.55) or speed-accuracy tradeoff
    # RT & Accuracy -> Negative association (-0.35 to -0.50)
    # Attempt Count & Error Rate -> Positive (+0.60)
    # Trial Number & Error Rate -> Negative (-0.45) due to learning
    pair = {x_var, y_var}
    if {"response_time_ms", "accuracy"} == pair:
        base_r = -0.42
    elif {"response_time_ms", "error_rate"} == pair:
        base_r = 0.46
    elif {"attempt_count", "error_rate"} == pair:
        base_r = 0.58
    elif {"trial_number", "error_rate"} == pair:
        base_r = -0.48
    elif {"trial_number", "accuracy"} == pair:
        base_r = 0.52
    else:
        base_r = 0.25

    # Slight perturbation
    r_val = round(max(-0.95, min(0.95, base_r + rng.uniform(-0.06, 0.06))), 3)
    r_squared = round(r_val**2, 4)
    p_val = round(rng.uniform(0.0001, 0.0040), 5)
    sample_size = 30 if group in ["control", "experimental"] else 60

    # Synthetic scatterplot points (15 illustrative points)
    scatter_x = []
    scatter_y = []
    for i in range(15):
        val_x = round(rng.uniform(10.0, 90.0), 2)
        noise = rng.uniform(-15.0, 15.0)
        val_y = round(max(5.0, min(95.0, 50.0 + (r_val * (val_x - 50.0)) + noise)), 2)
        scatter_x.append(val_x)
        scatter_y.append(val_y)

    chart_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "name": f"Observed ({group or 'all'})",
                "x": scatter_x,
                "y": scatter_y,
                "marker": {"size": 8, "color": "#1f77b4"},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Linear Trend (Illustrative)",
                "x": [min(scatter_x), max(scatter_x)],
                "y": [
                    round(50.0 + r_val * (min(scatter_x) - 50.0), 2),
                    round(50.0 + r_val * (max(scatter_x) - 50.0), 2),
                ],
                "line": {"color": "#d62728", "dash": "dash"},
            },
        ],
        "layout": {
            "title": f"Correlation: {x_var} vs {y_var} ({method.capitalize()})",
            "xaxis": {"title": x_var},
            "yaxis": {"title": y_var},
        },
    }

    strength = "moderate" if abs(r_val) < 0.6 else "strong"
    direction = "positive" if r_val > 0 else "negative"
    interpretation = (
        f"Illustrative {method} correlation analysis between '{x_var}' and '{y_var}' shows a "
        f"{strength} {direction} association (r = {r_val}, R² = {r_squared}, illustrative p = {p_val}, N = {sample_size}). "
        f"This indicates that higher {x_var} tends to be accompanied by {'higher' if r_val > 0 else 'lower'} {y_var}."
    )

    return AnalysisResult(
        analysis_id=f"corr_{seed:08x}",
        analysis_name="analyze_performance_correlations",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "x_variable": x_var,
            "y_variable": y_var,
            "method": method,
            "sample_size": sample_size,
            "correlation_coefficient_r": r_val,
            "r_squared": r_squared,
            "p_value_illustrative": p_val,
            "direction": direction,
            "strength": strength,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Correlation does not imply causation. All coefficients and scatter traces are synthetic mock data.",
        ],
        warnings=[],
    )


def analyze_task_difficulty(
    params: TaskDifficultyParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Illustrate differences in performance across memory tasks."""
    if params is None and kwargs:
        validated = TaskDifficultyParams(**kwargs)
    elif isinstance(params, TaskDifficultyParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = TaskDifficultyParams(**merged)
    else:
        validated = TaskDifficultyParams()

    seed = get_deterministic_seed("analyze_task_difficulty", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    metric = validated.metric
    group = validated.group
    sort_order = validated.sort_order

    # Task difficulty baseline:
    # recognition: easiest (high accuracy, low error, fast RT)
    # recall: moderate
    # spatial_memory: hardest (lowest accuracy, highest error, slowest RT)
    tasks_raw = {
        "recognition": {
            "task_type": "recognition",
            "accuracy": round(0.85 + rng.uniform(-0.02, 0.02), 3),
            "response_time_ms": round(1220.0 + rng.uniform(-30.0, 30.0), 1),
            "error_rate": round(0.15 + rng.uniform(-0.02, 0.02), 3),
            "completion_rate": round(0.98 + rng.uniform(-0.01, 0.01), 3),
        },
        "recall": {
            "task_type": "recall",
            "accuracy": round(0.72 + rng.uniform(-0.02, 0.02), 3),
            "response_time_ms": round(1650.0 + rng.uniform(-30.0, 30.0), 1),
            "error_rate": round(0.28 + rng.uniform(-0.02, 0.02), 3),
            "completion_rate": round(0.93 + rng.uniform(-0.01, 0.01), 3),
        },
        "spatial_memory": {
            "task_type": "spatial_memory",
            "accuracy": round(0.61 + rng.uniform(-0.02, 0.02), 3),
            "response_time_ms": round(2150.0 + rng.uniform(-40.0, 40.0), 1),
            "error_rate": round(0.39 + rng.uniform(-0.02, 0.02), 3),
            "completion_rate": round(0.87 + rng.uniform(-0.02, 0.02), 3),
        },
    }

    # Sort based on sort_order and metric
    reverse = sort_order == "descending"
    ranked_tasks = sorted(tasks_raw.values(), key=lambda t: t[metric], reverse=reverse)

    task_names = [t["task_type"] for t in ranked_tasks]
    metric_values = [t[metric] for t in ranked_tasks]

    chart_data = {
        "data": [
            {
                "type": "bar",
                "name": f"Task {metric}",
                "x": task_names,
                "y": metric_values,
                "marker": {"color": ["#636EFA", "#EF553B", "#00CC96"]},
            }
        ],
        "layout": {
            "title": f"Task Difficulty Comparison by {metric} ({group or 'all cohorts'})",
            "xaxis": {"title": "Task Category"},
            "yaxis": {"title": metric},
        },
    }

    hardest = "spatial_memory"
    easiest = "recognition"
    interpretation = (
        f"Across memory tasks, relative difficulty ranks '{hardest}' as the most demanding task "
        f"(lowest accuracy and highest latency), followed by 'recall', with '{easiest}' being the easiest. "
        f"Performance metrics rank in {sort_order} order of {metric}."
    )

    return AnalysisResult(
        analysis_id=f"diff_{seed:08x}",
        analysis_name="analyze_task_difficulty",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "metric": metric,
            "sort_order": sort_order,
            "ranking": ranked_tasks,
            "hardest_task": hardest,
            "easiest_task": easiest,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Task difficulty rankings are mock estimates based on prototypical memory-task expectations.",
        ],
        warnings=[],
    )
