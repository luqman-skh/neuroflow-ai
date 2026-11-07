"""Participant performance mock analysis tools."""

import random
from typing import Any

from neuroflow.schemas.analysis import (
    AnalysisResult,
    LearningCurveParams,
    MemoryPerformanceParams,
    ResponseTimeParams,
)
from neuroflow.tools.utils import MOCK_DISCLAIMER_LIMITATION, get_deterministic_seed


def analyze_memory_performance(
    params: MemoryPerformanceParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Summarise participant memory-task performance.

    Returns illustrative accuracy, recall score, completion time, and distribution summaries.
    """
    if params is None and kwargs:
        validated = MemoryPerformanceParams(**kwargs)
    elif isinstance(params, MemoryPerformanceParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = MemoryPerformanceParams(**merged)
    else:
        validated = MemoryPerformanceParams()

    seed = get_deterministic_seed("analyze_memory_performance", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    metric = validated.metric
    group = validated.group
    task_type = validated.task_type

    # Deterministic base adjustments depending on parameters
    group_offset = 0.08 if group == "experimental" else (-0.05 if group == "control" else 0.0)
    task_offset = {"recall": -0.04, "recognition": 0.06, "spatial_memory": -0.02}.get(
        task_type, 0.0
    )

    if metric == "accuracy":
        mean_val = round(min(0.96, max(0.40, 0.72 + group_offset + task_offset + rng.uniform(-0.02, 0.02))), 4)
        std_val = round(rng.uniform(0.06, 0.12), 4)
        median_val = round(mean_val + rng.uniform(-0.02, 0.02), 4)
        iqr_val = round(std_val * 1.35, 4)
        min_val = round(max(0.20, mean_val - 2.5 * std_val), 4)
        max_val = round(min(1.0, mean_val + 2.2 * std_val), 4)
        unit = "ratio"
    elif metric == "recall_score":
        mean_val = round(min(100.0, max(20.0, 74.0 + (group_offset * 100) + (task_offset * 100) + rng.uniform(-2.0, 2.0))), 2)
        std_val = round(rng.uniform(6.0, 12.0), 2)
        median_val = round(mean_val + rng.uniform(-2.0, 2.0), 2)
        iqr_val = round(std_val * 1.35, 2)
        min_val = round(max(0.0, mean_val - 2.5 * std_val), 2)
        max_val = round(min(100.0, mean_val + 2.0 * std_val), 2)
        unit = "points"
    elif metric == "error_rate":
        mean_val = round(min(0.60, max(0.04, 0.28 - group_offset - task_offset + rng.uniform(-0.02, 0.02))), 4)
        std_val = round(rng.uniform(0.05, 0.10), 4)
        median_val = round(mean_val + rng.uniform(-0.02, 0.02), 4)
        iqr_val = round(std_val * 1.35, 4)
        min_val = round(max(0.0, mean_val - 2.0 * std_val), 4)
        max_val = round(min(1.0, mean_val + 2.5 * std_val), 4)
        unit = "ratio"
    else:  # completion_time_ms
        mean_val = round(max(800.0, 2250.0 - (group_offset * 2000) - (task_offset * 1500) + rng.uniform(-50.0, 50.0)), 1)
        std_val = round(rng.uniform(220.0, 380.0), 1)
        median_val = round(mean_val + rng.uniform(-40.0, 40.0), 1)
        iqr_val = round(std_val * 1.35, 1)
        min_val = round(mean_val - 2.2 * std_val, 1)
        max_val = round(mean_val + 2.8 * std_val, 1)
        unit = "ms"

    sample_size = (
        len(validated.participant_ids)
        if validated.participant_ids
        else (30 if group in ["control", "experimental"] else 60)
    )

    task_breakdown = {
        "recall": round(mean_val * 0.94, 3),
        "recognition": round(mean_val * 1.08, 3),
        "spatial_memory": round(mean_val * 0.98, 3),
    }

    # Plotly distribution bar/box data
    chart_data = {
        "data": [
            {
                "type": "box",
                "name": f"{group or 'all_participants'} ({metric})",
                "q1": [round(median_val - iqr_val / 2, 3)],
                "median": [median_val],
                "q3": [round(median_val + iqr_val / 2, 3)],
                "lowerfence": [min_val],
                "upperfence": [max_val],
                "mean": [mean_val],
            }
        ],
        "layout": {
            "title": f"Memory Performance Distribution: {metric} ({task_type})",
            "xaxis": {"title": "Cohort"},
            "yaxis": {"title": f"{metric} [{unit}]"},
        },
    }

    interpretation = (
        f"Analysis of {metric} across {sample_size} participants (task: {task_type}, group: {group or 'all'}) "
        f"revealed an illustrative mean of {mean_val} {unit} (SD = {std_val}, median = {median_val}). "
        f"Performance appears stable with normal dispersion across the cohort."
    )

    return AnalysisResult(
        analysis_id=f"mem_perf_{seed:08x}",
        analysis_name="analyze_memory_performance",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "metric": metric,
            "unit": unit,
            "sample_size": sample_size,
            "mean": mean_val,
            "std_dev": std_val,
            "median": median_val,
            "iqr": iqr_val,
            "min": min_val,
            "max": max_val,
            "task_breakdown": task_breakdown if task_type == "all" else None,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Participant samples are synthetic; distribution quantiles are simulated.",
        ],
        warnings=[],
    )


def analyze_learning_curve(
    params: LearningCurveParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Illustrate performance progression over repeated trials or sessions."""
    if params is None and kwargs:
        validated = LearningCurveParams(**kwargs)
    elif isinstance(params, LearningCurveParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = LearningCurveParams(**merged)
    else:
        validated = LearningCurveParams()

    seed = get_deterministic_seed("analyze_learning_curve", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    metric = validated.metric
    group = validated.group
    trial_start = validated.trial_start
    trial_end = validated.trial_end

    trials = list(range(trial_start, trial_end + 1))
    n_trials = len(trials)

    group_boost = 0.05 if group == "experimental" else (-0.03 if group == "control" else 0.0)

    # Generate realistic learning curve values
    values = []
    for i, t in enumerate(trials):
        progress = i / max(1, n_trials - 1)
        if metric == "accuracy":
            # Asymptotic progression toward ~0.88 + group_boost
            val = 0.52 + group_boost + (0.35 * (1.0 - (0.65 ** (i + 1)))) + rng.uniform(-0.02, 0.02)
            val = round(min(0.99, max(0.20, val)), 4)
        elif metric == "error_rate":
            # Exponential decay
            val = 0.48 - group_boost - (0.32 * (1.0 - (0.68 ** (i + 1)))) + rng.uniform(-0.02, 0.02)
            val = round(min(0.80, max(0.02, val)), 4)
        else:  # response_time_ms
            # Latency reduction over practice
            val = 2600.0 - (group_boost * 1500) - (800.0 * (1.0 - (0.7 ** (i + 1)))) + rng.uniform(-40.0, 40.0)
            val = round(max(750.0, val), 1)
        values.append(val)

    rate_of_change = round((values[-1] - values[0]) / max(1, n_trials - 1), 4)
    plateau_trial = trials[min(len(trials) - 1, max(1, int(n_trials * 0.7)))]

    chart_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": f"Learning Curve ({group or 'combined'})",
                "x": trials,
                "y": values,
                "line": {"color": "#1f77b4", "width": 3},
                "marker": {"size": 7},
            }
        ],
        "layout": {
            "title": f"Learning Curve: {metric} over Trials {trial_start} to {trial_end}",
            "xaxis": {"title": "Trial Number"},
            "yaxis": {"title": metric},
        },
    }

    direction = "improvement" if (rate_of_change > 0 if metric == "accuracy" else rate_of_change < 0) else "plateau"
    interpretation = (
        f"Learning progression for {metric} from trial {trial_start} to {trial_end} demonstrates steady {direction}. "
        f"Initial mean was {values[0]}, progressing to {values[-1]} by trial {trial_end} "
        f"(average rate of change: {rate_of_change} per trial, plateau around trial {plateau_trial})."
    )

    return AnalysisResult(
        analysis_id=f"learn_curv_{seed:08x}",
        analysis_name="analyze_learning_curve",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "metric": metric,
            "trial_start": trial_start,
            "trial_end": trial_end,
            "trials": trials,
            "values": values,
            "initial_value": values[0],
            "final_value": values[-1],
            "rate_of_change": rate_of_change,
            "plateau_trial": plateau_trial,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Learning curves are modelled with synthetic power-law functions, not individual trial logs.",
        ],
        warnings=[],
    )


def analyze_response_time(
    params: ResponseTimeParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Illustrate response-time distributions and variability."""
    if params is None and kwargs:
        validated = ResponseTimeParams(**kwargs)
    elif isinstance(params, ResponseTimeParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = ResponseTimeParams(**merged)
    else:
        validated = ResponseTimeParams()

    seed = get_deterministic_seed("analyze_response_time", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    group = validated.group
    task_type = validated.task_type

    base_rt = 1580.0
    if group == "experimental":
        base_rt -= 180.0
    elif group == "control":
        base_rt += 90.0

    if task_type == "recall":
        base_rt += 220.0
    elif task_type == "recognition":
        base_rt -= 310.0
    elif task_type == "spatial_memory":
        base_rt += 450.0

    mean_rt = round(base_rt + rng.uniform(-30.0, 30.0), 1)
    median_rt = round(mean_rt - rng.uniform(20.0, 60.0), 1)  # positive skew typical in RT
    std_rt = round(rng.uniform(190.0, 340.0), 1)

    quantiles = {
        "p10": round(mean_rt - 1.28 * std_rt, 1),
        "p25": round(mean_rt - 0.67 * std_rt, 1),
        "p50": median_rt,
        "p75": round(mean_rt + 0.67 * std_rt, 1),
        "p90": round(mean_rt + 1.28 * std_rt, 1),
        "p99": round(mean_rt + 2.33 * std_rt, 1),
    }

    variability_cv = round(std_rt / mean_rt, 3)
    outliers_count = rng.randint(1, 4)

    # Histogram bin simulation
    bins = [round(quantiles["p10"] + i * (quantiles["p90"] - quantiles["p10"]) / 8, 1) for i in range(9)]
    counts = [rng.randint(3, 18) for _ in range(8)]

    chart_data = {
        "data": [
            {
                "type": "bar",
                "name": "Response Time Distribution",
                "x": [f"{bins[i]:.0f}-{bins[i+1]:.0f} ms" for i in range(8)],
                "y": counts,
                "marker": {"color": "#2ca02c"},
            }
        ],
        "layout": {
            "title": f"Response Time Latency Distribution ({task_type}, {group or 'all cohorts'})",
            "xaxis": {"title": "Latency Bin (ms)"},
            "yaxis": {"title": "Observed Trials"},
        },
    }

    interpretation = (
        f"Response latency analysis indicates a positively skewed distribution with mean RT of {mean_rt} ms "
        f"(SD = {std_rt} ms, median = {median_rt} ms). Coefficient of variation is {variability_cv}. "
        f"{outliers_count} trials exceeded the 99th percentile outlier threshold."
    )

    return AnalysisResult(
        analysis_id=f"rt_dist_{seed:08x}",
        analysis_name="analyze_response_time",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "mean_response_time_ms": mean_rt,
            "median_response_time_ms": median_rt,
            "std_dev_ms": std_rt,
            "coefficient_of_variation": variability_cv,
            "quantiles_ms": quantiles,
            "outlier_trials_count": outliers_count,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Response latency distributions are parameterized ex-Gaussian mock estimates.",
        ],
        warnings=[] if outliers_count < 4 else [f"Noted {outliers_count} mock outliers in upper tail."],
    )
