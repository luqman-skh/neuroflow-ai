"""Data quality validation and anomaly detection mock tools."""

import random
from typing import Any

from neuroflow.schemas.analysis import (
    AnalysisResult,
    DetectAnomaliesParams,
    ValidateExperimentDataParams,
)
from neuroflow.tools.utils import MOCK_DISCLAIMER_LIMITATION, get_deterministic_seed


def validate_experiment_data(
    params: ValidateExperimentDataParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Return a mock data-quality report. Required pre-analysis step in standard workflow."""
    if params is None and kwargs:
        validated = ValidateExperimentDataParams(**kwargs)
    elif isinstance(params, ValidateExperimentDataParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = ValidateExperimentDataParams(**merged)
    else:
        validated = ValidateExperimentDataParams()

    seed = get_deterministic_seed("validate_experiment_data", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    task_type = validated.task_type
    group = validated.group

    total_records = 1800 if group is None else 900
    missing_records = rng.randint(4, 9)
    duplicate_events = rng.randint(1, 4)
    invalid_timestamps = rng.randint(0, 2)
    incomplete_sessions = rng.randint(2, 5)

    valid_records = total_records - (missing_records + duplicate_events + invalid_timestamps)
    integrity_score = round((valid_records / total_records) * 100, 2)

    status = "success" if integrity_score >= 95.0 else "warning"

    chart_data = {
        "data": [
            {
                "type": "bar",
                "name": "Audit Metrics",
                "x": [
                    "Valid Records",
                    "Missing Records",
                    "Duplicate Events",
                    "Invalid Timestamps",
                    "Incomplete Sessions",
                ],
                "y": [
                    valid_records,
                    missing_records,
                    duplicate_events,
                    invalid_timestamps,
                    incomplete_sessions,
                ],
                "marker": {
                    "color": ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
                },
            }
        ],
        "layout": {
            "title": f"Data Integrity Audit ({task_type}, {group or 'all cohorts'})",
            "xaxis": {"title": "Quality Dimension"},
            "yaxis": {"title": "Count"},
        },
    }

    warnings = [
        f"Detected {incomplete_sessions} incomplete session(s) where session_completed=False.",
        f"Identified {missing_records} missing trial records across the dataset.",
    ]
    if duplicate_events > 0:
        warnings.append(f"Found {duplicate_events} duplicate event timestamps requiring deduplication.")

    interpretation = (
        f"Data validation completed with an overall integrity score of {integrity_score}% "
        f"({valid_records}/{total_records} valid records). Data quality is sufficient to proceed "
        f"with statistical analysis, with {incomplete_sessions} sessions flagged for monitoring."
    )

    return AnalysisResult(
        analysis_id=f"val_{seed:08x}",
        analysis_name="validate_experiment_data",
        status=status,
        mock_data=True,
        filters_applied=filters,
        statistics={
            "total_records_checked": total_records,
            "valid_records_count": valid_records,
            "missing_record_count": missing_records,
            "duplicate_event_count": duplicate_events,
            "invalid_timestamp_count": invalid_timestamps,
            "incomplete_session_count": incomplete_sessions,
            "integrity_score_pct": integrity_score,
            "validation_passed": True,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Data quality checks are synthetic simulations of log-file validation.",
        ],
        warnings=warnings,
    )


def detect_anomalous_sessions(
    params: DetectAnomaliesParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Return illustrative sessions flagged for researcher review."""
    if params is None and kwargs:
        validated = DetectAnomaliesParams(**kwargs)
    elif isinstance(params, DetectAnomaliesParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = DetectAnomaliesParams(**merged)
    else:
        validated = DetectAnomaliesParams()

    seed = get_deterministic_seed("detect_anomalous_sessions", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    sensitivity = validated.sensitivity
    task_type = validated.task_type
    group = validated.group

    total_sessions = 30 if group in ["control", "experimental"] else 60

    # Number of anomalies based on sensitivity
    if sensitivity == "low":
        anomaly_count = 1
        threshold_score = 0.85
    elif sensitivity == "high":
        anomaly_count = 5
        threshold_score = 0.65
    else:  # medium
        anomaly_count = 3
        threshold_score = 0.75

    flag_reasons = [
        "Rapid wrong-response bursts (<400ms per trial indicating disengagement)",
        "Extended idle latency between consecutive trials (>35s)",
        "Abnormal trial retry loop count exceeding 3 SD above group mean",
        "Multiple missing keypress events during spatial navigation phase",
        "Abrupt session termination prior to final trial block",
    ]

    flagged_sessions = []
    for i in range(anomaly_count):
        s_id = f"S{100 + i * 7:03d}"
        p_id = f"P{20 + i * 5:03d}"
        score = round(rng.uniform(threshold_score, 0.96), 3)
        reason = flag_reasons[i % len(flag_reasons)]
        flagged_sessions.append(
            {
                "session_id": s_id,
                "participant_id": p_id,
                "anomaly_score": score,
                "reason": reason,
                "recommended_action": "Inspect raw event timestamps before cohort aggregation",
            }
        )

    chart_data = {
        "data": [
            {
                "type": "bar",
                "name": "Flagged Session Anomaly Score",
                "x": [s["session_id"] for s in flagged_sessions],
                "y": [s["anomaly_score"] for s in flagged_sessions],
                "marker": {"color": "#e377c2"},
            }
        ],
        "layout": {
            "title": f"Anomalous Sessions Detected (Sensitivity: {sensitivity})",
            "xaxis": {"title": "Session ID"},
            "yaxis": {"title": "Anomaly Score [0-1]"},
        },
    }

    warnings = [
        "Flagged sessions are candidates for researcher inspection, not inherently invalid or corrupted.",
        f"Threshold sensitivity was set to '{sensitivity}' (cutoff score >= {threshold_score}).",
    ]

    interpretation = (
        f"Anomaly scan across {total_sessions} sessions with {sensitivity} sensitivity detected "
        f"{anomaly_count} flagged session(s) requiring manual review. "
        f"Highest anomaly score was {max(s['anomaly_score'] for s in flagged_sessions)} in session "
        f"'{flagged_sessions[0]['session_id']}'."
    )

    return AnalysisResult(
        analysis_id=f"anom_{seed:08x}",
        analysis_name="detect_anomalous_sessions",
        status="warning" if anomaly_count > 0 else "success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "sensitivity": sensitivity,
            "threshold_score": threshold_score,
            "total_sessions_audited": total_sessions,
            "anomalous_sessions_count": anomaly_count,
            "flagged_sessions": flagged_sessions,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "Anomaly scores are mock heuristics; no actual participant sessions were excluded.",
        ],
        warnings=warnings,
    )
