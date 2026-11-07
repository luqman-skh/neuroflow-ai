"""Unit tests for the ten deterministic placeholder analysis tools."""

import json
import pytest
from pydantic import ValidationError

from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.tools.behavioural_ml import (
    cluster_participant_behaviour,
    predict_task_performance,
)
from neuroflow.tools.data_quality import (
    detect_anomalous_sessions,
    validate_experiment_data,
)
from neuroflow.tools.group_analysis import (
    analyze_performance_correlations,
    analyze_task_difficulty,
    compare_experimental_groups,
)
from neuroflow.tools.performance import (
    analyze_learning_curve,
    analyze_memory_performance,
    analyze_response_time,
)


class TestParticipantPerformanceTools:
    """Tests for 5.1 Participant performance tools."""

    def test_analyze_memory_performance_defaults(self) -> None:
        result = analyze_memory_performance()
        assert isinstance(result, AnalysisResult)
        assert result.analysis_name == "analyze_memory_performance"
        assert result.status == "success"
        assert result.mock_data is True
        assert result.filters_applied["metric"] == "accuracy"
        assert result.filters_applied["task_type"] == "all"
        assert "mean" in result.statistics
        assert "median" in result.statistics
        assert "std_dev" in result.statistics
        assert result.chart_data is not None
        assert "data" in result.chart_data
        assert "layout" in result.chart_data
        assert len(result.limitations) > 0
        assert "DEMO/MOCK OUTPUT" in result.limitations[0]

    def test_analyze_memory_performance_deterministic(self) -> None:
        params = {"metric": "accuracy", "task_type": "recall", "group": "control"}
        res1 = analyze_memory_performance(**params)
        res2 = analyze_memory_performance(**params)
        assert res1.analysis_id == res2.analysis_id
        assert res1.statistics == res2.statistics
        assert res1.chart_data == res2.chart_data

    def test_analyze_memory_performance_metrics(self) -> None:
        for metric in ["accuracy", "recall_score", "completion_time_ms", "error_rate"]:
            res = analyze_memory_performance(metric=metric)
            assert res.statistics["metric"] == metric
            assert res.status == "success"

    def test_analyze_memory_performance_invalid_metric(self) -> None:
        with pytest.raises(ValidationError):
            analyze_memory_performance(metric="non_existent_metric")

    def test_analyze_learning_curve_progression(self) -> None:
        res = analyze_learning_curve(
            metric="accuracy", group="experimental", trial_start=1, trial_end=10
        )
        assert isinstance(res, AnalysisResult)
        assert res.mock_data is True
        assert res.filters_applied["trial_start"] == 1
        assert res.filters_applied["trial_end"] == 10
        assert len(res.statistics["trials"]) == 10
        assert len(res.statistics["values"]) == 10
        assert res.statistics["final_value"] > res.statistics["initial_value"]

    def test_analyze_learning_curve_invalid_range(self) -> None:
        with pytest.raises(ValidationError):
            analyze_learning_curve(trial_start=10, trial_end=2)

    def test_analyze_response_time_distributions(self) -> None:
        res = analyze_response_time(task_type="recognition", group="control")
        assert isinstance(res, AnalysisResult)
        assert res.mock_data is True
        stats = res.statistics
        assert stats["mean_response_time_ms"] > 0
        assert stats["median_response_time_ms"] > 0
        assert "p50" in stats["quantiles_ms"]
        assert stats["quantiles_ms"]["p90"] > stats["quantiles_ms"]["p10"]

    def test_analyze_response_time_invalid_range(self) -> None:
        with pytest.raises(ValidationError):
            analyze_response_time(trial_start=15, trial_end=5)


class TestGroupAndTaskAnalysisTools:
    """Tests for 5.2 Group and task analysis tools."""

    def test_compare_experimental_groups(self) -> None:
        res = compare_experimental_groups(
            metric="accuracy", group_a="control", group_b="experimental"
        )
        assert isinstance(res, AnalysisResult)
        assert res.mock_data is True
        stats = res.statistics
        assert stats["group_a"]["name"] == "control"
        assert stats["group_b"]["name"] == "experimental"
        assert "mean_difference" in stats
        assert "effect_size_cohens_d" in stats
        assert "p_value_illustrative" in stats
        assert "confidence_interval_95" in stats
        assert len(stats["confidence_interval_95"]) == 2

    def test_compare_experimental_groups_identical_groups_rejected(self) -> None:
        with pytest.raises(ValidationError):
            compare_experimental_groups(group_a="control", group_b="control")

    def test_analyze_performance_correlations(self) -> None:
        res = analyze_performance_correlations(
            x_variable="response_time_ms",
            y_variable="error_rate",
            correlation_method="pearson",
        )
        assert isinstance(res, AnalysisResult)
        assert res.mock_data is True
        stats = res.statistics
        assert stats["x_variable"] == "response_time_ms"
        assert stats["y_variable"] == "error_rate"
        assert -1.0 <= stats["correlation_coefficient_r"] <= 1.0
        assert stats["r_squared"] >= 0.0

    def test_analyze_performance_correlations_same_variable_rejected(self) -> None:
        with pytest.raises(ValidationError):
            analyze_performance_correlations(
                x_variable="response_time_ms", y_variable="response_time_ms"
            )

    def test_analyze_performance_correlations_deterministic(self) -> None:
        p = {"x_variable": "response_time_ms", "y_variable": "accuracy"}
        res1 = analyze_performance_correlations(**p)
        res2 = analyze_performance_correlations(**p)
        assert res1.analysis_id == res2.analysis_id
        assert res1.statistics == res2.statistics

    def test_analyze_task_difficulty_ordering(self) -> None:
        res_asc = analyze_task_difficulty(metric="accuracy", sort_order="ascending")
        res_desc = analyze_task_difficulty(metric="accuracy", sort_order="descending")
        assert res_asc.statistics["ranking"][0]["accuracy"] <= res_asc.statistics["ranking"][-1]["accuracy"]
        assert res_desc.statistics["ranking"][0]["accuracy"] >= res_desc.statistics["ranking"][-1]["accuracy"]


class TestBehaviouralMLTools:
    """Tests for 5.3 ML-based behavioural analysis tools."""

    def test_cluster_participant_behaviour(self) -> None:
        res = cluster_participant_behaviour(
            features=["response_time_ms", "accuracy", "error_rate"], n_clusters=4
        )
        assert isinstance(res, AnalysisResult)
        assert res.mock_data is True
        stats = res.statistics
        assert stats["n_clusters"] == 4
        assert len(stats["cluster_sizes"]) == 4
        assert len(stats["cluster_profiles"]) == 4
        assert any("EXPLICIT NOTICE" in lim for lim in res.limitations)

    def test_cluster_participant_behaviour_invalid_clusters(self) -> None:
        with pytest.raises(ValidationError):
            cluster_participant_behaviour(n_clusters=1)  # ge=2

        with pytest.raises(ValidationError):
            cluster_participant_behaviour(n_clusters=15)  # le=10

    def test_cluster_participant_behaviour_invalid_feature(self) -> None:
        with pytest.raises(ValidationError):
            cluster_participant_behaviour(features=["non_existent_feature"])

    def test_predict_task_performance(self) -> None:
        res = predict_task_performance(
            target="high_accuracy",
            features=["response_time_ms", "attempt_count", "trial_number"],
            task_type="recall",
        )
        assert isinstance(res, AnalysisResult)
        assert res.mock_data is True
        stats = res.statistics
        assert stats["target"] == "high_accuracy"
        assert "metrics" in stats
        assert 0.0 <= stats["metrics"]["accuracy"] <= 1.0
        assert "feature_importance" in stats
        assert len(stats["feature_importance"]) == 3
        assert any("EXPLICIT NOTICE" in lim for lim in res.limitations)

    def test_predict_task_performance_empty_features_rejected(self) -> None:
        with pytest.raises(ValidationError):
            predict_task_performance(features=[])


class TestDataQualityTools:
    """Tests for 5.4 Data quality and monitoring tools."""

    def test_validate_experiment_data(self) -> None:
        res = validate_experiment_data(task_type="all")
        assert isinstance(res, AnalysisResult)
        assert res.mock_data is True
        stats = res.statistics
        assert stats["total_records_checked"] > 0
        assert stats["valid_records_count"] > 0
        assert "integrity_score_pct" in stats
        assert stats["validation_passed"] is True
        assert len(res.warnings) > 0

    def test_detect_anomalous_sessions(self) -> None:
        for sens in ["low", "medium", "high"]:
            res = detect_anomalous_sessions(sensitivity=sens)
            assert isinstance(res, AnalysisResult)
            assert res.mock_data is True
            assert res.statistics["sensitivity"] == sens
            assert res.statistics["anomalous_sessions_count"] >= 1
            assert len(res.statistics["flagged_sessions"]) == res.statistics["anomalous_sessions_count"]
            assert any("Flagged sessions" in w for w in res.warnings)


class TestCommonContractsAndSerialization:
    """Verify contracts, JSON serializability, and echo of filters for all ten tools."""

    @pytest.mark.parametrize(
        "tool_func, kwargs",
        [
            (analyze_memory_performance, {"metric": "accuracy"}),
            (analyze_learning_curve, {"metric": "accuracy", "trial_end": 5}),
            (analyze_response_time, {"task_type": "recall"}),
            (compare_experimental_groups, {"group_a": "control", "group_b": "experimental"}),
            (analyze_performance_correlations, {"x_variable": "response_time_ms", "y_variable": "error_rate"}),
            (analyze_task_difficulty, {"metric": "accuracy"}),
            (cluster_participant_behaviour, {"n_clusters": 3}),
            (predict_task_performance, {"target": "high_accuracy"}),
            (validate_experiment_data, {"task_type": "recall"}),
            (detect_anomalous_sessions, {"sensitivity": "medium"}),
        ],
    )
    def test_all_tools_return_common_envelope_and_json_serializable(
        self, tool_func, kwargs
    ) -> None:
        result = tool_func(**kwargs)
        # 1. Type
        assert isinstance(result, AnalysisResult)
        # 2. Mock flag
        assert result.mock_data is True
        # 3. Echo parameters
        for key, val in kwargs.items():
            assert result.filters_applied.get(key) == val
        # 4. JSON serializable
        json_output = result.model_dump_json()
        assert isinstance(json_output, str)
        parsed = json.loads(json_output)
        assert parsed["mock_data"] is True
        assert parsed["analysis_name"] == result.analysis_name
        # 5. Plotly-compatible chart_data
        assert result.chart_data is not None
        assert isinstance(result.chart_data, dict)
        assert "data" in result.chart_data
        assert "layout" in result.chart_data
        # 6. Limitations
        assert len(result.limitations) > 0
