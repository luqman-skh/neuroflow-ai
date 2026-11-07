"""Demo script to execute and inspect NeuroFlow placeholder analysis tools."""

import json
import sys
from pathlib import Path

# Add src to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuroflow.tools import get_tool_registry



def main() -> None:
    print("=" * 70)
    print("NeuroFlow AI - Milestone 1 Analysis Tools Demonstration")
    print("=" * 70)

    registry = get_tool_registry()
    print(f"\nRegistered tools ({len(registry.list_tools())}):")
    for tool in registry.list_tools():
        pre = " [Prerequisite]" if tool.is_pre_analysis else ""
        print(f" - {tool.name:32} | {tool.category:30}{pre}")

    print("\n" + "-" * 70)
    print("1. Executing Prerequisite Data Quality Tool: validate_experiment_data")
    print("-" * 70)
    res_val = registry.execute_tool(
        "validate_experiment_data", {"task_type": "all", "group": "control"}
    )
    print(f"Analysis ID:    {res_val.analysis_id}")
    print(f"Status:         {res_val.status}")
    print(f"Mock Data Flag: {res_val.mock_data}")
    print(f"Statistics:     {json.dumps(res_val.statistics, indent=2)}")
    print(f"Interpretation: {res_val.interpretation}")
    print(f"Warnings:       {res_val.warnings}")

    print("\n" + "-" * 70)
    print("2. Executing Participant Performance Tool: analyze_memory_performance")
    print("-" * 70)
    res_mem = registry.execute_tool(
        "analyze_memory_performance",
        {"metric": "accuracy", "task_type": "recall", "group": "experimental"},
    )
    print(f"Analysis ID:    {res_mem.analysis_id}")
    print(f"Filters:        {res_mem.filters_applied}")
    print(f"Statistics:     {json.dumps(res_mem.statistics, indent=2)}")
    print(f"Interpretation: {res_mem.interpretation}")
    print(f"Chart Trace:    {res_mem.chart_data['data'][0]['name']}")

    print("\n" + "-" * 70)
    print("3. Executing Group Comparison Tool: compare_experimental_groups")
    print("-" * 70)
    res_grp = registry.execute_tool(
        "compare_experimental_groups",
        {
            "metric": "accuracy",
            "group_a": "control",
            "group_b": "experimental",
            "task_type": "all",
        },
    )
    print(f"Mean Difference: {res_grp.statistics['mean_difference']}")
    print(f"Cohen's d:       {res_grp.statistics['effect_size_cohens_d']}")
    print(f"p-value (mock):  {res_grp.statistics['p_value_illustrative']}")
    print(f"Interpretation:  {res_grp.interpretation}")

    print("\n" + "-" * 70)
    print("4. Executing Behavioural Clustering Tool: cluster_participant_behaviour")
    print("-" * 70)
    res_clust = registry.execute_tool(
        "cluster_participant_behaviour",
        {"n_clusters": 3, "features": ["response_time_ms", "accuracy", "error_rate"]},
    )
    print(f"Clusters:        {list(res_clust.statistics['cluster_sizes'].keys())}")
    print(f"Silhouette:      {res_clust.statistics['illustrative_silhouette_score']}")
    print(f"Limitations:     {res_clust.limitations}")

    print("\n" + "-" * 70)
    print("5. Verifying Determinism Across Consecutive Tool Calls")
    print("-" * 70)
    call_1 = registry.execute_tool(
        "analyze_learning_curve",
        {"metric": "accuracy", "group": "experimental", "trial_start": 1, "trial_end": 8},
    )
    call_2 = registry.execute_tool(
        "analyze_learning_curve",
        {"metric": "accuracy", "group": "experimental", "trial_start": 1, "trial_end": 8},
    )
    assert call_1.analysis_id == call_2.analysis_id, "IDs must match!"
    assert call_1.statistics == call_2.statistics, "Statistics must match!"
    assert call_1.chart_data == call_2.chart_data, "Chart data must match!"
    print("PASS: Consecutive executions returned 100% identical deterministic results.")

    print("\n" + "-" * 70)
    print("6. Verifying JSON Serialization of AnalysisResult")
    print("-" * 70)
    json_str = call_1.model_dump_json(indent=2)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert parsed["mock_data"] is True
    print(f"PASS: AnalysisResult serialized to JSON ({len(json_str)} bytes).")
    print("\nAll demonstration checks passed successfully!")


if __name__ == "__main__":
    main()
