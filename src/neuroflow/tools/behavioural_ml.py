"""ML-based behavioural clustering and supervised prediction mock analysis tools."""

import random
from typing import Any

from neuroflow.schemas.analysis import (
    AnalysisResult,
    ClusterBehaviourParams,
    PredictPerformanceParams,
)
from neuroflow.tools.utils import MOCK_DISCLAIMER_LIMITATION, get_deterministic_seed


def cluster_participant_behaviour(
    params: ClusterBehaviourParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Return illustrative participant behavioural clusters based on performance features."""
    if params is None and kwargs:
        validated = ClusterBehaviourParams(**kwargs)
    elif isinstance(params, ClusterBehaviourParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = ClusterBehaviourParams(**merged)
    else:
        validated = ClusterBehaviourParams()

    seed = get_deterministic_seed("cluster_participant_behaviour", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    features = validated.features
    k = validated.n_clusters
    group = validated.group

    total_pts = 30 if group in ["control", "experimental"] else 60

    # Partition total_pts across k clusters deterministically
    base_size = total_pts // k
    remainder = total_pts % k
    cluster_sizes = {}
    for i in range(k):
        size = base_size + (1 if i < remainder else 0)
        cluster_sizes[f"Cluster_{i}"] = size

    # Predefined archetypes for realistic behavioural phenotypes
    archetype_names = [
        "High-Accuracy Deliberate (Slow, precise)",
        "Fast Automated (Rapid, low errors)",
        "Struggling / High Error (Multiple retries)",
        "Exploratory Variable (High RT variance)",
        "Fast Guessing (Very low RT, high errors)",
        "Adaptive Learner (Steep trial progression)",
        "Fatigued / Declining (Late trial drop)",
        "Risk-Averse (Excessive deliberation)",
        "Average Baseline (Moderate performance)",
        "Residual Atypical Cluster",
    ]

    cluster_profiles = {}
    scatter_traces = []
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    for i in range(k):
        c_name = f"Cluster_{i}"
        label = archetype_names[i % len(archetype_names)]
        size = cluster_sizes[c_name]

        # Generate cluster centroid and synthetic 2D PCA projection coordinates
        cx = round(rng.uniform(-2.5, 2.5), 2)
        cy = round(rng.uniform(-2.5, 2.5), 2)

        pts_x = [round(cx + rng.gauss(0, 0.45), 2) for _ in range(size)]
        pts_y = [round(cy + rng.gauss(0, 0.45), 2) for _ in range(size)]

        cluster_profiles[c_name] = {
            "label": label,
            "participant_count": size,
            "percentage": round((size / total_pts) * 100, 1),
            "centroid_pca": [cx, cy],
            "key_traits": f"Characteristics associated with {label.lower()} across {len(features)} features.",
        }

        scatter_traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": f"{c_name}: {label[:20]}...",
                "x": pts_x,
                "y": pts_y,
                "marker": {"size": 8, "color": color_palette[i % len(color_palette)]},
            }
        )

    silhouette = round(rng.uniform(0.48, 0.62), 3)

    chart_data = {
        "data": scatter_traces,
        "layout": {
            "title": f"Behavioural Clusters (k={k}, PCA Projection) - {group or 'all cohorts'}",
            "xaxis": {"title": "Principal Component 1 (Synthetic)"},
            "yaxis": {"title": "Principal Component 2 (Synthetic)"},
        },
    }

    interpretation = (
        f"K-means behavioural clustering (k={k}) partitioned {total_pts} participants across {len(features)} "
        f"features with an illustrative silhouette score of {silhouette}. "
        f"Largest subgroup is Cluster_0 ({cluster_sizes['Cluster_0']} participants, {cluster_profiles['Cluster_0']['label']}). "
        f"Note: This is an illustrative simulation."
    )

    return AnalysisResult(
        analysis_id=f"clust_{seed:08x}",
        analysis_name="cluster_participant_behaviour",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "n_clusters": k,
            "features_used": features,
            "total_participants": total_pts,
            "cluster_sizes": cluster_sizes,
            "cluster_profiles": cluster_profiles,
            "illustrative_silhouette_score": silhouette,
            "model_architecture_note": "Unsupervised K-Means clustering (mock illustrative interface)",
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "EXPLICIT NOTICE: No machine learning model was trained on participant data. "
            "Clusters and PCA coordinates are deterministic synthetic placeholders.",
        ],
        warnings=[],
    )


def predict_task_performance(
    params: PredictPerformanceParams | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Demonstrate the interface of a supervised prediction pipeline for a predefined outcome."""
    if params is None and kwargs:
        validated = PredictPerformanceParams(**kwargs)
    elif isinstance(params, PredictPerformanceParams):
        validated = params
    elif isinstance(params, dict):
        merged = {**params, **kwargs}
        validated = PredictPerformanceParams(**merged)
    else:
        validated = PredictPerformanceParams()

    seed = get_deterministic_seed("predict_task_performance", validated.model_dump())
    rng = random.Random(seed)

    filters = validated.model_dump()
    target = validated.target
    features = validated.features
    task_type = validated.task_type

    # Generate deterministic feature importances summing to 1.0
    raw_weights = [round(rng.uniform(0.2, 0.8), 3) for _ in features]
    total_w = sum(raw_weights)
    feature_importances = {
        feat: round(w / total_w, 3) for feat, w in zip(features, raw_weights)
    }

    # Illustrative classification performance metrics
    accuracy = round(rng.uniform(0.79, 0.86), 3)
    precision = round(rng.uniform(0.78, 0.85), 3)
    recall = round(rng.uniform(0.75, 0.84), 3)
    f1_score = round(2 * (precision * recall) / (precision + recall), 3)
    roc_auc = round(rng.uniform(0.85, 0.91), 3)

    confusion_matrix = {
        "true_positive": rng.randint(22, 26),
        "false_positive": rng.randint(3, 6),
        "true_negative": rng.randint(23, 27),
        "false_negative": rng.randint(4, 7),
    }

    # Horizontal bar chart of feature importances
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1])
    chart_data = {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "name": "Feature Importance (Gini / MDI)",
                "x": [val for _, val in sorted_features],
                "y": [feat for feat, _ in sorted_features],
                "marker": {"color": "#ab63fa"},
            }
        ],
        "layout": {
            "title": f"Predictive Feature Importance: Target = {target} ({task_type})",
            "xaxis": {"title": "Relative Importance (Mock)"},
            "yaxis": {"title": "Feature"},
        },
    }

    interpretation = (
        f"Supervised model interface for predicting '{target}' achieved illustrative accuracy of {accuracy} "
        f"(F1 = {f1_score}, ROC-AUC = {roc_auc}). Top predictive feature was "
        f"'{max(feature_importances, key=feature_importances.get)}' (weight = {max(feature_importances.values())})."
    )

    return AnalysisResult(
        analysis_id=f"pred_{seed:08x}",
        analysis_name="predict_task_performance",
        status="success",
        mock_data=True,
        filters_applied=filters,
        statistics={
            "target": target,
            "features": features,
            "model_pipeline": "StandardScaler + RandomForestClassifier (Mock Interface)",
            "cross_validation_folds": 5,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "roc_auc": roc_auc,
            },
            "confusion_matrix": confusion_matrix,
            "feature_importance": feature_importances,
        },
        chart_data=chart_data,
        interpretation=interpretation,
        limitations=[
            MOCK_DISCLAIMER_LIMITATION,
            "EXPLICIT NOTICE: No supervised model was trained or evaluated on participant data. "
            "All classification scores and confusion matrix counts are mock demonstration values.",
        ],
        warnings=[],
    )
