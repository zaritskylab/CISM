#!/usr/bin/env python3
"""SHAP-style PC attributions for the original two-cluster k-means analysis."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = ANALYSIS_DIR / "outputs" / "fanmod_patient_motif_clustering_NN_NP_tutorial_artifact"
OUTPUT = ROOT / "pc_label_association" / "shap"
N_COMPONENTS = 20
SEED = 0


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    matrix = pd.read_csv(ROOT / "patient_motif_matrix_cism_frequency.csv", index_col=0)
    assignments = pd.read_csv(ROOT / "selected_cluster_assignments.csv").set_index("patient_id")
    if matrix.shape != (38, 1000) or not matrix.index.is_unique or not matrix.index.isin(assignments.index).all():
        raise ValueError("Expected 38 aligned patients and 1,000 motifs from the tutorial artifact")
    classes = assignments.loc[matrix.index, "class_label"]
    if classes.value_counts().to_dict() != {"NN": 20, "NP": 18}:
        raise ValueError("Expected 20 NN and 18 NP patients")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Motif frequencies must be finite and nonnegative")
    return matrix, classes


def fit_clustering(matrix: pd.DataFrame) -> tuple[pd.DataFrame, KMeans, np.ndarray]:
    log_frequencies = np.log1p(matrix.to_numpy(dtype=float) * 1_000_000)
    scaled_motifs = StandardScaler().fit_transform(log_frequencies)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    scores = pca.fit_transform(scaled_motifs)
    frame = pd.DataFrame(scores, index=matrix.index, columns=[f"PC{i}" for i in range(1, N_COMPONENTS + 1)])
    clusters = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(scores)
    return frame, clusters, pca.explained_variance_ratio_


def nearest_centroid_margin(frame: pd.DataFrame, clusters: KMeans, nn_cluster: int,
                            np_cluster: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Positive signed distance to the centroid bisector favors the NP-majority cluster."""
    c_nn = clusters.cluster_centers_[nn_cluster]
    c_np = clusters.cluster_centers_[np_cluster]
    separation = np.linalg.norm(c_np - c_nn)
    if separation == 0:
        raise ValueError("The two cluster centroids coincide")
    coefficients = (c_np - c_nn) / separation
    intercept = (c_nn @ c_nn - c_np @ c_np) / (2 * separation)
    values = frame.to_numpy() @ coefficients + intercept
    direct = (np.sum((frame.to_numpy() - c_nn) ** 2, axis=1)
              - np.sum((frame.to_numpy() - c_np) ** 2, axis=1)) / (2 * separation)
    np.testing.assert_allclose(values, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(values > 0, clusters.labels_ == np_cluster)
    return coefficients, values, float(intercept)


def make_waterfall(explanations: shap.Explanation, index: int, patient_id: str,
                   class_label: str, cluster: int, margin: float) -> plt.Figure:
    shap.plots.waterfall(explanations[index], max_display=N_COMPONENTS, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10.2, 10.8)
    fig.subplots_adjust(left=.33, right=.94, top=.88, bottom=.11)
    fig.suptitle(f"Why {patient_id} was assigned to cluster {cluster}", x=.51, y=.98,
                 fontsize=17, weight="bold")
    fig.text(.05, .935, f"True class: {class_label} | cluster {cluster} is NP-majority | signed margin: {margin:+.2f}",
             fontsize=10)
    fig.text(.05, .045, "Red favors the NP-majority centroid; blue favors the NN-majority centroid.", fontsize=9)
    fig.text(.05, .025, "SHAP values explain the k-means distance margin, not an NN/NP probability or a classifier.", fontsize=9)
    return fig


def make_beeswarm(explanations: shap.Explanation) -> plt.Figure:
    plt.close("all")
    shap.plots.beeswarm(explanations, max_display=N_COMPONENTS, show=False,
                        plot_size=(10.8, 9.7), s=24)
    fig = plt.gcf()
    fig.subplots_adjust(left=.22, right=.90, top=.87, bottom=.12)
    fig.axes[0].set_xlabel("SHAP value (change in signed k-means margin)")
    fig.suptitle("PC contributions to k-means assignment", x=.49, y=.98,
                 fontsize=17, weight="bold")
    fig.text(.07, .94, "Each dot is one patient; right favors the NP-majority cluster, left the NN-majority cluster.",
             fontsize=10)
    fig.text(.07, .91, "Rows are ranked by mean absolute contribution across patients, not by PC number.", fontsize=9)
    fig.text(.07, .043, "Dot color indicates PC score (red = high, blue = low), not the NN/NP class.", fontsize=9)
    fig.text(.07, .022, "The margin is a geometric cluster score, not a predicted class probability.", fontsize=9)
    return fig


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrix, classes = load_data()
    frame, clusters, explained = fit_clustering(matrix)
    composition = pd.crosstab(pd.Series(clusters.labels_, name="cluster"),
                              pd.Series(classes.to_numpy(), name="class_label"))
    majority = composition.idxmax(axis=1)
    if set(majority) != {"NN", "NP"}:
        raise ValueError("The two clusters do not have different NN/NP majority classes")
    nn_cluster = int(majority[majority == "NN"].index[0])
    np_cluster = int(majority[majority == "NP"].index[0])
    coefficients, margins, intercept = nearest_centroid_margin(frame, clusters, nn_cluster, np_cluster)

    # This is an exact linear expression for the fitted two-centroid decision boundary.
    explainer = shap.LinearExplainer((coefficients, intercept), shap.maskers.Independent(frame))
    explanations = explainer(frame)
    np.testing.assert_allclose(explanations.base_values + explanations.values.sum(axis=1),
                               margins, rtol=1e-10, atol=1e-10)

    np_indices = np.flatnonzero(clusters.labels_ == np_cluster)
    median_margin = np.median(margins[np_indices])
    example_index = int(np_indices[np.argmin(np.abs(margins[np_indices] - median_margin))])
    patient_id = str(matrix.index[example_index])

    summary = {
        "method": "log1p(frequency x 1e6) -> motif StandardScaler -> PCA-20 -> KMeans(k=2)",
        "patients": len(frame),
        "pc_cumulative_variance_pct": float(explained.sum() * 100),
        "cluster_composition": {str(i): composition.loc[i].to_dict() for i in composition.index},
        "majority_label_by_cluster": {str(i): str(majority.loc[i]) for i in majority.index},
        "attribution_target": "signed distance to centroid bisector; positive favors NP-majority cluster",
        "waterfall_patient_id": patient_id,
        "waterfall_patient_true_class": str(classes.iloc[example_index]),
        "waterfall_patient_cluster": int(clusters.labels_[example_index]),
        "waterfall_patient_signed_margin": float(margins[example_index]),
    }
    (OUTPUT / "clustering_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    contributions = pd.DataFrame(explanations.values, index=frame.index, columns=frame.columns)
    contributions.insert(0, "signed_margin", margins)
    contributions.insert(0, "cluster_majority_label", [majority.loc[i] for i in clusters.labels_])
    contributions.insert(0, "cluster", clusters.labels_)
    contributions.insert(0, "class_label", classes.to_numpy())
    contributions.insert(0, "patient_id", frame.index)
    contributions.to_csv(OUTPUT / "pc_shap_values_margin.csv", index=False, float_format="%.8g")
    importance = pd.DataFrame({
        "PC": frame.columns,
        "variance_explained_pct": explained * 100,
        "mean_abs_shap_margin": np.abs(explanations.values).mean(axis=0),
    }).sort_values("mean_abs_shap_margin", ascending=False)
    importance.to_csv(OUTPUT / "pc_shap_importance.csv", index=False, float_format="%.8g")

    waterfall = make_waterfall(explanations, example_index, patient_id,
                               str(classes.iloc[example_index]), np_cluster, float(margins[example_index]))
    waterfall.savefig(OUTPUT / "pc_shap_waterfall.png", dpi=220, bbox_inches="tight")
    waterfall.savefig(OUTPUT / "pc_shap_waterfall.pdf", bbox_inches="tight")
    with PdfPages(OUTPUT / "pc_shap_figures.pdf") as pdf:
        pdf.savefig(waterfall, bbox_inches="tight")
        plt.close(waterfall)
        beeswarm = make_beeswarm(explanations)
        beeswarm.savefig(OUTPUT / "pc_shap_beeswarm.png", dpi=220, bbox_inches="tight")
        beeswarm.savefig(OUTPUT / "pc_shap_beeswarm.pdf", bbox_inches="tight")
        beeswarm.savefig(OUTPUT / "pc_shap_beeswarm_clean.png", dpi=220, bbox_inches="tight")
        beeswarm.savefig(OUTPUT / "pc_shap_beeswarm_clean.pdf", bbox_inches="tight")
        pdf.savefig(beeswarm, bbox_inches="tight")
        plt.close(beeswarm)

    (OUTPUT / "README.md").write_text(
        "# PC contributions to the original k-means clustering\n\n"
        "This is the original 38-patient log-frequency, motif-standardized PCA-20 and k-means (k=2) pipeline. "
        f"Cluster {nn_cluster} contains {int(composition.loc[nn_cluster, 'NN'])} NN and "
        f"{int(composition.loc[nn_cluster, 'NP'])} NP patients; cluster {np_cluster} contains "
        f"{int(composition.loc[np_cluster, 'NN'])} NN and {int(composition.loc[np_cluster, 'NP'])} NP patients. "
        "Cluster majority labels are assigned post hoc and were not used to fit PCA or k-means.\n\n"
        "The SHAP waterfall and beeswarm explain a continuous signed k-means margin: "
        "(squared distance to the NN-majority centroid minus squared distance to the NP-majority centroid) "
        "divided by twice the distance between centroids. Positive scores mean a patient is closer to the "
        "NP-majority centroid; negative scores mean closer to the NN-majority centroid. "
        "For two centroids this margin is exactly linear in the PC coordinates, so the independent-background "
        "LinearExplainer gives exact additive PC contributions to that margin. This is not a class probability "
        "and does not show biological effects of individual motifs. The waterfall patient is the member of the "
        "NP-majority cluster whose margin is closest to that cluster's median. Beeswarm red/blue indicates "
        "high/low PC score, not patient class. PC signs are arbitrary.\n",
        encoding="utf-8",
    )
    print(OUTPUT / "pc_shap_figures.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
