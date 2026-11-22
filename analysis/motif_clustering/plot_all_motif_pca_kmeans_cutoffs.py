#!/usr/bin/env python3
"""Compare PCA-20 and PCA-25 k-means on all observed NN/NP motif frequencies."""

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


ROOT = ANALYSIS_DIR
INPUT_DIR = ROOT / "outputs" / "fanmod_patient_motif_clustering_NN_NP_tutorial_artifact_all_motifs"
LABELS_CSV = ROOT.parent / "melanoma_tutorials" / "pet_data" / "Melanoma" / "patient_class.csv"
OUTPUT_DIR = INPUT_DIR / "pca_cutoff_review"
CLASS_COLORS = {"NN": "#2F6C9E", "NP": "#D77A3D"}
GREY = "#879196"
CUTOFFS = (20, 25)


def load_pca() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    matrix = pd.read_csv(INPUT_DIR / "patient_motif_matrix_cism_frequency.csv", index_col=0)
    labels = pd.read_csv(LABELS_CSV, header=None, names=["patient_id", "class_label"]).set_index("patient_id")
    if matrix.shape != (38, 59359):
        raise ValueError(f"Unexpected all-motif matrix shape: {matrix.shape}")
    if not matrix.index.isin(labels.index).all():
        raise ValueError("Missing patient disease label")
    classes = labels.loc[matrix.index, "class_label"].to_numpy()
    if set(classes) != {"NN", "NP"}:
        raise ValueError("Expected only NN and NP patients")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Motif frequencies must be finite and nonnegative")
    scaled = StandardScaler().fit_transform(np.log1p(values * 1_000_000))
    pca = PCA(n_components=matrix.shape[0] - 1, svd_solver="full")
    scores = pca.fit_transform(scaled)
    return matrix, classes, scores, pca.explained_variance_ratio_


def evaluate(scores: np.ndarray, classes: np.ndarray, cutoff: int, explained: np.ndarray,
             reference: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    model = KMeans(n_clusters=2, n_init=10, random_state=0).fit(scores[:, :cutoff])
    clusters = model.labels_
    if reference is not None and np.count_nonzero(clusters == reference) < np.count_nonzero(clusters != reference):
        clusters = 1 - clusters
    counts = pd.crosstab(pd.Series(classes, name="Disease state"), pd.Series(clusters, name="Cluster"))
    counts = counts.reindex(index=["NN", "NP"], columns=[0, 1], fill_value=0)
    mapping = {cluster: counts[cluster].idxmax() for cluster in (0, 1)}
    predicted = np.array([mapping[cluster] for cluster in clusters])
    metrics = {
        "cutoff": cutoff,
        "cumulative_variance_pct": float(explained[:cutoff].sum() * 100),
        "cluster_0_size": int((clusters == 0).sum()),
        "cluster_1_size": int((clusters == 1).sum()),
        "cluster_0_NN": int(counts.loc["NN", 0]),
        "cluster_0_NP": int(counts.loc["NP", 0]),
        "cluster_1_NN": int(counts.loc["NN", 1]),
        "cluster_1_NP": int(counts.loc["NP", 1]),
        "posthoc_majority_accuracy": float((predicted == classes).mean()),
        "posthoc_majority_balanced_accuracy": float(balanced_accuracy_score(classes, predicted)),
        "class_ari": float(adjusted_rand_score(classes, clusters)),
        "kmeans_inertia": float(model.inertia_),
    }
    return clusters, metrics


def draw_pipeline(ax: plt.Axes, cutoff: int, n_motifs: int, explained_pct: float,
                  transform_label: str = "log1p(frequency\nx 10^6)") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-.04, .94, "A", fontsize=13, weight="bold", va="top")
    ax.text(0, .96, "Unfiltered patient motif clustering", fontsize=14, weight="bold", va="top")
    boxes = [
        ("Patient x motif\n38 x {:,}".format(n_motifs), "#ECF0F3"),
        (transform_label, "#ECF0F3"),
        ("Standardize\neach motif", "#ECF0F3"),
        (f"PCA\n{cutoff} components", "#ECF0F3"),
        ("k-means\nk = 2", "#E6F2EF"),
    ]
    width = .16
    starts = np.linspace(.015, .825, len(boxes))
    for index, ((label, face), x) in enumerate(zip(boxes, starts)):
        ax.add_patch(FancyBboxPatch((x, .20), width, .52, boxstyle="round,pad=0.005,rounding_size=0.012",
                                     facecolor=face, edgecolor="#D5DFE3", linewidth=.8))
        ax.text(x + width / 2, .46, label, ha="center", va="center", fontsize=9.3, weight="bold" if index == 4 else "normal")
        if index < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + width + .008, .46), (starts[index + 1] - .010, .46),
                                         arrowstyle="-|>", mutation_scale=16, color="#627078", linewidth=1))
    ax.text(.015, .06, f"All {n_motifs:,} observed motifs retained; {cutoff} PCs explain {explained_pct:.1f}% of standardized matrix variance.", fontsize=9.4)


def draw_composition(ax: plt.Axes, metrics: dict) -> None:
    ax.axis("off")
    ax.text(-.08, 1.10, "B", fontsize=13, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0, 1.05, "Cluster composition", fontsize=12, weight="bold", transform=ax.transAxes)
    x = (.02, .21, .59, .97)
    rows = (
        ("NN", .43, .66, "#DCEBF6", CLASS_COLORS["NN"]),
        ("NP", .20, .43, "#F9E6D7", CLASS_COLORS["NP"]),
    )
    for cluster in (0, 1):
        left, right = x[cluster + 1], x[cluster + 2]
        ax.add_patch(Rectangle((left, .66), right - left, .19, transform=ax.transAxes,
                               facecolor="#E8EEF1", edgecolor="#293038", linewidth=1.1))
        ax.text((left + right) / 2, .755, f"Cluster {cluster}", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, weight="bold", color="#1F2933")
    for disease, bottom, top, face, color in rows:
        for column in range(3):
            left, right = x[column], x[column + 1]
            ax.add_patch(Rectangle((left, bottom), right - left, top - bottom,
                                   transform=ax.transAxes, facecolor=face,
                                   edgecolor=color, linewidth=1.0))
            label = disease if column == 0 else str(metrics[f"cluster_{column - 1}_{disease}"])
            ax.text((left + right) / 2, (bottom + top) / 2, label, transform=ax.transAxes,
                    ha="center", va="center", fontsize=14 if column else 12,
                    weight="bold", color=color)
    ax.text(0, .07, f"Post-hoc majority accuracy: {metrics['posthoc_majority_accuracy']:.3f}", transform=ax.transAxes, fontsize=9.5)


def draw_scatter(ax: plt.Axes, scores: np.ndarray, classes: np.ndarray, clusters: np.ndarray, explained: np.ndarray,
                 limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    for cluster, marker in ((0, "o"), (1, "^")):
        for label in ("NN", "NP"):
            mask = (clusters == cluster) & (classes == label)
            ax.scatter(scores[mask, 0], scores[mask, 1], marker=marker, s=54,
                       color=CLASS_COLORS[label], edgecolor="white", linewidth=.65, alpha=.92, zorder=3)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.axhline(0, color="#E2E7E9", linewidth=.8, zorder=0)
    ax.axvline(0, color="#E2E7E9", linewidth=.8, zorder=0)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.tick_params(labelsize=8)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=CLASS_COLORS["NN"], label="NN", markersize=7),
        Line2D([], [], marker="o", linestyle="none", color=CLASS_COLORS["NP"], label="NP", markersize=7),
        Line2D([], [], marker="o", linestyle="none", color="#666666", label="Cluster 0", markersize=7),
        Line2D([], [], marker="^", linestyle="none", color="#666666", label="Cluster 1", markersize=7),
    ], ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.13))


def draw_bars(ax: plt.Axes, heights: np.ndarray, cutoff: int, ylabel: str, title: str) -> None:
    pcs = np.arange(1, cutoff + 1)
    ax.bar(pcs, heights[:cutoff], color=GREY, width=.74, edgecolor="white", linewidth=.5, zorder=2)
    ax.set_xlim(.4, cutoff + .6)
    ax.set_ylim(0, max(heights[:cutoff]) * 1.14)
    ax.set_xticks(sorted({1, cutoff, *range(5, cutoff + 1, 5)}))
    ax.set_xlabel("Principal component", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, weight="bold", pad=9)
    ax.grid(axis="y", color="#E3E8EA", linewidth=.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def make_figure(n_motifs: int, cutoff: int, scores: np.ndarray, classes: np.ndarray, clusters: np.ndarray,
                explained: np.ndarray, r: np.ndarray, metrics: dict,
                transform_label: str = "log1p(frequency\nx 10^6)") -> plt.Figure:
    fig = plt.figure(figsize=(12.6, 9.4), facecolor="white")
    grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 2.85, 2.3], left=.075, right=.95,
                           top=.97, bottom=.095, hspace=.31)
    draw_pipeline(fig.add_subplot(grid[0]), cutoff, n_motifs, metrics["cumulative_variance_pct"], transform_label)
    mid = grid[1].subgridspec(1, 2, width_ratios=[.88, 1.12], wspace=.16)
    draw_composition(fig.add_subplot(mid[0]), metrics)
    scatter = fig.add_subplot(mid[1])
    xpad = np.ptp(scores[:, 0]) * .07
    ypad = np.ptp(scores[:, 1]) * .08
    limits = ((scores[:, 0].min() - xpad, scores[:, 0].max() + xpad),
              (scores[:, 1].min() - ypad, scores[:, 1].max() + ypad))
    draw_scatter(scatter, scores, classes, clusters, explained, limits)
    bottom = grid[2].subgridspec(1, 2, wspace=.23)
    var_ax = fig.add_subplot(bottom[0])
    var_ax.text(-.08, 1.18, "C", fontsize=13, weight="bold", transform=var_ax.transAxes)
    draw_bars(var_ax, explained * 100, cutoff, "Motif variance explained (%)", "Variance by PC")
    draw_bars(fig.add_subplot(bottom[1]), np.abs(r), cutoff, "|Pearson r| with NN/NP", "Disease-state association by PC")
    fig.text(.075, .043, "NN/NP labels were used only after PCA and k-means. Bar heights on the right are descriptive, not validated prediction.", fontsize=9)
    fig.text(.075, .022, "Cluster numbers are arbitrary; colors represent true disease state and marker shapes represent cluster assignment.", fontsize=9)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix, classes, scores, explained = load_pca()
    patient_ids = matrix.index.to_numpy()
    y = (classes == "NP").astype(float)
    r = ((y - y.mean()) / np.linalg.norm(y - y.mean())) @ (scores / np.linalg.norm(scores, axis=0))
    component_metrics = pd.DataFrame({"PC": np.arange(1, len(explained) + 1),
                                      "variance_explained_pct": explained * 100,
                                      "cumulative_variance_pct": np.cumsum(explained) * 100,
                                      "point_biserial_r": r})
    component_metrics.to_csv(OUTPUT_DIR / "component_metrics.csv", index=False, float_format="%.8g")
    pd.DataFrame(scores, index=patient_ids, columns=[f"PC{i}" for i in range(1, scores.shape[1] + 1)]).to_csv(
        OUTPUT_DIR / "patient_pca_scores.csv", index_label="patient_id", float_format="%.8g")

    all_metrics = []
    all_clusters = {}
    with PdfPages(OUTPUT_DIR / "pca20_vs_pca25_kmeans_figures.pdf") as pdf:
        for cutoff in CUTOFFS:
            clusters, metrics = evaluate(scores, classes, cutoff, explained, reference=all_clusters.get(20))
            all_clusters[cutoff] = clusters
            all_metrics.append(metrics)
            pd.DataFrame({"patient_id": patient_ids, "class_label": classes, "cluster": clusters,
                          "PC1": scores[:, 0], "PC2": scores[:, 1]}).to_csv(
                OUTPUT_DIR / f"pca{cutoff}_kmeans_assignments.csv", index=False, float_format="%.8g")
            fig = make_figure(matrix.shape[1], cutoff, scores, classes, clusters, explained, r, metrics)
            fig.savefig(OUTPUT_DIR / f"pca{cutoff}_kmeans_reference_panels.png", dpi=220)
            fig.savefig(OUTPUT_DIR / f"pca{cutoff}_kmeans_reference_panels.pdf")
            pdf.savefig(fig)
            plt.close(fig)

    comparison = pd.DataFrame(all_metrics)
    comparison.to_csv(OUTPUT_DIR / "pca20_vs_pca25_kmeans_comparison.csv", index=False, float_format="%.8g")
    between_cutoffs_ari = float(adjusted_rand_score(all_clusters[20], all_clusters[25]))
    details = {
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "transform": "log1p(frequency * 1_000_000)",
        "scaling": "StandardScaler per motif across patients",
        "pca": "full SVD; first 20 or 25 components",
        "kmeans": "k=2, n_init=10, random_state=0; lowest-inertia run retained",
        "cluster_id_alignment": "PCA-25 cluster IDs aligned to PCA-20 by patient membership, without disease labels",
        "cluster_ari_between_cutoffs": between_cutoffs_ari,
    }
    (OUTPUT_DIR / "pca20_vs_pca25_kmeans_methods.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    singleton_features = int(((matrix.to_numpy(dtype=float) > 0).sum(axis=0) == 1).sum())
    report = [
        "# All-motif PCA + k-means: 20 versus 25 components",
        "",
        "## Methods",
        "",
        f"The melanoma-tutorial-derived matrix has {matrix.shape[0]} NN/NP patients and {matrix.shape[1]:,} observed motif columns. "
        "No motif prevalence or top-variance filter was applied. Frequencies were transformed with "
        "log1p(frequency x 10^6), each motif was standardized across patients, and exact PCA was fitted "
        "without disease labels. K-means used k=2, n_init=10, and random_state=0 on the first 20 or 25 "
        "PC scores; the lowest-inertia initialization was retained. Labels were used only for post-hoc "
        "annotation and point-biserial Pearson r. PCA-25 cluster IDs were aligned to PCA-20 by patient "
        "membership only, without disease labels.",
        "",
        "## Results",
        "",
        f"- PCA-20 explains {all_metrics[0]['cumulative_variance_pct']:.2f}% of standardized motif variance; "
        f"PCA-25 explains {all_metrics[1]['cumulative_variance_pct']:.2f}%.",
        f"- Between-cutoff cluster agreement is ARI={between_cutoffs_ari:.3f}; this compares clusterings, not disease labels.",
        *[f"- PCA-{m['cutoff']} cluster 0: {m['cluster_0_NN']} NN, {m['cluster_0_NP']} NP; "
          f"cluster 1: {m['cluster_1_NN']} NN, {m['cluster_1_NP']} NP. "
          f"Post-hoc accuracy={m['posthoc_majority_accuracy']:.3f}, "
          f"balanced accuracy={m['posthoc_majority_balanced_accuracy']:.3f}." for m in all_metrics],
        "- Post-hoc accuracy and balanced accuracy are descriptive in-sample values, not validated predictive accuracy.",
        f"- {singleton_features:,} motifs occur in only one patient. Standardizing rare motif columns gives "
        "each nonconstant column unit variance; this can affect distances, but its effect on this "
        "partition was not isolated in the present analysis.",
        "",
        "## Figures",
        "",
        "- `pca_scree.pdf`: component-variance and cumulative-variance review.",
        "- `pca20_kmeans_reference_panels.pdf`: PCA-20 pipeline, composition, PC1-PC2 projection, and per-PC bars.",
        "- `pca25_kmeans_reference_panels.pdf`: the corresponding PCA-25 panels.",
        "",
        "Cluster numbers are arbitrary. The PC1-PC2 scatter displays the first two axes, while "
        "k-means used all 20 or 25 retained PC axes. The r bars show association with true class, "
        "not the basis for PCA or clustering.",
    ]
    (OUTPUT_DIR / "pca20_vs_pca25_kmeans_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"ARI between cutoff assignments: {between_cutoffs_ari:.3f}")
    print(OUTPUT_DIR / "pca20_vs_pca25_kmeans_figures.pdf")


if __name__ == "__main__":
    main()
