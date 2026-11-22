#!/usr/bin/env python3
"""Plot focused cluster views for one FANMOD+ clustering configuration."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
import sys
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = ANALYSIS_DIR
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fanmod_patient_motif_clustering_NN_NP_tutorial_artifact"
FIGURE_DIR = OUTPUT_DIR / "figures" / "focused"
CLASS_COLORS = {"NN": "#2F6C9E", "NP": "#D77A3D"}
CLUSTER_COLORS = {0: "#2A6FBB", 1: "#C45A3A"}
RANDOM_STATE = 0


def load_umap():
    extra = os.environ.get("UMAP_SITE_PACKAGES")
    if extra and extra not in sys.path:
        sys.path.append(extra)
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    return umap


def scatter_view(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, classes: pd.Series, title: str) -> None:
    markers = {0: "o", 1: "^"}
    for cluster in sorted(np.unique(labels)):
        for class_label in ["NN", "NP"]:
            mask = (labels == cluster) & (classes.to_numpy() == class_label)
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=58,
                color=CLASS_COLORS[class_label],
                marker=markers[int(cluster)],
                edgecolor="white",
                linewidth=0.7,
                label=f"{class_label}, cluster {cluster}",
            )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.axhline(0, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.5, zorder=0)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(OUTPUT_DIR / "patient_motif_matrix_cism_frequency.csv", index_col=0)
    assignments = pd.read_csv(OUTPUT_DIR / "selected_cluster_assignments.csv")[["patient_id", "class_label"]]
    assignments = assignments.set_index("patient_id").loc[matrix.index]

    log_freq = np.log1p(matrix.to_numpy(dtype=float) * 1_000_000)
    scaled = StandardScaler().fit_transform(log_freq)
    pca20 = PCA(n_components=20, random_state=RANDOM_STATE).fit_transform(scaled)
    labels = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit_predict(pca20)

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    tsne = TSNE(
        n_components=2,
        perplexity=8,
        init="pca",
        learning_rate="auto",
        max_iter=750,
        random_state=RANDOM_STATE,
    ).fit_transform(scaled)

    umap_module = load_umap()
    if umap_module is not None:
        umap_coords = umap_module.UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.1,
            n_epochs=100,
            random_state=RANDOM_STATE,
        ).fit_transform(scaled)
    else:
        umap_coords = None

    views = [("PCA2 display", pca2), ("UMAP display", umap_coords), ("t-SNE display", tsne)]
    views = [(name, coords) for name, coords in views if coords is not None]
    fig, axes = plt.subplots(1, len(views), figsize=(5.0 * len(views), 4.2), squeeze=False)
    for ax, (name, coords) in zip(axes.ravel(), views):
        scatter_view(ax, coords, labels, assignments["class_label"], name)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS["NN"], markeredgecolor="white", markersize=8, label="NN"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS["NP"], markeredgecolor="white", markersize=8, label="NP"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555", linestyle="none", markersize=8, label="cluster 0"),
        Line2D([0], [0], marker="^", color="#555555", markerfacecolor="#555555", linestyle="none", markersize=8, label="cluster 1"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("log(frequency) -> PCA-20 -> k-means clusters", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"log_frequency_pca20_kmeans_views.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    rows = []
    for patient_id, class_label, cluster in zip(matrix.index, assignments["class_label"], labels):
        rows.append({"patient_id": patient_id, "class_label": class_label, "cluster": int(cluster)})
    out = pd.DataFrame(rows)
    out.to_csv(FIGURE_DIR / "log_frequency_pca20_kmeans_assignments.csv", index=False)
    print(FIGURE_DIR / "log_frequency_pca20_kmeans_views.png")
    print(pd.crosstab(out["cluster"], out["class_label"]).to_string())


if __name__ == "__main__":
    main()
