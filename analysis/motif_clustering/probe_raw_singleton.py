#!/usr/bin/env python3
"""Probe the raw-frequency k-means singleton and refit after its removal."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis.motif_clustering.compare_all_motif_log_raw_kmeans import INPUT_DIR, LABELS_CSV


OUTPUT_DIR = INPUT_DIR / "raw_singleton_probe"
N_INIT = 5000
PROBE_STARTS = 2000
CUTOFFS = (20, 25)
COLORS = {"NN": "#2F6C9E", "NP": "#D77A3D"}


def fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = StandardScaler().fit_transform(values)
    pca = PCA(n_components=values.shape[0] - 1, svd_solver="full")
    scores = pca.fit_transform(scaled)
    squared_radius = np.sum((scaled - scaled.mean(axis=0)) ** 2, axis=1)
    return scores, pca.explained_variance_ratio_, squared_radius


def summarize(patient_ids: np.ndarray, classes: np.ndarray, scores: np.ndarray,
              explained: np.ndarray, state: str, cutoff: int) -> tuple[np.ndarray, dict]:
    model = KMeans(n_clusters=2, n_init=N_INIT, random_state=0).fit(scores[:, :cutoff])
    labels = model.labels_.copy()
    if (labels == 0).sum() > (labels == 1).sum():
        labels = 1 - labels
    counts = pd.crosstab(pd.Series(classes), pd.Series(labels)).reindex(
        index=["NN", "NP"], columns=[0, 1], fill_value=0)
    mapping = {cluster: counts[cluster].idxmax() for cluster in (0, 1)}
    predicted = np.array([mapping[x] for x in labels])
    metrics = {
        "sample": state, "n_patients": len(patient_ids), "PCs": cutoff,
        "variance_explained_pct": explained[:cutoff].sum() * 100,
        "cluster_0_size": int((labels == 0).sum()),
        "cluster_1_size": int((labels == 1).sum()),
        "cluster_0_NN": int(counts.loc["NN", 0]), "cluster_0_NP": int(counts.loc["NP", 0]),
        "cluster_1_NN": int(counts.loc["NN", 1]), "cluster_1_NP": int(counts.loc["NP", 1]),
        "posthoc_accuracy": float((predicted == classes).mean()),
        "posthoc_balanced_accuracy": float(balanced_accuracy_score(classes, predicted)),
        "class_ari": float(adjusted_rand_score(classes, labels)),
        "kmeans_inertia": float(model.inertia_),
    }
    pd.DataFrame({"patient_id": patient_ids, "class_label": classes, "cluster": labels,
                  "PC1": scores[:, 0], "PC2": scores[:, 1]}).to_csv(
        OUTPUT_DIR / f"{state}_pca{cutoff}_assignments.csv", index=False, float_format="%.8g")
    return labels, metrics


def probe_local_minima(scores: np.ndarray, classes: np.ndarray, best_inertia: float) -> dict:
    rows = []
    best_non_singleton: tuple[float, np.ndarray] | None = None
    for seed in range(PROBE_STARTS):
        model = KMeans(n_clusters=2, n_init=1, random_state=seed).fit(scores[:, :25])
        smallest = int(np.bincount(model.labels_, minlength=2).min())
        rows.append((seed, smallest, float(model.inertia_)))
        if smallest > 1 and (best_non_singleton is None or model.inertia_ < best_non_singleton[0]):
            best_non_singleton = (float(model.inertia_), model.labels_.copy())
    pd.DataFrame(rows, columns=["seed", "smaller_cluster_size", "inertia"]).to_csv(
        OUTPUT_DIR / "raw_pca25_single_start_landscape.csv", index=False, float_format="%.8g")
    if best_non_singleton is None:
        return {"probe_starts": PROBE_STARTS, "non_singleton_starts": 0}
    inertia, labels = best_non_singleton
    counts = pd.crosstab(pd.Series(classes), pd.Series(labels)).reindex(
        index=["NN", "NP"], columns=[0, 1], fill_value=0)
    return {
        "probe_starts": PROBE_STARTS,
        "non_singleton_starts": int(sum(row[1] > 1 for row in rows)),
        "best_non_singleton_inertia": inertia,
        "best_unconstrained_inertia": best_inertia,
        "non_singleton_inertia_penalty_pct": (inertia / best_inertia - 1) * 100,
        "non_singleton_cluster_sizes": [int(x) for x in np.bincount(labels, minlength=2)],
        "non_singleton_composition": {str(c): {d: int(counts.loc[d, c]) for d in ("NN", "NP")}
                                      for c in (0, 1)},
    }


def plot(samples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    for ax, (name, patient_ids, scores, classes, labels, explained, metrics) in zip(axes, samples):
        for cluster, marker in ((0, "o"), (1, "^")):
            for disease in ("NN", "NP"):
                selected = (labels == cluster) & (classes == disease)
                if not selected.any():
                    continue
                ax.scatter(scores[selected, 0], scores[selected, 1], color=COLORS[disease],
                           marker=marker, s=55, edgecolor="white", linewidth=.7,
                           label=f"{disease}, cluster {cluster}")
        singleton = np.flatnonzero(labels == 0)
        if len(singleton) == 1:
            i = int(singleton[0])
            ax.annotate(str(patient_ids[i]), xy=scores[i, :2], xytext=(-12, -17),
                        textcoords="offset points", ha="right", fontsize=9, weight="bold")
        ax.axhline(0, color="#DCE3E6", lw=.8)
        ax.axvline(0, color="#DCE3E6", lw=.8)
        ax.set_title(f"{name}: {metrics['cluster_0_size']} / {metrics['cluster_1_size']} patients")
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Raw-frequency k-means, 25 PCs, 5,000 starts; PCA refitted after exclusion", fontsize=13)
    fig.savefig(OUTPUT_DIR / "raw_pca25_before_after_exclusion.png", dpi=220)
    fig.savefig(OUTPUT_DIR / "raw_pca25_before_after_exclusion.pdf")
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(OUTPUT_DIR / "raw_pca25_before_after_exclusion.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(INPUT_DIR / "patient_motif_matrix_cism_frequency.csv", index_col=0)
    label_table = pd.read_csv(LABELS_CSV, header=None, names=["patient_id", "class_label"]).set_index("patient_id")
    patient_ids = matrix.index.to_numpy()
    classes = label_table.loc[matrix.index, "class_label"].to_numpy()
    values = matrix.to_numpy(dtype=float)
    if matrix.shape != (38, 59359) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Unexpected matrix or invalid frequencies")

    scores, explained, radius = fit(values)
    full: dict[int, tuple[np.ndarray, dict]] = {}
    for cutoff in CUTOFFS:
        full[cutoff] = summarize(patient_ids, classes, scores, explained, "full_raw", cutoff)
    singleton = patient_ids[full[25][0] == 0]
    if len(singleton) != 1 or not np.array_equal(full[20][0], full[25][0]):
        raise ValueError("Expected the same singleton at both cutoffs")
    excluded = str(singleton[0])
    keep = patient_ids != excluded
    print(f"Singleton: {excluded} ({classes[~keep][0]})", flush=True)
    print("Probing 2,000 individual k-means starts", flush=True)
    landscape = probe_local_minima(scores, classes, full[25][1]["kmeans_inertia"])

    reduced_scores, reduced_explained, reduced_radius = fit(values[keep])
    reduced: dict[int, tuple[np.ndarray, dict]] = {}
    for cutoff in CUTOFFS:
        reduced[cutoff] = summarize(patient_ids[keep], classes[keep], reduced_scores,
                                    reduced_explained, "without_singleton", cutoff)
    next_singleton = patient_ids[keep][reduced[25][0] == 0]
    if len(next_singleton) != 1:
        raise ValueError("Expected one singleton after exclusion")
    rows = [full[cutoff][1] for cutoff in CUTOFFS] + [reduced[cutoff][1] for cutoff in CUTOFFS]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison.csv", index=False, float_format="%.8g")
    pd.DataFrame({"patient_id": patient_ids, "class_label": classes,
                  "standardized_squared_radius": radius,
                  "radius_rank_descending": pd.Series(radius).rank(ascending=False, method="min").astype(int)}).to_csv(
        OUTPUT_DIR / "full_sample_outlier_diagnostics.csv", index=False, float_format="%.8g")
    overlap_ari = float(adjusted_rand_score(full[25][0][keep], reduced[25][0]))
    plot([("All 38", patient_ids, scores, classes, full[25][0], explained, full[25][1]),
          ("Without " + excluded, patient_ids[keep], reduced_scores, classes[keep], reduced[25][0], reduced_explained,
           reduced[25][1])])
    report = [
        "# Raw-frequency singleton probe",
        "",
        f"Singleton: **{excluded} ({classes[~keep][0]})**. All runs use raw motif frequencies, per-motif "
        f"StandardScaler, full PCA refitted on the available patients, and k-means k=2 with {N_INIT} "
        "k-means++ initializations (random_state=0). All 59,359 motif columns remain in both samples; "
        "constant columns after exclusion contribute zero.",
        "",
        "| Sample | PCs | Variance explained | Cluster 0 (NN/NP) | Cluster 1 (NN/NP) | "
        "Post-hoc accuracy | Balanced accuracy | Class ARI | Inertia |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['sample']} | {row['PCs']} | {row['variance_explained_pct']:.1f}% | "
            f"{row['cluster_0_NN']}/{row['cluster_0_NP']} | {row['cluster_1_NN']}/{row['cluster_1_NP']} | "
            f"{row['posthoc_accuracy']:.3f} | {row['posthoc_balanced_accuracy']:.3f} | "
            f"{row['class_ari']:.3f} | {row['kmeans_inertia']:.0f} |"
        )
    report += [
        "",
        f"{excluded} has standardized squared radius {radius[~keep][0]:.1f}, rank "
        f"{int(pd.Series(radius).rank(ascending=False, method='min')[~keep].iloc[0])} of 38 (1 = furthest).",
        f"After removing {excluded}, the new singleton is **{next_singleton[0]} "
        f"({classes[patient_ids == next_singleton[0]][0]})**.",
        f"PCA-25 adjusted Rand agreement among the 37 shared patients before versus after exclusion: "
        f"{overlap_ari:.3f}.",
        "",
        f"Among {landscape['probe_starts']} independent single-start raw/PCA-25 fits, "
        f"{landscape['non_singleton_starts']} yielded a non-singleton split.",
    ]
    if landscape["non_singleton_starts"]:
        report += [
            f"The best non-singleton start gave sizes {landscape['non_singleton_cluster_sizes']} and "
            f"inertia {landscape['best_non_singleton_inertia']:.0f}, "
            f"{landscape['non_singleton_inertia_penalty_pct']:.3f}% above the best 5,000-start "
            "unconstrained fit. This is an alternative local minimum, not evidence of a lower global minimum.",
            f"Its composition was {landscape['non_singleton_composition']}.",
        ]
    report += [
        "",
        "Inertia cannot be compared directly between the 38- and 37-patient fits, because the data, "
        "standardization, and PCA basis change. A lower inertia is better only within the same dataset, "
        "scaling, and PCA cutoff. The 5,000-start result is the best found, not a proof of global optimality. "
        "Post-hoc accuracy is descriptive, not validated prediction.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Alternative starts: {landscape}", flush=True)
    print(OUTPUT_DIR / "raw_pca25_before_after_exclusion.pdf", flush=True)


if __name__ == "__main__":
    main()
