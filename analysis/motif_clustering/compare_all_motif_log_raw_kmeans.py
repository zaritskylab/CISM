#!/usr/bin/env python3
"""Compare log and untransformed all-motif frequencies at two PCA cutoffs."""

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
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Bbox
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from analysis.motif_clustering.plot_all_motif_pca_kmeans_cutoffs import INPUT_DIR, LABELS_CSV, make_figure


OUTPUT_DIR = INPUT_DIR / "log_vs_raw_kmeans_5000_starts"
CUTOFFS = (20, 25)
N_INIT = 5000
plt.rcParams["svg.fonttype"] = "none"


def save_editable_assets(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.svg")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = ("pipeline", "composition", "scatter", "variance", "class_association")
    if len(fig.axes) != len(panels):
        raise ValueError(f"Expected {len(panels)} editable panels, found {len(fig.axes)}")
    for name, ax in zip(panels, fig.axes):
        bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        padded = Bbox.from_extents(bounds.x0 - .12, bounds.y0 - .12,
                                   bounds.x1 + .12, bounds.y1 + .12)
        fig.savefig(OUTPUT_DIR / f"{stem}_{name}.svg", bbox_inches=padded)


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
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
    return matrix, classes


def fit_pca(values: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.log1p(values * 1_000_000) if transform == "log" else values
    scaled = StandardScaler().fit_transform(transformed)
    pca = PCA(n_components=values.shape[0] - 1, svd_solver="full")
    scores = pca.fit_transform(scaled)
    return scores, pca.explained_variance_ratio_


def evaluate(scores: np.ndarray, classes: np.ndarray, cutoff: int, explained: np.ndarray,
             reference: np.ndarray | None) -> tuple[np.ndarray, dict]:
    model = KMeans(n_clusters=2, n_init=N_INIT, random_state=0).fit(scores[:, :cutoff])
    clusters = model.labels_.copy()
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix, classes = load_data()
    values = matrix.to_numpy(dtype=float)
    y = (classes == "NP").astype(float)
    patient_ids = matrix.index.to_numpy()
    rows: list[dict] = []
    assignments: dict[tuple[str, int], np.ndarray] = {}

    with PdfPages(OUTPUT_DIR / "log_vs_raw_kmeans_5000_starts.pdf") as pdf:
        for transform in ("log", "raw"):
            print(f"Fitting PCA for {transform} frequencies", flush=True)
            scores, explained = fit_pca(values, transform)
            r = ((y - y.mean()) / np.linalg.norm(y - y.mean())) @ (scores / np.linalg.norm(scores, axis=0))
            pd.DataFrame({"PC": np.arange(1, len(explained) + 1),
                          "variance_explained_pct": explained * 100,
                          "cumulative_variance_pct": np.cumsum(explained) * 100,
                          "point_biserial_r": r}).to_csv(
                OUTPUT_DIR / f"{transform}_component_metrics.csv", index=False, float_format="%.8g")
            reference = None
            for cutoff in CUTOFFS:
                print(f"Fitting k-means for {transform}, PCA-{cutoff}, n_init={N_INIT}", flush=True)
                clusters, metrics = evaluate(scores, classes, cutoff, explained, reference)
                if reference is None:
                    reference = clusters
                assignments[(transform, cutoff)] = clusters
                rows.append({"transform": transform, **metrics})
                pd.DataFrame({"patient_id": patient_ids, "class_label": classes, "cluster": clusters,
                              "PC1": scores[:, 0], "PC2": scores[:, 1]}).to_csv(
                    OUTPUT_DIR / f"{transform}_pca{cutoff}_assignments.csv", index=False, float_format="%.8g")
                label = "log1p(frequency\nx 10^6)" if transform == "log" else "Raw frequency\n(no log)"
                fig = make_figure(matrix.shape[1], cutoff, scores, classes, clusters, explained, r, metrics,
                                  transform_label=label)
                fig.savefig(OUTPUT_DIR / f"{transform}_pca{cutoff}_panels.png", dpi=220)
                fig.savefig(OUTPUT_DIR / f"{transform}_pca{cutoff}_panels.pdf")
                save_editable_assets(fig, f"{transform}_pca{cutoff}_panels")
                pdf.savefig(fig)
                plt.close(fig)
                print(f"  cluster sizes: {metrics['cluster_0_size']} / {metrics['cluster_1_size']}", flush=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False, float_format="%.8g")
    agreements = {transform: float(adjusted_rand_score(assignments[(transform, 20)],
                                                        assignments[(transform, 25)]))
                  for transform in ("log", "raw")}
    agreements["log_vs_raw_pca20"] = float(adjusted_rand_score(assignments[("log", 20)],
                                                                assignments[("raw", 20)]))
    agreements["log_vs_raw_pca25"] = float(adjusted_rand_score(assignments[("log", 25)],
                                                                assignments[("raw", 25)]))
    details = {
        "source": "melanoma tutorial FANMOD+ artifact; all observed NN/NP motif columns",
        "matrix_shape": list(matrix.shape),
        "transform_log": "log1p(frequency * 1_000_000)",
        "transform_raw": "frequency; no log transform",
        "scaling": "StandardScaler, separately fit per motif in each branch",
        "pca": "exact full SVD; first 20 or 25 components",
        "kmeans": f"k=2, n_init={N_INIT}, random_state=0; lowest-inertia run retained",
        "cluster_id_alignment": "within each transform, PCA-25 IDs aligned to PCA-20 membership without disease labels",
        "adjusted_rand_agreements": agreements,
        "note": "Post-hoc accuracy is in-sample and label-derived; neither cluster balance nor accuracy was optimized.",
    }
    (OUTPUT_DIR / "methods.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    report = [
        "# All-motif k-means: log versus raw frequencies",
        "",
        "38 NN/NP patients by 59,359 observed motifs. Both branches standardize each motif, fit exact PCA, "
        f"and use k-means (k=2, n_init={N_INIT}, random_state=0). The raw branch omits only the log step. "
        "The first 20 and 25 PCs were evaluated separately. Disease labels were not used for PCA or clustering.",
        "",
        "| Input | PCs | PCA variance | Cluster 0 (NN/NP) | Cluster 1 (NN/NP) | Post-hoc accuracy | Balanced accuracy | Class ARI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['transform']} | {row['cutoff']} | {row['cumulative_variance_pct']:.1f}% | "
            f"{row['cluster_0_NN']}/{row['cluster_0_NP']} | {row['cluster_1_NN']}/{row['cluster_1_NP']} | "
            f"{row['posthoc_majority_accuracy']:.3f} | {row['posthoc_majority_balanced_accuracy']:.3f} | "
            f"{row['class_ari']:.3f} |"
        )
    report += [
        "",
        f"Within-input agreement (PCA-20 vs PCA-25): log ARI={agreements['log']:.3f}; "
        f"raw ARI={agreements['raw']:.3f}.",
        f"Between-input agreement: PCA-20 ARI={agreements['log_vs_raw_pca20']:.3f}; "
        f"PCA-25 ARI={agreements['log_vs_raw_pca25']:.3f}.",
        "",
        "Post-hoc accuracy uses each cluster's majority disease label. These are descriptive in-sample values, "
        "not validated predictive performance. No minimum cluster-size constraint was imposed; choosing a "
        "more balanced result by eye would alter the scientific question.",
        "",
        "The combined four-page PDF contains the pipeline, cluster composition, PC1-PC2 projection, variance "
        "explained, and per-PC absolute Pearson correlation with the true disease state for each configuration.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    asset_readme = [
        "# Editable figure package",
        "",
        "Each configuration has a full-page SVG and five separately editable SVG panels: "
        "pipeline, composition, scatter, variance, and class_association. "
        "SVG text is preserved as text rather than outlines. Open SVG in Inkscape, Illustrator, "
        "Affinity Designer, or another vector editor to edit the individual shapes and labels.",
        "",
        "The CSV files contain patient assignments and per-PC variance/correlation values. "
        "The Python scripts reproduce the figures from the source matrix in the repository. "
        "The PNG/PDF files are previews, not the editable masters.",
        "",
        "The log branch gives the 12/26 split at both cutoffs; the raw branch gives 1/37. "
        "Cluster balance was not optimized and post-hoc class metrics are descriptive only.",
    ]
    (OUTPUT_DIR / "EDITABLE_ASSETS_README.md").write_text("\n".join(asset_readme) + "\n", encoding="utf-8")
    archive = OUTPUT_DIR / "editable_kmeans_figures_and_assets.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(OUTPUT_DIR.glob("*.svg")):
            bundle.write(path, path.name)
        for path in sorted(OUTPUT_DIR.glob("*.csv")):
            bundle.write(path, path.name)
        for name in ("report.md", "methods.json", "EDITABLE_ASSETS_README.md"):
            bundle.write(OUTPUT_DIR / name, name)
        for name in ("compare_all_motif_log_raw_kmeans.py", "plot_all_motif_pca_kmeans_cutoffs.py"):
            bundle.write(Path(__file__).parent / name, f"source/{name}")
    print(comparison.to_string(index=False), flush=True)
    print(OUTPUT_DIR / "log_vs_raw_kmeans_5000_starts.pdf", flush=True)
    print(archive, flush=True)


if __name__ == "__main__":
    main()
