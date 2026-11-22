#!/usr/bin/env python3
"""Build the final concise FANMOD+ NN/NP clustering PDF report."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import ast
import os
import textwrap
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


SCRIPT_DIR = ANALYSIS_DIR
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fanmod_patient_motif_clustering_NN_NP_tutorial_artifact"
PDF_PATH = OUTPUT_DIR / "final_fanmod_nn_np_clustering_report.pdf"


def write_wrapped(fig: plt.Figure, text: str, x: float, y: float, width: int = 92, size: int = 10, line_height: float = 0.027) -> float:
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y -= line_height
            continue
        for line in textwrap.wrap(paragraph, width=width, break_long_words=False):
            fig.text(x, y, line, ha="left", va="top", fontsize=size)
            y -= line_height
    return y


def add_table(fig: plt.Figure, df: pd.DataFrame, bbox: list[float], font_size: int = 8) -> None:
    ax = fig.add_axes(bbox)
    ax.axis("off")
    display = df.copy()
    display.columns = [str(col) for col in display.columns]
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def short_feature(value: str) -> str:
    return {
        "cism_frequency": "raw freq",
        "log_cism_frequency": "log freq",
        "motif_presence": "presence",
        "log_count_tfidf": "tfidf",
    }.get(value, value)


def short_reduction(value: str) -> str:
    return (
        value.replace("umap_neighbors_10", "UMAP")
        .replace("spectral_embedding", "spectral")
        .replace("mds_cosine_2", "MDS")
        .replace("svd_20", "SVD20")
        .replace("pca_20", "PCA20")
        .replace("pca_8", "PCA8")
        .replace("nmf_8", "NMF8")
    )


def short_cluster(value: str) -> str:
    return (
        value.replace("spectral_clustering", "spectral")
        .replace("agglomerative_complete", "agg complete")
        .replace("agglomerative_average", "agg average")
        .replace("agglomerative_ward", "agg ward")
        .replace("gmm_full", "GMM")
    )


def paginate_dataframe(df: pd.DataFrame, page_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + page_size].copy() for start in range(0, len(df), page_size)]


def infer_binary_cluster_counts(row: pd.Series, nn_total: int, np_total: int) -> pd.DataFrame:
    mapping = ast.literal_eval(row["cluster_majority_label_map"])
    cluster_sizes = {0: int(row["n_cluster_0"]), 1: int(row["n_cluster_1"])}
    target_correct = round(float(row["majority_label_accuracy"]) * (nn_total + np_total))

    for nn0 in range(cluster_sizes[0] + 1):
        np0 = cluster_sizes[0] - nn0
        nn1 = nn_total - nn0
        np1 = np_total - np0
        if nn1 < 0 or np1 < 0 or nn1 + np1 != cluster_sizes[1]:
            continue
        correct = 0
        for cluster, label in mapping.items():
            cluster = int(cluster)
            correct += (nn0 if cluster == 0 else nn1) if label == "NN" else (np0 if cluster == 0 else np1)
        if correct == target_correct:
            return pd.DataFrame(
                [
                    {"cluster": 0, "NN": nn0, "NP": np0, "majority label": mapping["0"]},
                    {"cluster": 1, "NN": nn1, "NP": np1, "majority label": mapping["1"]},
                ]
            )
    raise ValueError(f"Could not infer class counts for {row['strategy']}")


def main() -> None:
    results = pd.read_csv(OUTPUT_DIR / "strategy_results.csv")
    predictive = pd.read_csv(OUTPUT_DIR / "cluster_predictive_results.csv")
    assignments = pd.read_csv(OUTPUT_DIR / "selected_cluster_assignments.csv")

    selected = results.sort_values(["selection_rank_score", "unsupervised_rank_score"]).iloc[0]
    best_predictive = predictive.iloc[0]
    nn_total = int((assignments["class_label"] == "NN").sum())
    np_total = int((assignments["class_label"] == "NP").sum())

    selected_counts = pd.crosstab(assignments["cluster"], assignments["class_label"]).reset_index()
    selected_counts["majority label"] = selected_counts[["NN", "NP"]].idxmax(axis=1)
    best_counts = infer_binary_cluster_counts(best_predictive, nn_total, np_total)

    summary_rows = [
        {
            "analysis view": "Best post-hoc predictive",
            "configuration": best_predictive["strategy"],
            "accuracy": format_metric(best_predictive["majority_label_accuracy"]),
            "balanced acc.": format_metric(best_predictive["majority_label_balanced_accuracy"]),
            "macro F1": format_metric(best_predictive["majority_label_macro_f1"]),
            "stability ARI": format_metric(best_predictive["stability_ari"]),
            "clusters": f"{int(best_predictive['n_cluster_0'])}/{int(best_predictive['n_cluster_1'])}",
        },
        {
            "analysis view": "Best unsupervised selection",
            "configuration": selected["strategy"],
            "accuracy": format_metric(selected["majority_label_accuracy"]),
            "balanced acc.": format_metric(selected["majority_label_balanced_accuracy"]),
            "macro F1": format_metric(selected["majority_label_macro_f1"]),
            "stability ARI": format_metric(selected["stability_ari"]),
            "clusters": f"{int(selected['n_cluster_0'])}/{int(selected['n_cluster_1'])}",
        },
    ]
    summary = pd.DataFrame(summary_rows)

    full_predictive = predictive[
        [
            "feature_set",
            "reduction",
            "cluster_method",
            "n_cluster_0",
            "n_cluster_1",
            "majority_label_accuracy",
            "majority_label_balanced_accuracy",
            "majority_label_macro_f1",
            "stability_ari",
            "min_cluster_purity",
        ]
    ].copy()
    full_predictive.insert(0, "rank", range(1, len(full_predictive) + 1))
    full_predictive["feature_set"] = full_predictive["feature_set"].map(short_feature)
    full_predictive["reduction"] = full_predictive["reduction"].map(short_reduction)
    full_predictive["cluster_method"] = full_predictive["cluster_method"].map(short_cluster)
    for column in [
        "majority_label_accuracy",
        "majority_label_balanced_accuracy",
        "majority_label_macro_f1",
        "stability_ari",
        "min_cluster_purity",
    ]:
        full_predictive[column] = full_predictive[column].map(format_metric)
    full_predictive = full_predictive.rename(
        columns={
            "feature_set": "feature",
            "reduction": "reduction",
            "cluster_method": "cluster",
            "n_cluster_0": "c0",
            "n_cluster_1": "c1",
            "majority_label_accuracy": "accuracy",
            "majority_label_balanced_accuracy": "bal. acc.",
            "majority_label_macro_f1": "macro F1",
            "stability_ari": "stability",
            "min_cluster_purity": "min purity",
        }
    )

    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.955, "Final FANMOD+ NN/NP Patient Clustering Report", fontsize=18, fontweight="bold", ha="left", va="top")
        y = 0.90
        y = write_wrapped(
            fig,
            (
                "Cohort: 38 melanoma tutorial-artifact patients with FANMOD+ motif profiles "
                f"({nn_total} NN, {np_total} NP). Cluster labels were evaluated post hoc by assigning each cluster "
                "its majority class label, which favors homogeneous clusters."
            ),
            0.06,
            y,
            width=95,
            size=10.5,
        )
        y -= 0.02
        fig.text(0.06, y, "Summary Metrics", fontsize=13, fontweight="bold", ha="left", va="top")
        add_table(fig, summary, [0.04, 0.52, 0.92, 0.25], font_size=7)
        y = 0.48
        fig.text(0.06, y, "Class Composition Per Cluster", fontsize=13, fontweight="bold", ha="left", va="top")
        fig.text(0.06, y - 0.04, "Best post-hoc predictive configuration", fontsize=10, fontweight="bold", ha="left")
        add_table(fig, best_counts, [0.08, 0.28, 0.38, 0.13], font_size=9)
        fig.text(0.54, y - 0.04, "Best unsupervised selected configuration", fontsize=10, fontweight="bold", ha="left")
        add_table(fig, selected_counts, [0.56, 0.28, 0.34, 0.13], font_size=9)
        y = 0.22
        fig.text(0.06, y, "Interpretation", fontsize=13, fontweight="bold", ha="left", va="top")
        write_wrapped(
            fig,
            (
                "The strongest post-hoc NN/NP separation achieved 0.711 accuracy and 0.719 balanced accuracy, "
                "but its bootstrap stability was modest (ARI 0.311). The most stable unsupervised selection "
                "had stability ARI 1.000 but weaker post-hoc NN/NP accuracy (0.553). Thus, motif profiles contain "
                "a detectable NN/NP signal, but the cluster structure is not a clean or highly stable supervised surrogate."
            ),
            0.06,
            y - 0.04,
            width=96,
            size=10,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        pages = paginate_dataframe(full_predictive, page_size=24)
        for index, page_df in enumerate(pages, start=1):
            fig = plt.figure(figsize=(11.69, 8.27))
            title = f"All Configurations by Post-Hoc Predictive Ability ({index}/{len(pages)})"
            fig.text(0.04, 0.95, title, fontsize=15, fontweight="bold", ha="left", va="top")
            if index == 1:
                fig.text(
                    0.04,
                    0.905,
                    "Sorted by majority-label balanced accuracy, then accuracy, purity, and silhouette. Short labels: raw freq = raw CISM motif frequency; tfidf = log-count TF-IDF.",
                    fontsize=8.5,
                    ha="left",
                    va="top",
                )
            add_table(fig, page_df, [0.02, 0.06, 0.96, 0.80], font_size=6.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(PDF_PATH)


if __name__ == "__main__":
    main()
