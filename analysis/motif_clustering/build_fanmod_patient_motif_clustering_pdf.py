#!/usr/bin/env python3
"""Build an integrated PDF report for the FANMOD+ patient clustering analysis."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
import textwrap
import argparse
import json
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


SCRIPT_DIR = ANALYSIS_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fanmod_patient_motif_clustering_NN_NP"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_page_number(fig: plt.Figure, page: int) -> None:
    fig.text(0.95, 0.025, str(page), ha="right", va="bottom", fontsize=8, color="#555555")


def write_wrapped(
    fig: plt.Figure,
    text: str,
    *,
    x: float,
    y: float,
    width: int = 92,
    size: int = 10,
    line_height: float = 0.026,
    weight: str = "normal",
) -> float:
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y -= line_height
            continue
        wrapped = textwrap.wrap(paragraph, width=width, break_long_words=False, replace_whitespace=False)
        for line in wrapped:
            fig.text(x, y, line, ha="left", va="top", fontsize=size, fontweight=weight)
            y -= line_height
    return y


def add_title_page(pdf: PdfPages, page: int, summary: dict[str, str]) -> int:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "FANMOD+ Patient Motif Clustering", fontsize=23, fontweight="bold", ha="left")
    fig.text(0.08, 0.895, "Integrated reviewer-facing report", fontsize=13, color="#555555", ha="left")

    y = 0.82
    overview = (
        "We built a patient-by-motif matrix directly from FANMOD+ output and tested whether the motif "
        "profiles divide patients into two unsupervised groups. Motifs were represented by canonical "
        "color/topology signatures, so motif columns are comparable across FOVs and patients."
    )
    y = write_wrapped(fig, overview, x=0.08, y=y, width=88, size=11, line_height=0.031)

    fig.text(0.08, y - 0.02, "Selected Result", fontsize=15, fontweight="bold", ha="left")
    y -= 0.065
    selected_lines = [
        ("Matrix", summary["matrix_shape"]),
        ("Strategy", summary["strategy"]),
        ("Cluster sizes", summary["cluster_sizes"]),
        ("Silhouette", summary["silhouette"]),
        ("Patient-subsampling ARI", summary["stability"]),
        ("Davies-Bouldin", summary["davies"]),
        ("Post hoc clinical agreement", summary["clinical_agreement"]),
    ]
    for label, value in selected_lines:
        fig.text(0.10, y, f"{label}:", fontsize=10.5, fontweight="bold", ha="left", va="top")
        fig.text(0.38, y, value, fontsize=10.5, ha="left", va="top")
        y -= 0.036

    fig.text(0.08, y - 0.02, "Core Interpretation", fontsize=15, fontweight="bold", ha="left")
    y -= 0.065
    interpretation = (
        "Clinical labels were not used during clustering. They were added only after the unsupervised "
        "two-group assignment was chosen by internal separation and stability. The post hoc clinical "
        "agreement values quantify how closely the unsupervised motif groups align with the retained "
        "patient labels."
    )
    write_wrapped(fig, interpretation, x=0.08, y=y, width=88, size=10.5, line_height=0.029)

    add_page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page + 1


def add_text_table_page(
    pdf: PdfPages,
    page: int,
    title: str,
    paragraphs: list[str],
    tables: list[tuple[str, pd.DataFrame]],
) -> int:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.95, title, fontsize=17, fontweight="bold", ha="left", va="top")
    y = 0.90
    for paragraph in paragraphs:
        y = write_wrapped(fig, paragraph, x=0.06, y=y, width=95, size=10, line_height=0.027)
        y -= 0.012

    for table_title, table in tables:
        fig.text(0.06, y, table_title, fontsize=12, fontweight="bold", ha="left", va="top")
        y -= 0.035
        ax = fig.add_axes([0.06, max(0.08, y - 0.23), 0.88, 0.20])
        ax.axis("off")
        display = table.copy()
        display.columns = [str(col) for col in display.columns]
        tbl = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.25)
        for (row, _col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#EAEAEA")
            cell.set_edgecolor("#CCCCCC")
        y -= 0.27

    add_page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page + 1


def add_figure_page(
    pdf: PdfPages,
    page: int,
    title: str,
    image_path: Path,
    caption: str,
    *,
    landscape: bool = False,
) -> int:
    fig = plt.figure(figsize=(11.69, 8.27) if landscape else (8.27, 11.69))
    fig.text(0.055, 0.955, title, fontsize=16, fontweight="bold", ha="left", va="top")
    y_caption = 0.12 if landscape else 0.10
    ax = fig.add_axes([0.06, y_caption + 0.07, 0.88, 0.74 if landscape else 0.76])
    ax.axis("off")
    image = Image.open(image_path)
    ax.imshow(image)
    write_wrapped(fig, caption, x=0.06, y=y_caption, width=130 if landscape else 95, size=9.5, line_height=0.022)
    add_page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page + 1


def format_float(value: float) -> str:
    return f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-mode-gallery", action="store_true")
    return parser.parse_args()


def build_pdf(output_dir: Path, include_mode_gallery: bool = False) -> Path:
    configure_style()
    figure_dir = output_dir / "figures"
    pdf_path = output_dir / "fanmod_patient_motif_clustering_integrated_report.pdf"
    results = pd.read_csv(output_dir / "strategy_results.csv")
    sort_columns = ["selection_rank_score", "unsupervised_rank_score"] if "selection_rank_score" in results.columns else ["unsupervised_rank_score"]
    results = results.sort_values(sort_columns)
    assignments = pd.read_csv(output_dir / "selected_cluster_assignments.csv")
    fov_summary = pd.read_csv(output_dir / "fanmod_fov_summary.csv")
    prevalence = pd.read_csv(output_dir / "selected_motif_prevalence.csv")
    config = json.loads((output_dir / "analysis_config.json").read_text(encoding="utf-8"))
    mode_index_path = output_dir / "mode_visualization_index.csv"
    mode_index = pd.read_csv(mode_index_path) if mode_index_path.exists() else pd.DataFrame()
    best = results.iloc[0]

    class_table = pd.crosstab(assignments["cluster"], assignments["class_label"]).reset_index()
    axis_1_table = pd.crosstab(assignments["cluster"], assignments["axis_1"]).reset_index()
    axis_2_table = pd.crosstab(assignments["cluster"], assignments["axis_2"]).reset_index()

    top_strategies = results.head(12)[
        [
            "feature_set",
            "reduction",
            "cluster_method",
            "n_cluster_0",
            "n_cluster_1",
            "silhouette",
            "stability_ari",
            "davies_bouldin",
            "majority_label_accuracy",
            "majority_label_balanced_accuracy",
            "min_cluster_purity",
        ]
    ].copy()
    for column in [
        "silhouette",
        "stability_ari",
        "davies_bouldin",
        "majority_label_accuracy",
        "majority_label_balanced_accuracy",
        "min_cluster_purity",
    ]:
        top_strategies[column] = top_strategies[column].map(format_float)

    predictive_table = pd.read_csv(output_dir / "cluster_predictive_results.csv").head(15)[
        [
            "feature_set",
            "reduction",
            "cluster_method",
            "n_cluster_0",
            "n_cluster_1",
            "majority_label_accuracy",
            "majority_label_balanced_accuracy",
            "cluster_purity",
            "min_cluster_purity",
            "cluster_majority_label_map",
        ]
    ].copy()
    for column in [
        "majority_label_accuracy",
        "majority_label_balanced_accuracy",
        "cluster_purity",
        "min_cluster_purity",
    ]:
        predictive_table[column] = predictive_table[column].map(format_float)

    assignment_table = assignments[["patient_id", "class_label", "cluster"]].copy()
    assignment_table = assignment_table.sort_values(["cluster", "class_label", "patient_id"]).reset_index(drop=True)

    summary = {
        "matrix_shape": f"{len(assignments)} patients x {len(prevalence):,} motifs",
        "strategy": str(best["strategy"]),
        "cluster_sizes": f"{int(best['n_cluster_0'])} and {int(best['n_cluster_1'])}",
        "silhouette": format_float(float(best["silhouette"])),
        "stability": format_float(float(best["stability_ari"])),
        "davies": format_float(float(best["davies_bouldin"])),
        "clinical_agreement": (
            f"ARI class {format_float(float(best['ari_4class']))}; "
            f"NMI class {format_float(float(best['nmi_4class']))}"
        ),
    }

    with PdfPages(pdf_path) as pdf:
        page = 1
        page = add_title_page(pdf, page, summary)
        page = add_text_table_page(
            pdf,
            page,
            "Data, Matrix, and Method",
            [
                (
                    f"The analysis parsed {len(fov_summary)} FANMOD+ CSV files. These files contained "
                    f"{int(fov_summary['fanmod_rows'].sum()):,} FANMOD motif rows, which collapsed to "
                    "canonical patient-by-motif entries after grouping by patient and motif signature."
                ),
                (
                    "The primary matrix value is the CISM motif frequency: total motif count divided by "
                    "the summed FOV-specific subgraph opportunity estimate, computed as Count / Frequency. "
                    "The strategy sweep also tested log-frequency, binary motif presence, and log-count "
                    "TF-IDF transforms."
                ),
                (f"No prevalence or top-variance motif filter was applied: all {len(prevalence):,} observed "
                 "motif signatures were retained."
                 if config.get("all_motifs") else
                 f"Motif columns were filtered to signatures present in at least "
                 f"{config['min_patients_per_motif']} patients, then the top {len(prevalence):,} "
                 "variable signatures were retained."),
            ],
            [
                ("Clinical-class composition of selected clusters", class_table),
                ("Binary clinical axes, post hoc", pd.concat([axis_1_table.assign(axis="first"), axis_2_table.assign(axis="second")], ignore_index=True).fillna("")),
            ],
        )
        page = add_figure_page(
            pdf,
            page,
            "Strategy Sweep",
            figure_dir / "strategy_quality_sweep.png",
            "Each point is a feature transform, dimensionality reduction, and two-cluster method. The selected strategy was chosen by unsupervised separation, patient-subsampling stability, Davies-Bouldin index, and cluster balance.",
            landscape=True,
        )
        page = add_text_table_page(
            pdf,
            page,
            "Top Strategy Table",
            [
                "The table below lists the highest-ranked candidate strategies under the unsupervised selection score. Majority-label columns treat each cluster as predicting its own majority class.",
            ],
            [("Top 12 strategies by unsupervised rank score", top_strategies)],
        )
        page = add_text_table_page(
            pdf,
            page,
            "Cluster Predictive Ability",
            [
                "For this table, each cluster is assigned its own majority class label. This intentionally benefits homogeneous clusters and does not require a one-to-one mapping between the two clusters and the two clinical labels.",
            ],
            [("Top configurations by majority-label balanced accuracy", predictive_table)],
        )
        if include_mode_gallery and not mode_index.empty:
            for _, mode_row in mode_index.iterrows():
                page = add_figure_page(
                    pdf,
                    page,
                    f"Mode Gallery: {mode_row['feature_set']} + {mode_row['reduction']}",
                    Path(mode_row["figure_png"]),
                    (
                        "Each panel uses the same display coordinates for this feature/reduction mode, "
                        "then colors points by the two clusters from one clustering method. Marker shape "
                        "shows the retained clinical label, so visual separation and label alignment can "
                        "be inspected method by method."
                    ),
                    landscape=True,
                )
        page = add_figure_page(
            pdf,
            page,
            "Selected Two-Group Structure",
            figure_dir / "selected_embedding.png",
            "The same patient embedding is shown twice: first colored by clinical class and then by the selected unsupervised cluster labels.",
            landscape=True,
        )
        page = add_figure_page(
            pdf,
            page,
            "Cluster Clinical Composition",
            figure_dir / "selected_cluster_class_composition.png",
            "Clinical labels were overlaid after clustering to assess whether the unsupervised motif groups align with the known patient annotations.",
        )
        page = add_figure_page(
            pdf,
            page,
            "Patient Motif Heatmap",
            figure_dir / "selected_patient_motif_heatmap.png",
            "Rows are patients ordered by selected cluster. Columns are top variable motif signatures. Values are column z-scores of log-scaled motif frequency.",
            landscape=True,
        )
        page = add_figure_page(
            pdf,
            page,
            "Hierarchical Clustering",
            figure_dir / "patient_motif_dendrogram.png",
            "Ward hierarchical clustering provides an independent visualization of relatedness among patient motif profiles.",
            landscape=True,
        )
        page = add_text_table_page(
            pdf,
            page,
            "Selected Cluster Assignments",
            [
                "Patient-level assignments for the selected two-cluster solution are listed below. These are the rows to cite if a reviewer asks which patients define each group.",
            ],
            [("Assignments", assignment_table)],
        )

    return pdf_path


def main() -> None:
    args = parse_args()
    path = build_pdf(args.output_dir, include_mode_gallery=args.include_mode_gallery)
    print(path)


if __name__ == "__main__":
    main()
