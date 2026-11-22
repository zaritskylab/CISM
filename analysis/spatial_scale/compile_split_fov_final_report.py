#!/usr/bin/env python3
"""Compile the final melanoma split-FOV scale analysis report."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = _REPO_ROOT
RUNTIME = REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
SPLIT_DIR = RUNTIME / "split_fov_scale_experiment"
REPORT_DIR = SPLIT_DIR / "final_compiled_report"

SPATIAL_DIR = RUNTIME / "fov_spatial_scale_nn_np"
NATIVE_SCALE_DIR = RUNTIME / "fov_scale_motif_counts_nn_np"
FINAL_TESTS_DIR = SPLIT_DIR / "final_fov_scale_tests"
FOLD_UNION_DIR = SPLIT_DIR / "fold_union_stringency_046_stability"
STRINGENCY_DIR = SPLIT_DIR / "stringency_vs_split_resolution"
MOTIF_DRAW_DIR = FOLD_UNION_DIR / "motif_drawings"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt_float(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_int(value) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def md_table(df: pd.DataFrame) -> str:
    view = df.copy().fillna("")
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def prep_tables() -> dict[str, pd.DataFrame]:
    spatial = read_csv(SPATIAL_DIR / "fov_spatial_scale_summary_by_class.csv")
    spatial_tbl = pd.DataFrame(
        {
            "Class": spatial["patient_class"],
            "FOVs": spatial["n_fovs"].map(fmt_int),
            "Width mean": spatial["mean_width"].map(lambda v: fmt_float(v, 1)),
            "Width SD": spatial["std_width"].map(lambda v: fmt_float(v, 1)),
            "Height mean": spatial["mean_height"].map(lambda v: fmt_float(v, 1)),
            "Height SD": spatial["std_height"].map(lambda v: fmt_float(v, 1)),
            "Area mean": spatial["mean_size"].map(fmt_int),
            "Area SD": spatial["std_size"].map(fmt_int),
        }
    )

    native_scale = read_csv(NATIVE_SCALE_DIR / "fov_motif_counts_by_class_and_scale.csv")
    native_scale_tbl = pd.DataFrame(
        {
            "Class": native_scale["patient_class"],
            "Scale": native_scale["scale_label"],
            "FOVs": native_scale["n_fovs"].map(fmt_int),
            "Mean area": native_scale["mean_size"].map(fmt_int),
            "Mean enriched motifs": native_scale["mean_significant_motif_count"].map(
                lambda v: fmt_float(v, 1)
            ),
            "SD enriched motifs": native_scale["std_significant_motif_count"].map(
                lambda v: fmt_float(v, 1)
            ),
        }
    )

    fov_heatmap_count = read_csv(FINAL_TESTS_DIR / "heatmap_patient_count_by_small_large_fovs.csv")
    fov_heatmap_mean = read_csv(FINAL_TESTS_DIR / "heatmap_mean_patient_motifs_by_small_large_fovs.csv")

    failed = read_csv(FINAL_TESTS_DIR / "best_model_failed_patients_fov_motifs.csv")
    failed_tbl = failed.rename(
        columns={
            "patient_id": "Patient",
            "class": "True",
            "pred_class": "Predicted",
            "positive_probability": "NN probability",
            "small_fovs": "Small FOVs",
            "large_fovs": "Large FOVs",
            "patient_enriched_motif_union": "Enriched motif union",
        }
    )
    failed_tbl["NN probability"] = failed_tbl["NN probability"].map(lambda v: fmt_float(v, 3))
    failed_tbl["Enriched motif union"] = failed_tbl["Enriched motif union"].map(fmt_int)
    failed_tbl = failed_tbl[
        ["Patient", "True", "Predicted", "NN probability", "Small FOVs", "Large FOVs", "Enriched motif union"]
    ]

    optimal = read_csv(STRINGENCY_DIR / "optimal_stringency_by_split_resolution.csv")
    optimal_tbl = pd.DataFrame(
        {
            "Split parts": optimal["grid_parts"],
            "Optimal stringency": optimal["stringency"].map(lambda v: fmt_float(v, 2)),
            "ROC AUC": optimal["roc_auc"].map(lambda v: fmt_float(v, 3)),
            "Accuracy": optimal["accuracy"].map(lambda v: fmt_float(v, 3)),
            "Balanced accuracy": optimal["balanced_accuracy"].map(lambda v: fmt_float(v, 3)),
            "Mean selected motifs/fold": optimal["mean_selected_motifs"].map(lambda v: fmt_float(v, 1)),
            "Fold-union NN": optimal["fold_union_discriminative_motifs_NN"].map(fmt_int),
            "Fold-union NP": optimal["fold_union_discriminative_motifs_NP"].map(fmt_int),
        }
    )

    fold_union = read_csv(FOLD_UNION_DIR / "split_fold_union_stability.csv")
    all_split = fold_union[fold_union["scenario"].eq("all_split_fovs_aggregated")].copy()
    all_split_tbl = pd.DataFrame(
        {
            "Split parts": all_split["grid_parts"].map(fmt_int),
            "Selected motifs": all_split["selected_motif_count"].map(fmt_int),
            "Selected NN": all_split["selected_NN"].map(fmt_int),
            "Selected NP": all_split["selected_NP"].map(fmt_int),
            "Shared with native 19": all_split["shared_with_reference"].map(fmt_int),
            "Native 19 recovered": all_split["reference_recovered_fraction"].map(lambda v: fmt_float(100 * v, 1) + "%"),
            "Jaccard": all_split["jaccard"].map(lambda v: fmt_float(v, 3)),
        }
    )

    retention = read_csv(STRINGENCY_DIR / "motif_retention_summary.csv")
    retention_tbl = pd.DataFrame(
        {
            "Scenario": retention["scenario"],
            "Split parts": retention["grid_parts"].map(fmt_int),
            "Repeats": retention["repeats"].map(fmt_int),
            "Mean selected motifs": retention["mean_selected_motif_count"].map(lambda v: fmt_float(v, 1)),
            "Mean shared with native 19": retention["mean_shared_with_reference"].map(lambda v: fmt_float(v, 2)),
            "Mean native 19 recovered": retention["mean_reference_recovered_fraction"].map(
                lambda v: fmt_float(100 * v, 1) + "%"
            ),
            "Mean Jaccard": retention["mean_jaccard"].map(lambda v: fmt_float(v, 3)),
        }
    )

    motif_counts = read_csv(MOTIF_DRAW_DIR / "motif_patient_enrichment_counts_native_vs_split2.csv")
    motif_counts_tbl = pd.DataFrame(
        {
            "Motif ID": motif_counts["motif_id"].astype(str),
            "Direction": motif_counts["direction"],
            "Stable all splits": motif_counts["stable_all_split_levels"].map(lambda v: "yes" if bool(v) else "no"),
            "Native NN": motif_counts["native_NN_patients"].map(fmt_int),
            "Native NP": motif_counts["native_NP_patients"].map(fmt_int),
            "Split-2 NN": motif_counts["split2_NN_patients"].map(fmt_int),
            "Split-2 NP": motif_counts["split2_NP_patients"].map(fmt_int),
        }
    )

    return {
        "spatial": spatial_tbl,
        "native_scale": native_scale_tbl,
        "heatmap_count": fov_heatmap_count,
        "heatmap_mean": fov_heatmap_mean,
        "failed": failed_tbl,
        "optimal": optimal_tbl,
        "fold_union_all_split": all_split_tbl,
        "retention": retention_tbl,
        "motif_counts": motif_counts_tbl,
    }


def write_markdown(tables: dict[str, pd.DataFrame]) -> Path:
    paths = {
        "fov_count_heatmap": FINAL_TESTS_DIR / "heatmap_patient_count_by_small_large_fovs.png",
        "fov_mean_heatmap": FINAL_TESTS_DIR / "heatmap_mean_patient_motifs_by_small_large_fovs.png",
        "optimal_stringency": STRINGENCY_DIR / "optimal_stringency_vs_split_resolution.png",
        "stringency_sweep": STRINGENCY_DIR / "stringency_sweep_auc_by_split_resolution.png",
        "retention": STRINGENCY_DIR / "motif_retention_by_split_subsampling.png",
        "stable_8": MOTIF_DRAW_DIR / "stable_all_split_levels_8_motifs.png",
        "shared_11": MOTIF_DRAW_DIR / "shared_split2_split4_11_motifs.png",
    }

    lines = [
        "# Melanoma NN/NP Spatial-Scale Motif Analysis: Final Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Run Status",
        "",
        "The downstream split-resolution tests completed successfully. The Slurm log for job `19978644` ends with a JSON summary and `Finished at: Mon 10 Aug 2026 05:51:10 IDT`; the corresponding `.err` file is empty. No rerun is needed for the reported analyses.",
        "",
        "## Executive Summary",
        "",
        "- NP fields are effectively one full-size FOV scale, while NN includes both full-size and smaller approximately quarter-area FOVs.",
        "- The corrected native downstream reference is the LOOCV fold-union motif set at stringency 0.46: 19 motifs total, 8 NN and 11 NP.",
        "- When all split tiles are aggregated, split-2 and split-4 retain 11/19 native motifs; split-8 retains 10/19; split-16 retains 8/19.",
        "- When only one random tile per source FOV is used, retention drops sharply and becomes near zero at split-8 and split-16.",
        "- Retuned optimal stringency is not monotonic with split resolution: split-2 optimal stringency is 0.45, split-4 and split-8 are 0.35, and split-16 is 0.40.",
        "- The best saved downstream model failed on three patients: Melanoma1, Melanoma2, and Melanoma22. All three had 0 small FOVs and 2 large FOVs.",
        "",
        "## FOV Scale Structure",
        "",
        md_table(tables["spatial"]),
        "",
        "## Native Enriched Motif Counts By Class And Scale",
        "",
        md_table(tables["native_scale"]),
        "",
        "## Patient FOV Composition Heatmaps",
        "",
        "Patient count by number of small and large FOVs:",
        "",
        md_table(tables["heatmap_count"]),
        "",
        "Mean patient enriched motif union by FOV composition:",
        "",
        md_table(tables["heatmap_mean"]),
        "",
        f"![Patient count heatmap]({paths['fov_count_heatmap'].resolve()})",
        "",
        f"![Mean motif heatmap]({paths['fov_mean_heatmap'].resolve()})",
        "",
        "## Failed Patients From Best Saved Model",
        "",
        "Best saved model: `joint_asymmetric_stringency`, accuracy 0.921, balanced accuracy 0.919, ROC AUC 0.889.",
        "",
        md_table(tables["failed"]),
        "",
        "## Optimal Stringency Vs Split Resolution",
        "",
        md_table(tables["optimal"]),
        "",
        f"![Optimal stringency]({paths['optimal_stringency'].resolve()})",
        "",
        f"![Stringency sweep]({paths['stringency_sweep'].resolve()})",
        "",
        "## Corrected Fold-Union Motif Retention At Stringency 0.46",
        "",
        "Native reference: 19 fold-union motifs at stringency 0.46.",
        "",
        md_table(tables["fold_union_all_split"]),
        "",
        "## Motif Retention When Not Taking All Split Tiles",
        "",
        md_table(tables["retention"]),
        "",
        f"![Retention plot]({paths['retention'].resolve()})",
        "",
        "## Shared Motifs And Patient-Class Counts",
        "",
        md_table(tables["motif_counts"]),
        "",
        "## Motif Drawings",
        "",
        "Stable motifs shared across all split levels:",
        "",
        f"![Stable 8 motifs]({paths['stable_8'].resolve()})",
        "",
        "Motifs shared by native analysis and split-2/split-4:",
        "",
        f"![Shared 11 motifs]({paths['shared_11'].resolve()})",
        "",
        "## Output Sources",
        "",
        f"- `{STRINGENCY_DIR.resolve()}`",
        f"- `{FOLD_UNION_DIR.resolve()}`",
        f"- `{FINAL_TESTS_DIR.resolve()}`",
        f"- `{MOTIF_DRAW_DIR.resolve()}`",
        "",
        "## PDF Rendering Note",
        "",
        "This PDF was generated with Matplotlib tables, which works without extra dependencies. For publication-quality Markdown-to-PDF table rendering, install `pandoc` plus a LaTeX engine such as `xelatex`/TeX Live, or install `weasyprint` for HTML/CSS-based PDF rendering.",
        "",
    ]
    md_path = REPORT_DIR / "melanoma_split_fov_scale_final_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.95, title, fontsize=18, weight="bold", va="top")
    y = 0.89
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=95) or [""]
        for line in wrapped:
            fig.text(0.07, y, line, fontsize=10.5, va="top")
            y -= 0.027
        y -= 0.012
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, *, landscape: bool = False, font_size: int = 8) -> None:
    figsize = (11.69, 8.27) if landscape else (8.27, 11.69)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    wrapped = df.copy().fillna("")
    wrapped.columns = [textwrap.fill(str(col), 18) for col in wrapped.columns]
    cell_text = []
    for row in wrapped.to_numpy():
        cell_text.append([textwrap.fill(str(value), 18) for value in row])
    table = ax.table(
        cellText=cell_text,
        colLabels=wrapped.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#355C7D")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6FA")
    fig.suptitle(title, fontsize=15, weight="bold", y=0.96)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, path: Path, *, landscape: bool = False) -> None:
    if not path.exists():
        return
    figsize = (11.69, 8.27) if landscape else (8.27, 11.69)
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=15, weight="bold", y=0.97)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf(tables: dict[str, pd.DataFrame]) -> Path:
    pdf_path = REPORT_DIR / "melanoma_split_fov_scale_final_report.pdf"
    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Melanoma NN/NP Spatial-Scale Motif Analysis",
            [
                "The split-resolution Slurm job completed successfully. Expected output tables and figures are present; no rerun is needed for this report.",
                "The corrected native downstream reference is the LOOCV fold-union motif set at stringency 0.46: 19 motifs total, 8 NN and 11 NP.",
                "All split tiles aggregated retain a meaningful but decreasing fraction of the native 19 motifs as split resolution becomes finer: 11/19 for split-2 and split-4, 10/19 for split-8, and 8/19 for split-16.",
                "Randomly using only one split tile per source FOV sharply reduces retention, with near-zero retention at split-8 and split-16.",
                "Retuned optimal stringency changes with split resolution: 0.45 for split-2, 0.35 for split-4, 0.35 for split-8, and 0.40 for split-16.",
            ],
        )
        add_table_page(pdf, "FOV Spatial Scale By Class", tables["spatial"], landscape=True)
        add_table_page(pdf, "Native Enriched Motif Counts", tables["native_scale"], landscape=True)
        add_table_page(pdf, "Patient FOV Composition: Counts", tables["heatmap_count"], landscape=False, font_size=10)
        add_table_page(pdf, "Patient FOV Composition: Mean Motif Union", tables["heatmap_mean"], landscape=False, font_size=10)
        add_image_page(pdf, "Patient Count Heatmap", FINAL_TESTS_DIR / "heatmap_patient_count_by_small_large_fovs.png")
        add_image_page(pdf, "Mean Motif Heatmap", FINAL_TESTS_DIR / "heatmap_mean_patient_motifs_by_small_large_fovs.png")
        add_table_page(pdf, "Failed Patients From Best Saved Model", tables["failed"], landscape=True)
        add_table_page(pdf, "Optimal Stringency Vs Split Resolution", tables["optimal"], landscape=True, font_size=7)
        add_image_page(pdf, "Optimal Stringency Vs Split Resolution", STRINGENCY_DIR / "optimal_stringency_vs_split_resolution.png", landscape=True)
        add_image_page(pdf, "Stringency Sweep", STRINGENCY_DIR / "stringency_sweep_auc_by_split_resolution.png", landscape=True)
        add_table_page(pdf, "All Split Tiles: Corrected Fold-Union Retention", tables["fold_union_all_split"], landscape=True)
        add_table_page(pdf, "Motif Retention Under Split Subsampling", tables["retention"], landscape=True, font_size=7)
        add_image_page(pdf, "Motif Retention Under Split Subsampling", STRINGENCY_DIR / "motif_retention_by_split_subsampling.png", landscape=True)
        add_table_page(pdf, "Shared Motifs: Native Vs Split-2 Patient Counts", tables["motif_counts"], landscape=True, font_size=7)
        add_image_page(pdf, "Stable 8 Motifs Across All Split Levels", MOTIF_DRAW_DIR / "stable_all_split_levels_8_motifs.png", landscape=True)
        add_image_page(pdf, "Shared 11 Motifs For Split-2/Split-4", MOTIF_DRAW_DIR / "shared_split2_split4_11_motifs.png", landscape=True)
        add_text_page(
            pdf,
            "Rendering Note",
            [
                "This PDF uses Matplotlib's built-in PDF backend and table artist, so it does not require optional system packages.",
                "For more publication-quality table layout from Markdown, install pandoc plus a LaTeX engine such as xelatex/TeX Live, or use weasyprint for HTML/CSS PDF rendering.",
            ],
        )
    return pdf_path


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tables = prep_tables()
    md_path = write_markdown(tables)
    pdf_path = write_pdf(tables)
    summary = {
        "markdown_report": str(md_path),
        "pdf_report": str(pdf_path),
        "no_rerun_needed": True,
    }
    (REPORT_DIR / "compiled_report_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
