#!/usr/bin/env python3
"""Build a PDF report from motif-noise experiment outputs."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cism-motif-noise")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_OUTPUT_DIR = Path("/home/omriavit/CISM/CISM_paper_revision/motif_noise_summary_outputs")
DEFAULT_ROBUSTNESS_DIR = Path("/home/omriavit/CISM/CISM_paper_revision/motif_induced_distribution_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PDF report for motif-noise experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robustness-dir", type=Path, default=DEFAULT_ROBUSTNESS_DIR)
    parser.add_argument("--pdf-path", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    if len(text) > 74:
        return text[:34] + "..." + text[-34:]
    return text


def wrap_lines(text: str, width: int = 92) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width))
    return lines


def add_text_page(pdf: PdfPages, title: str, paragraphs: Iterable[str], footer: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.92, title, fontsize=22, weight="bold", va="top")

    y = 0.84
    for paragraph in paragraphs:
        for line in wrap_lines(paragraph):
            ax.text(0.06, y, line, fontsize=11.5, va="top")
            y -= 0.034 if line else 0.02
        y -= 0.018

    if footer:
        ax.text(0.06, 0.05, footer, fontsize=8.5, color="#555555", va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    caption: Optional[str] = None,
    max_rows: int = 24,
) -> None:
    if df.empty:
        add_text_page(pdf, title, ["No table data were available for this section."])
        return

    table_df = df.head(max_rows).copy()
    for col in table_df.columns:
        table_df[col] = table_df[col].map(format_value)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=18, weight="bold", pad=18)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#333333")
        else:
            cell.set_facecolor("#f6f6f6" if row % 2 == 0 else "white")

    if caption:
        note = caption
        if len(df) > max_rows:
            note += f" Showing first {max_rows} of {len(df)} rows."
        fig.text(0.06, 0.055, note, fontsize=9, color="#555555", va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: Optional[str] = None) -> None:
    if not image_path.exists():
        add_text_page(pdf, title, [f"Missing plot: {image_path}"])
        return

    img = mpimg.imread(image_path)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.955, title, fontsize=18, weight="bold", va="top")

    bottom = 0.1 if caption else 0.055
    ax = fig.add_axes([0.04, bottom, 0.92, 0.82])
    ax.imshow(img)
    ax.axis("off")

    if caption:
        fig.text(0.05, 0.045, caption, fontsize=9.5, color="#444444", va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(noise_summary: pd.DataFrame, inventory: pd.DataFrame) -> List[str]:
    paragraphs = [
        "This report summarizes motif-induced distributions across repeated CISM runs grouped by noise condition. "
        "The screenshot-style cell-type and pairwise panels are averaged over runs with the same noise label; compact noise trend plots are used where repeating one panel per noise level would be too bulky.",
    ]

    if not inventory.empty:
        total_files = int(inventory["motif_files"].sum())
        conditions = ", ".join(inventory["condition"].astype(str).tolist())
        paragraphs.append(
            f"Current inventory: {len(inventory)} conditions and {total_files} discriminative motif CSV files were scanned. "
            f"Conditions: {conditions}."
        )

    if not noise_summary.empty:
        raw = noise_summary[noise_summary["condition"] == "raw"]
        if not raw.empty:
            raw_row = raw.iloc[0]
            paragraphs.append(
                f"Raw runs have mean optimal stringency {raw_row['mean_optimal_stringency']:.3f} "
                f"and mean best AUC {raw_row['mean_auc']:.3f}."
            )
            changed = []
            for _, row in noise_summary[noise_summary["condition"] != "raw"].iterrows():
                stringency_delta = row["mean_optimal_stringency"] - raw_row["mean_optimal_stringency"]
                auc_delta = row["mean_auc"] - raw_row["mean_auc"]
                changed.append(
                    f"{row['condition']}: stringency {row['mean_optimal_stringency']:.3f} "
                    f"({stringency_delta:+.3f} vs raw), AUC {row['mean_auc']:.3f} ({auc_delta:+.3f} vs raw)"
                )
            if changed:
                paragraphs.append("Noise-associated shifts among patient-count conditions: " + "; ".join(changed) + ".")

    paragraphs.append(
        "Important limitation: NN/NP weighted cell fractions, NN/NP pairwise heatmaps, and AUC/stringency estimates require "
        "`nodes` and `counts_by_patient` columns. Conditions with only `motif_id,edges` are still included in the unlabelled pairwise distribution comparison."
    )
    return paragraphs


def add_optional_robustness_plots(pdf: PdfPages, robustness_dir: Path) -> None:
    optional_plots = [
        (
            "Motif Count by Condition",
            robustness_dir / "motif_count_by_condition.png",
            "Number of discriminative motifs selected per run and noise condition.",
        ),
        (
            "Distance from Raw Motif-Induced Distributions",
            robustness_dir / "motif_distribution_distance_to_raw.png",
            "Distribution distance to the raw condition; larger values indicate stronger noise-induced shifts.",
        ),
        (
            "Within-Condition Run Variability",
            robustness_dir / "within_condition_motif_distribution_variability.png",
            "Run-to-run variability within each condition.",
        ),
        (
            "Motif Set Jaccard Overlap",
            robustness_dir / "motif_set_jaccard_overlap.png",
            "Overlap of selected motif identities across raw/noise conditions.",
        ),
    ]
    for title, path, caption in optional_plots:
        if path.exists():
            add_image_page(pdf, title, path, caption)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    pdf_path = args.pdf_path.resolve() if args.pdf_path else output_dir / "motif_noise_experiment_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    inventory = read_csv(output_dir / "motif_input_inventory.csv")
    noise_summary = read_csv(output_dir / "noise_effect_on_stringency_and_auc.csv")
    auc_curve = read_csv(output_dir / "average_auc_by_stringency_and_noise.csv")

    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Motif Noise Experiment Report",
            summarize_metrics(noise_summary, inventory),
            footer=f"Generated from {output_dir}",
        )
        add_table_page(
            pdf,
            "Input Inventory",
            inventory,
            "Each condition is discovered from output*/Melanoma/Motifs folders.",
        )
        add_table_page(
            pdf,
            "Noise Effect on Optimal Stringency and AUC",
            noise_summary,
            "AUC/stringency rows are available only for conditions with patient-level motif counts.",
        )
        if not auc_curve.empty:
            add_table_page(
                pdf,
                "Mean AUC by Stringency",
                auc_curve,
                "Mean and standard deviation are computed across runs for each tested shared_percentage threshold.",
            )

        add_image_page(
            pdf,
            "Optimal Stringency by Noise",
            output_dir / "optimal_stringency_by_noise.png",
            "Optimal stringency is the threshold giving the best run-level AUC in the tested grid.",
        )
        add_image_page(
            pdf,
            "Best AUC by Noise",
            output_dir / "auc_by_noise.png",
            "Points show mean best AUC across runs; error bars show run-to-run standard deviation.",
        )
        add_image_page(
            pdf,
            "AUC Across Stringency Thresholds",
            output_dir / "auc_vs_stringency_by_noise.png",
            "Noise changes both the best threshold and the stability of the AUC curve.",
        )
        add_image_page(
            pdf,
            "Average Motif-Induced Cell-Type Fractions",
            output_dir / "average_cell_type_fraction_by_noise.png",
            "Representative NN/NP panels averaged across runs within condition; error bars show run-to-run standard deviation.",
        )
        add_image_page(
            pdf,
            "Average Motif-Induced Pairwise Heatmaps",
            output_dir / "average_pairwise_heatmaps_by_noise.png",
            "Representative NN/NP pairwise interaction probabilities averaged across runs.",
        )
        add_image_page(
            pdf,
            "Average Unlabelled Pairwise Distribution Across All Noise Levels",
            output_dir / "average_unlabelled_pair_distribution_by_noise.png",
            "This compact comparison includes all discovered noise levels, including edge-only motif files.",
        )
        add_optional_robustness_plots(pdf, args.robustness_dir.resolve())

    print(f"Wrote PDF report to {pdf_path}")


if __name__ == "__main__":
    main()
