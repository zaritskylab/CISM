#!/usr/bin/env python3
"""Plot seed-based SMORE and CISM4 stringency comparisons."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = _REPO_ROOT
DATA_DIR = ROOT / "analysis" / "outputs" / "stringency_clean"
OUTPUT_DIR = DATA_DIR / "figures"

CISM_AUC = DATA_DIR / "cism4_melanoma_auc_vs_stringency_grid.csv"
CISM_COUNTS = DATA_DIR / "cism4_melanoma_motif_count_vs_stringency_grid.csv"
SMORE_COUNTS = DATA_DIR / "smore_motif_count_vs_stringency_grid.csv"
SMORE_SEED_AUC = DATA_DIR / "smore_seed_cism_adapter_auc_vs_stringency.csv"
COMBINED_AUC = DATA_DIR / "combined_auc_vs_stringency_seed_smore.csv"

SMORE_AUC_ROWS = [
    {"method": "SMORE", "stringency": 0.20, "auc": 0.6264705882352941, "source": "smore_analysis.ipynb seed-ID CISM adapter"},
    {"method": "SMORE", "stringency": 0.25, "auc": 0.6852941176470588, "source": "smore_analysis.ipynb seed-ID CISM adapter"},
    {"method": "SMORE", "stringency": 0.30, "auc": 0.18823529411764708, "source": "smore_analysis.ipynb seed-ID CISM adapter"},
]

COLORS = {
    "CISM4": "#8E44AD",
    "CISM4_LIGHT": "#C084D2",
    "SMORE": "#4F46E5",
    "SMORE_LIGHT": "#8B9AF7",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def write_auc_inputs() -> pd.DataFrame:
    smore_auc = pd.DataFrame(SMORE_AUC_ROWS)
    smore_auc.to_csv(SMORE_SEED_AUC, index=False)

    cism_auc = pd.read_csv(CISM_AUC)
    cism_auc = cism_auc[["method", "stringency", "auc"]].copy()
    cism_auc["source"] = "CISM4 clean stringency grid"

    combined = pd.concat([cism_auc, smore_auc], ignore_index=True)
    combined.to_csv(COMBINED_AUC, index=False)
    return combined


def plot_motif_counts(ax: plt.Axes) -> None:
    cism = pd.read_csv(CISM_COUNTS)
    smore = pd.read_csv(SMORE_COUNTS)
    counts = pd.concat([cism, smore], ignore_index=True)

    styles = [
        ("CISM4", "NN_motif_count", "CISM NN", COLORS["CISM4"], "--"),
        ("CISM4", "NP_motif_count", "CISM NP", COLORS["CISM4"], ":"),
        ("SMORE", "NN_motif_count", "SMORE NN", COLORS["SMORE_LIGHT"], "--"),
        ("SMORE", "NP_motif_count", "SMORE NP", COLORS["SMORE_LIGHT"], ":"),
    ]
    for method, column, label, color, linestyle in styles:
        group = counts[counts["method"] == method].sort_values("stringency")
        y = group[column].where(group[column] > 0)
        ax.plot(
            group["stringency"],
            y,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        zeros = group[(group[column] == 0) & group["stringency"].between(0.0, 0.6)]
        if not zeros.empty:
            ax.scatter(
                zeros["stringency"],
                [0.82] * len(zeros),
                marker="x",
                s=28,
                linewidths=1.4,
                color=color,
                clip_on=True,
            )

    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.6)
    ax.set_ylim(0.8, 20000)
    ax.set_xlabel("Discrimination stringency")
    ax.set_ylabel("# discriminative motifs")
    ax.set_title("Discriminative motifs vs stringency")
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)


def plot_auc(ax: plt.Axes, auc: pd.DataFrame) -> None:
    auc = auc.dropna(subset=["auc"]).copy()
    auc = auc[(auc["stringency"] >= 0.2) & (auc["stringency"] <= 0.5)]

    for method, group in auc.groupby("method", observed=True):
        group = group.sort_values("stringency")
        ax.plot(
            group["stringency"],
            group["auc"],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=COLORS["CISM4"] if method == "CISM4" else COLORS["SMORE_LIGHT"],
            label="CISM" if method == "CISM4" else "SMORE",
        )

    ax.set_xlim(0.2, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Discrimination stringency")
    ax.set_ylabel("LOOCV AUC")
    ax.set_title("AUC vs stringency")
    ax.set_xticks([0.2, 0.3, 0.4, 0.5])
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    auc = write_auc_inputs()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    plot_motif_counts(axes[0])
    plot_auc(axes[1], auc)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"seed_smore_cism_stringency_side_by_side.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {SMORE_SEED_AUC}")
    print(f"Wrote {COMBINED_AUC}")
    print(f"Wrote plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
