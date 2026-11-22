#!/usr/bin/env python3
"""Replot historical and 38-patient SMORE comparisons with benchmark CISM AUCs."""

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


ROOT = _REPO_ROOT
DATA = ROOT / "analysis/outputs/stringency_clean"
FIGURES = DATA / "figures"
CISM_COLOR = "#8E44AD"
SMORE_COLOR = "#6678C7"


def cism_auc() -> pd.DataFrame:
    source = pd.read_csv(DATA / "cism_benchmark_smore_provisional_auc.csv")
    return source[source.method == "CISM 4-node benchmark"].sort_values("stringency")


def plot_counts(ax: plt.Axes, counts: pd.DataFrame, current: bool) -> None:
    for method, color in (("CISM 4-node", CISM_COLOR),
                          ("SMORE seeds, provisional", SMORE_COLOR)):
        for label, linestyle in (("NN", "--"), ("NP", ":")):
            if current:
                group = counts[(counts.method == method) & (counts["class"] == label)]
                x, y = group.stringency, group.motif_count
            else:
                group = counts[counts.method == ("CISM4" if method.startswith("CISM") else "SMORE")]
                x, y = group.stringency, group[f"{label}_motif_count"]
            positive = y > 0
            ax.plot(x[positive], y[positive], color=color, linestyle=linestyle,
                    marker="o", markersize=3.2, linewidth=1.7,
                    label=f"{'CISM' if method.startswith('CISM') else 'SMORE'} {label}")
            if (~positive).any():
                ax.scatter(x[~positive], np.full((~positive).sum(), 0.8),
                           color=color, marker="x", s=24, linewidths=1.2)
    ax.set(xlim=(0, 0.6), ylim=(0.7, 20000), yscale="log",
           xlabel="Discrimination stringency", ylabel="Discriminative motifs",
           title="A  Motif count vs stringency")
    ax.set_xticks(np.arange(0, 0.61, 0.1))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)


def plot_auc(ax: plt.Axes, smore: pd.DataFrame, current: bool) -> None:
    cism = cism_auc()
    ax.plot(cism.stringency, cism.auc, color=CISM_COLOR, marker="o",
            linewidth=2, markersize=4, label="CISM 4-node benchmark")
    ycol = "auc_failed_equal" if current else "auc"
    ax.plot(smore.stringency, smore[ycol], color=SMORE_COLOR, marker="s",
            linewidth=2, markersize=4,
            label="SMORE provisional (38 patients)" if current else "SMORE notebook (37 patients)")
    ax.set(xlim=(0.2, 0.5), ylim=(0, 1), xlabel="Discrimination stringency",
           ylabel="LOOCV AUC", title="B  AUC vs stringency")
    ax.set_xticks(np.arange(0.2, 0.501, 0.05))
    ax.set_xticklabels([f"{x:.2f}" for x in np.arange(0.2, 0.501, 0.05)])
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def make_plot(current: bool) -> None:
    if current:
        counts = pd.read_csv(DATA / "cism4_smore_seed38_motif_count_vs_stringency.csv")
        smore = pd.read_csv(DATA / "smore_provisional_cism_pipeline_auc.csv")
        stem = "smore_cism_38_patient_provisional_benchmark_corrected"
        note = "SMORE: 38 patients; seed identity; forced p=0.01 placeholder. Native SMORE validation pending."
    else:
        cism = pd.read_csv(DATA / "cism4_melanoma_motif_count_vs_stringency_grid.csv")
        smore_counts = pd.read_csv(DATA / "smore_motif_count_vs_stringency_grid.csv")
        counts = pd.concat([cism, smore_counts], ignore_index=True)
        smore = pd.read_csv(DATA / "smore_seed_cism_adapter_auc_vs_stringency.csv")
        stem = "seed_smore_cism_august_comparison_corrected"
        note = "Historical SMORE AUC: 37 patients, three notebook values only. Purple: CISM benchmark, 38 patients."

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    plot_counts(axes[0], counts, current)
    plot_auc(axes[1], smore, current)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CISM and SMORE melanoma motif comparison", fontsize=12)
    fig.text(0.5, -0.02, note, ha="center", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(FIGURES / f"{stem}.png")


if __name__ == "__main__":
    make_plot(current=False)
    make_plot(current=True)
