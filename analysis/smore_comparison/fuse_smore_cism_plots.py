#!/usr/bin/env python3
"""Fuse SMORE stringency plots with CISM 4-node figure outputs."""

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
import numpy as np
import pandas as pd
from PIL import Image


ROOT = _REPO_ROOT
OUT_DIR = ROOT / "analysis" / "outputs" / "smore_compact" / "fused_plots"
SMORE_DIR = ROOT / "analysis" / "outputs" / "smore_compact"
CISM4_DIR = ROOT / "tutorial_runtime" / "cscc_cosmx_pipeline_5color_full_100_motif4" / "analysis_outputs"
PAPER_PAGE = Path("/tmp/motifs_pdf/page10.png")

COLORS = {
    "smore_nn": "#2F80ED",
    "smore_np": "#F28E2B",
    "cism_responder": "#8E44AD",
    "cism_nonresponder": "#C77DFF",
    "auc_smore": "#2F80ED",
    "auc_cism": "#8E44AD",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_smore_counts(ax: plt.Axes) -> None:
    counts = pd.read_csv(SMORE_DIR / "smore_motif_count_vs_stringency.csv")
    ax.step(
        counts["stringency"],
        counts["NN_motif_count"],
        where="post",
        color=COLORS["smore_nn"],
        linewidth=1.7,
        label="SMORE NN",
    )
    ax.step(
        counts["stringency"],
        counts["NP_motif_count"],
        where="post",
        color=COLORS["smore_np"],
        linewidth=1.7,
        label="SMORE NP",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Discrimination stringency")
    ax.set_ylabel("Discriminative motifs")
    ax.set_title("SMORE motif count vs stringency")
    ax.legend(frameon=False, ncols=2)


def plot_smore_auc(ax: plt.Axes) -> None:
    auc = pd.read_csv(SMORE_DIR / "smore_auc_vs_stringency.csv")
    ax.plot(
        auc["stringency"],
        auc["auc"],
        marker="o",
        color=COLORS["auc_smore"],
        linewidth=1.7,
        markersize=4,
        label="SMORE",
    )
    ax.set_xlabel("Stringency")
    ax.set_ylabel("AUC")
    ax.set_title("SMORE AUC vs stringency")
    ax.set_ylim(0.25, 0.58)
    ax.legend(frameon=False)


def crop_paper_panels() -> tuple[Image.Image, Image.Image]:
    if not PAPER_PAGE.exists():
        raise FileNotFoundError(
            f"Expected rendered paper page at {PAPER_PAGE}. "
            "Run: convert -density 220 motifs.pdf[10] /tmp/motifs_pdf/page10.png"
        )
    page = Image.open(PAPER_PAGE).convert("RGB")
    panel_c = page.crop((245, 1535, 1640, 2075))
    panel_d = page.crop((300, 2110, 1535, 2375))
    return panel_c, panel_d


def make_composite() -> None:
    configure_style()
    panel_c, panel_d = crop_paper_panels()

    fig = plt.figure(figsize=(10, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.72, 0.42])
    ax_count = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])

    plot_smore_counts(ax_count)
    plot_smore_auc(ax_auc)

    ax_c.imshow(panel_c)
    ax_c.axis("off")
    ax_c.set_title("CISM 4-node Figure 2C")

    ax_d.imshow(panel_d)
    ax_d.axis("off")
    ax_d.set_title("CISM 4-node Figure 2D")

    for label, ax in zip(("A", "B", "C", "D"), (ax_count, ax_auc, ax_c, ax_d)):
        ax.text(
            -0.08,
            1.06,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="top",
            ha="left",
        )

    save(fig, "smore_with_cism4_figure2cd")


def cism4_count_curve() -> pd.DataFrame:
    cism = pd.read_csv(CISM4_DIR / "motif_count_vs_stringency.csv")
    thresholds = sorted(cism["stringency"].dropna().unique())
    rows = []
    for threshold in thresholds:
        kept = cism[cism["stringency"] >= threshold]
        counts = kept["patient_class"].value_counts()
        rows.append(
            {
                "stringency": threshold,
                "Responder": int(counts.get("Responder", 0)),
                "NonResponder": int(counts.get("NonResponder", 0)),
            }
        )
    return pd.DataFrame(rows)


def cism4_auc_curve() -> pd.DataFrame:
    trials = pd.read_csv(CISM4_DIR / "stringency_tuning_trials.csv")
    return trials.rename(columns={"params_shared_percentage": "stringency", "value": "auc"})[
        ["stringency", "auc"]
    ].sort_values("stringency")


def make_overlay() -> None:
    configure_style()
    smore_counts = pd.read_csv(SMORE_DIR / "smore_motif_count_vs_stringency.csv")
    smore_auc = pd.read_csv(SMORE_DIR / "smore_auc_vs_stringency.csv")
    cism_counts = cism4_count_curve()
    cism_auc = cism4_auc_curve()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    axes[0].step(
        smore_counts["stringency"],
        smore_counts["NN_motif_count"],
        where="post",
        color=COLORS["smore_nn"],
        linewidth=1.5,
        label="SMORE NN",
    )
    axes[0].step(
        smore_counts["stringency"],
        smore_counts["NP_motif_count"],
        where="post",
        color=COLORS["smore_np"],
        linewidth=1.5,
        label="SMORE NP",
    )
    axes[0].step(
        cism_counts["stringency"],
        cism_counts["Responder"],
        where="post",
        color=COLORS["cism_responder"],
        linewidth=1.8,
        linestyle="--",
        label="CISM4 Responder",
    )
    axes[0].step(
        cism_counts["stringency"],
        cism_counts["NonResponder"],
        where="post",
        color=COLORS["cism_nonresponder"],
        linewidth=1.8,
        linestyle="--",
        label="CISM4 NonResponder",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Discrimination stringency")
    axes[0].set_ylabel("Discriminative motifs")
    axes[0].set_title("Motif count vs stringency")
    axes[0].legend(frameon=False)

    axes[1].plot(
        smore_auc["stringency"],
        smore_auc["auc"],
        marker="o",
        color=COLORS["auc_smore"],
        linewidth=1.6,
        label="SMORE",
    )
    axes[1].plot(
        cism_auc["stringency"],
        cism_auc["auc"],
        marker="o",
        color=COLORS["auc_cism"],
        linewidth=1.6,
        label="CISM4",
    )
    axes[1].set_xlabel("Stringency")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC vs stringency")
    axes[1].set_ylim(0, 0.6)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save(fig, "smore_cism4_stringency_overlay")


def main() -> None:
    make_composite()
    make_overlay()
    print(f"Wrote fused plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
