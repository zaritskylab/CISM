#!/usr/bin/env python3
"""Plot CISM4 and SMORE AUC versus stringency from clean grid CSVs."""

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
INPUT = ROOT / "analysis" / "outputs" / "stringency_clean" / "combined_auc_vs_stringency_grid.csv"
OUTPUT_DIR = ROOT / "analysis" / "outputs" / "stringency_clean" / "figures"

COLORS = {
    "CISM4": "#8E44AD",
    "SMORE": "#4F46E5",
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
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT).dropna(subset=["auc"]).copy()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for method, group in df.groupby("method", observed=True):
        group = group.sort_values("stringency")
        ax.plot(
            group["stringency"],
            group["auc"],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=COLORS.get(method),
            label="CISM" if method == "CISM4" else method,
        )

    ax.set_xlabel("Discrimination stringency")
    ax.set_ylabel("LOOCV AUC")
    ax.set_title("AUC vs stringency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([round(x / 10, 1) for x in range(0, 11)])
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"auc_vs_stringency_cism_smore.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote AUC plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
