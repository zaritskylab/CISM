#!/usr/bin/env python3
"""Plot native 10% SMORE reported-motif LOOCV AUC with CISM benchmark AUC."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = _REPO_ROOT
OUTPUT = ROOT / "analysis/outputs/smore_matched"
BENCHMARK = ROOT / "analysis/outputs/stringency_clean/cism_benchmark_smore_provisional_auc.csv"
FANMOD_HALF = ROOT / "analysis/outputs/stringency_clean/cism4_melanoma_auc_vs_stringency_grid.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="gdelaunay_p0.1_n5_train50_score10")
    args = parser.parse_args()
    run_dir = OUTPUT / "runs" / args.run_tag
    smore = pd.read_csv(run_dir / "smore_matched_auc.csv")
    metadata_file = next(run_dir.glob("FOV*_metadata.csv"))
    metadata = pd.read_csv(metadata_file).iloc[0]
    uncapped = int(metadata.max_motifs) == 0
    cism = pd.read_csv(BENCHMARK)
    cism = cism[cism.method == "CISM 4-node benchmark"]
    half = pd.read_csv(FANMOD_HALF)
    half = half[(half.method == "CISM4") & (np.isclose(half.stringency, 0.5))]
    if len(half) != 1 or int(half.iloc[0].valid_folds) != 38:
        raise ValueError("Expected one 38-fold FANMOD+ clean-grid result at stringency 0.50")
    half_auc = float(half.iloc[0].auc)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.23)
    cism_plot = pd.concat(
        [cism[["stringency", "auc"]],
         pd.DataFrame({"stringency": [0.5], "auc": [half_auc]})],
        ignore_index=True,
    ).sort_values("stringency")
    ax.plot(cism_plot.stringency, cism_plot.auc, color="#8E44AD", marker="o",
            linewidth=2.2, label="FANMOD+")
    evaluable = smore[smore.auc_reportable & smore.auc_equal.notna()]
    if not evaluable.empty:
        ax.plot(evaluable.stringency, evaluable.auc_equal, color="#6678C7", marker="s",
                linewidth=2.2, label="SMORE")
    else:
        first = smore.iloc[0]
        ax.text(0.55, 0.48,
                f"SMORE AUC unavailable\n{int(first.failed_folds)}/38 folds failed at {first.stringency:.2f}",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color="#4B5563")
    ax.set(xlim=(0.195, 0.505), ylim=(0, 1), xlabel="Discrimination stringency",
           ylabel="LOOCV AUC", title="AUC vs stringency (38 melanoma patients)")
    ticks = np.arange(0.2, 0.501, 0.05)
    ax.set_xticks(ticks, [f"{x:.2f}" for x in ticks])
    ax.set_title("AUC vs stringency (38 melanoma patients)", fontsize=17)
    ax.set_xlabel("Discrimination stringency", fontsize=14)
    ax.set_ylabel("LOOCV AUC", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=12)
    discovery = ("uncapped search, 3 nonsignificant-motif stop" if uncapped
                 else f"up to {int(metadata.max_motifs)} enriched motifs/FOV")
    fig.text(0.5, 0.04,
             f"SMORE: {discovery}; unreported motifs absent; >5 failed folds withheld.\n"
             "FANMOD+ at 0.50: separate clean-grid LOOCV evaluation (38 folds).",
             ha="center", fontsize=10)
    stem = "smore_full_cism_auc" if float(metadata.path_probability) == 1 else "smore_10pct_cism_auc"
    for ext in ("png", "pdf"):
        fig.savefig(run_dir / f"{stem}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(run_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
