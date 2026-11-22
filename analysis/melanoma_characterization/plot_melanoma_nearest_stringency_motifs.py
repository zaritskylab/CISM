#!/usr/bin/env python3
"""Plot optimized motif counts against stringency and identify nearest shuffles."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = _REPO_ROOT
DEFAULT_REPORT = (ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
                  / "cism_label_permutation_test" / "report")
COUNT = "n_fold_union_discriminative_motifs"
STRINGENCY = "optimal_stringency"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--nearest", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv(args.report_dir / "permutation_metric_scatterplot_data.csv")
    original_rows = frame.loc[~frame.shuffle_labels]
    shuffles = frame.loc[frame.shuffle_labels].copy()
    if len(original_rows) != 1 or shuffles.seed.duplicated().any() or frame[[STRINGENCY, COUNT]].isna().any().any():
        raise ValueError("Expected one original and unique, complete shuffle metrics")
    if not 1 <= args.nearest <= len(shuffles):
        parser.error("--nearest must be between 1 and the number of shuffled runs")

    original = original_rows.iloc[0]
    shuffles["absolute_stringency_difference"] = (
        shuffles[STRINGENCY] - original[STRINGENCY]).abs()
    nearest = shuffles.sort_values(["absolute_stringency_difference", "seed"]).head(args.nearest).copy()
    nearest["motifs_vs_original"] = nearest[COUNT] - original[COUNT]
    nearest["rank_by_stringency_proximity"] = range(1, args.nearest + 1)
    nearest[["rank_by_stringency_proximity", "seed", STRINGENCY,
             "absolute_stringency_difference", COUNT, "motifs_vs_original"]].to_csv(
        args.report_dir / f"nearest_{args.nearest}_optimal_stringency_shuffles.csv", index=False)

    n_more = int((nearest[COUNT] > original[COUNT]).sum())
    fig, ax = plt.subplots(figsize=(8.7, 6.1), constrained_layout=True)
    ax.scatter(shuffles[STRINGENCY], shuffles[COUNT], s=43, color="#6c91b8",
               alpha=0.72, edgecolor="white", linewidth=0.4, label="Other shuffles", zorder=2)
    ax.scatter(nearest[STRINGENCY], nearest[COUNT], s=96, color="#e09b42",
               edgecolor="#744613", linewidth=0.8, label=f"{args.nearest} nearest shuffles", zorder=3)
    ax.scatter([original[STRINGENCY]], [original[COUNT]], s=155, marker="D",
               color="#b54435", edgecolor="#60251f", linewidth=1,
               label="Unpermuted", zorder=4)
    ax.axvline(original[STRINGENCY], color="#929aa1", linestyle="--", linewidth=1, zorder=1)
    ax.axhline(original[COUNT], color="#929aa1", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlabel("Optimal stringency")
    ax.set_ylabel("Discriminative motifs (fold union)")
    ax.set_title(f"Motif count vs optimal stringency | {n_more}/{args.nearest} nearest shuffles have more motifs")
    ax.set_xlim(left=min(0, shuffles[STRINGENCY].min() - 0.02))
    ax.set_ylim(bottom=0)
    ax.grid(color="#dbe1e6", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(args.report_dir / f"optimal_motifs_vs_stringency_nearest_{args.nearest}.{extension}",
                    dpi=300)
    plt.close(fig)
    print(f"Unpermuted: stringency={original[STRINGENCY]:.6f}, motifs={int(original[COUNT])}")
    print(nearest[["rank_by_stringency_proximity", "seed", STRINGENCY,
                   "absolute_stringency_difference", COUNT, "motifs_vs_original"]].to_string(index=False))
    print(f"More: {n_more}/{args.nearest}; equal: {int((nearest[COUNT] == original[COUNT]).sum())}; "
          f"fewer: {int((nearest[COUNT] < original[COUNT]).sum())}")


if __name__ == "__main__":
    main()
