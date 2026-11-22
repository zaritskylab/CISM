#!/usr/bin/env python3
"""Plot optimized melanoma motif count against selected stringency."""

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
from scipy.stats import spearmanr

from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    args = parser.parse_args()

    counts = pd.read_csv(args.output_dir / "optimal_fold_union_motif_counts.csv")
    trials = pd.read_csv(args.run_root / "aggregate/all_trial_summaries.csv")
    shuffled = trials.loc[trials.shuffle_labels,
                          ["seed", "optimal_stringency", "n_selected_unique_motifs"]].copy()
    original = trials.loc[~trials.shuffle_labels]
    if len(original) != 1 or counts.seed.duplicated().any() or shuffled.seed.duplicated().any():
        raise ValueError("Expected one original and unique shuffled seeds")
    data = counts.merge(shuffled, on="seed", how="outer", validate="one_to_one", indicator=True)
    if not data._merge.eq("both").all() or data.optimal_stringency.isna().any():
        raise ValueError("Motif counts and optimized trial summaries must match")
    if not data.fold_union_discriminative_motifs.eq(data.n_selected_unique_motifs).all():
        raise ValueError("Fold-union motif counts disagree with trial summaries")
    data = data.drop(columns=["n_selected_unique_motifs", "_merge"])
    distances = pd.read_csv(args.output_dir / "permutation_distances_to_baseline_and_original.csv")
    distances = distances.loc[(distances.variant == "optimal") &
                              (distances.distribution == "cell_type")]
    if not distances.groupby("seed").patient_class.nunique().eq(2).all():
        raise ValueError("Expected NN and NP cell-type distances for every shuffle")
    distances = (distances.groupby("seed", as_index=False).js_vs_baseline.mean()
                 .rename(columns={"js_vs_baseline": "mean_cell_type_js_distance_from_baseline"}))
    if distances.seed.duplicated().any():
        raise ValueError("Expected unique shuffled seeds in distance data")
    data = data.merge(distances, on="seed", how="outer", validate="one_to_one", indicator=True)
    if not data._merge.eq("both").all() or data.mean_cell_type_js_distance_from_baseline.isna().any():
        raise ValueError("Every optimized trial needs a cell-type JS distance")
    data = data.drop(columns="_merge")
    data.to_csv(args.output_dir / "optimal_motif_count_vs_stringency.csv", index=False)

    rho, p_value = spearmanr(
        data.fold_union_discriminative_motifs, data.optimal_stringency)
    if pd.isna(rho) or pd.isna(p_value):
        raise ValueError("Spearman correlation is undefined")
    original_motifs = int(original.n_selected_unique_motifs.iloc[0])
    original_stringency = float(original.optimal_stringency.iloc[0])
    original_distances = pd.read_csv(
        args.output_dir / "original_direct_motif_induced_vs_general_distribution_tests.csv")
    original_distances = original_distances.loc[
        (original_distances.variant == "optimal") &
        (original_distances.distribution == "cell_type")]
    if set(original_distances.patient_class) != {"NN", "NP"} or len(original_distances) != 2:
        raise ValueError("Expected original NN and NP cell-type distances")
    original_js = float(original_distances.jensen_shannon_distance.mean())

    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        scatter = ax.scatter(data.fold_union_discriminative_motifs, data.optimal_stringency,
                             c=data.mean_cell_type_js_distance_from_baseline, cmap="viridis",
                             vmin=0, vmax=1,
                             s=48, alpha=0.85, edgecolors="white",
                             linewidths=0.35, label=f"Shuffles (n={len(data)})")
        ax.scatter([original_motifs], [original_stringency], s=105, marker="D",
                   color="#df5948", edgecolors="#222222", linewidths=0.9,
                   zorder=4, label=f"Original ({original_motifs}, {original_stringency:.3f})")
        ax.set(xlabel="Fold-union discriminative motifs", ylabel="Optimal stringency",
               title="Optimized-stringency permutation trials",
               xlim=(0, max(200, int(data.fold_union_discriminative_motifs.max()) + 10)),
               ylim=(0, max(0.7, float(data.optimal_stringency.max()) + 0.05)))
        ax.text(0.98, 0.97, f"Spearman rho = {rho:.3f}\np = {p_value:.2e}",
                ha="right", va="top", transform=ax.transAxes,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "square,pad=0.4"})
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#d7dee8", linewidth=0.6, alpha=0.65)
        ax.legend(frameon=False, loc="lower left")
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Mean cell-type JS distance to original general distribution")
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"optimal_motif_count_vs_stringency_spearman.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame([{"n_shuffles": len(data), "spearman_rho": rho,
                   "spearman_p": p_value, "original_motif_count": original_motifs,
                   "original_optimal_stringency": original_stringency,
                   "original_mean_cell_type_js_distance_from_baseline": original_js}]).to_csv(
                       args.output_dir / "optimal_motif_count_vs_stringency_spearman.csv", index=False)
    print(f"n={len(data)}, Spearman rho={rho:.6f}, p={p_value:.6g}; "
          f"original=({original_motifs}, {original_stringency:.6f}), "
          f"original mean cell-type JS={original_js:.6f}")


if __name__ == "__main__":
    main()
