#!/usr/bin/env python3
"""Compare each optimized motif composition with its own-label general composition."""

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
import numpy as np
import pandas as pd
from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_OUTPUT_ROOT


KINDS = ("cell_type", "pairwise")
CLASSES = ("NN", "NP")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    runs_dir = args.runs_dir
    output_dir = args.output_dir or runs_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The direct-test table uses unweighted motif node/edge counts and reads
    # general distributions from each run directory, preserving the prior metric.
    source = pd.read_csv(output_dir / "direct_motif_induced_vs_general_distribution_tests.csv")
    distances = source.loc[source.variant == "optimal", [
        "run_type", "seed", "distribution", "patient_class", "n_selected_motifs",
        "observed_total", "jensen_shannon_distance",
    ]].rename(columns={"jensen_shannon_distance": "js_to_own_general"})
    for kind in KINDS:
        for patient_class in CLASSES:
            subset = distances.loc[(distances.distribution == kind)
                                   & (distances.patient_class == patient_class)]
            perm = subset.loc[subset.run_type == "permutation"]
            if (len(subset.loc[subset.run_type == "original"]) != 1 or len(perm) != 100
                    or perm.seed.duplicated().any() or set(perm.seed.astype(int)) != set(range(100))):
                raise ValueError(f"Expected one original and seeds 0-99: {kind}, {patient_class}")
            if not subset.loc[subset.run_type == "original", "js_to_own_general"].notna().all():
                raise ValueError(f"Undefined original distance: {kind}, {patient_class}")
            if not (subset.js_to_own_general.isna() == (subset.observed_total == 0)).all():
                raise ValueError(f"Unexpected missing distance: {kind}, {patient_class}")
    distance_path = output_dir / "optimal_motif_distances_to_own_general.csv"
    distances.to_csv(distance_path, index=False)

    results = []
    for kind in KINDS:
        for patient_class in CLASSES:
            subset = distances.loc[(distances.distribution == kind)
                                   & (distances.patient_class == patient_class)]
            original = float(subset.loc[subset.run_type == "original", "js_to_own_general"].iloc[0])
            shuffled = subset.loc[subset.run_type == "permutation", "js_to_own_general"]
            valid = shuffled.dropna().to_numpy(dtype=float)
            n_ge = int(np.count_nonzero(valid >= original))
            n_le = int(np.count_nonzero(valid <= original))
            upper = (n_ge + 1) / (len(valid) + 1)
            lower = (n_le + 1) / (len(valid) + 1)
            results.append({
                "distribution": kind, "patient_class": patient_class,
                "original_js_to_own_general": original,
                "mean_shuffle_js_to_own_general": float(np.mean(valid)),
                "median_shuffle_js_to_own_general": float(np.median(valid)),
                "n_valid_shuffles": len(valid),
                "n_undefined_no_motifs": int(shuffled.isna().sum()),
                "n_shuffles_farther_than_original": n_ge,
                "fraction_shuffles_farther_than_original": n_ge / len(valid),
                "empirical_p_original_more_distant": upper,
                "empirical_p_original_less_distant": lower,
                "empirical_p_two_sided": min(1.0, 2 * min(upper, lower)),
            })
    summary = pd.DataFrame(results)
    summary["holm_p_two_sided_across_four_tests"] = holm_adjust(
        summary.empirical_p_two_sided.to_numpy())
    summary_path = output_dir / "optimal_motif_vs_shuffles_own_general_tests.csv"
    summary.to_csv(summary_path, index=False)

    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True,
                                 constrained_layout=True)
        for ax, row in zip(axes.flat, summary.itertuples(index=False)):
            values = distances.loc[
                (distances.run_type == "permutation")
                & (distances.distribution == row.distribution)
                & (distances.patient_class == row.patient_class), "js_to_own_general"
            ].dropna()
            ax.hist(values, bins=np.linspace(0, 1, 21), color="#5d82ae",
                    edgecolor="white", linewidth=0.5)
            ax.axvline(row.original_js_to_own_general, color="#bd4b38", linewidth=2,
                       label="Unpermuted")
            ax.set_title(f"{'Cell types' if row.distribution == 'cell_type' else 'Pairwise'} | {row.patient_class}")
            ax.text(0.97, 0.94,
                    f"Valid shuffles: {row.n_valid_shuffles}/100\n"
                    f"Two-sided p = {row.empirical_p_two_sided:.3f}\n"
                    f"Holm p = {row.holm_p_two_sided_across_four_tests:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "#dddddd", "pad": 4})
            ax.grid(axis="y", color="#d7dee8", alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("JS distance to each run's own general distribution (base 2)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Shuffled trials")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 1.04), frameon=False)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"optimal_motif_vs_shuffles_own_general.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(summary.to_string(index=False))
    print(f"\nPer-run distances: {distance_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
