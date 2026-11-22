#!/usr/bin/env python3
"""Compare optimized motif-induced distances from original tissue baselines."""

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
from scipy.spatial.distance import jensenshannon

from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_OUTPUT_ROOT


COMPARISONS = (
    ("cell_type", "NN", "shuffled_mean_cell_type_vs_original_general_distributions.csv",
     "shuffled_mean_cell_type_proportion", "original_general_cell_proportion"),
    ("cell_type", "NP", "shuffled_mean_cell_type_vs_original_general_distributions.csv",
     "shuffled_mean_cell_type_proportion", "original_general_cell_proportion"),
    ("pairwise", "NN", "shuffled_mean_pairwise_vs_original_general_distributions.csv",
     "shuffled_mean_pairwise_proportion", "original_general_pairwise_proportion"),
    ("pairwise", "NP", "shuffled_mean_pairwise_vs_original_general_distributions.csv",
     "shuffled_mean_pairwise_proportion", "original_general_pairwise_proportion"),
)


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    args = parser.parse_args()
    root = args.output_dir
    distances = pd.read_csv(root / "permutation_distances_to_baseline_and_original.csv")
    original = pd.read_csv(root / "original_direct_motif_induced_vs_general_distribution_tests.csv")
    records = []
    nulls = {}

    for distribution, patient_class, filename, mean_col, baseline_col in COMPARISONS:
        frame = pd.read_csv(root / filename)
        frame = frame.loc[(frame.variant == "optimal") & (frame.patient_class == patient_class)]
        if frame.empty or frame[[mean_col, baseline_col]].isna().any().any():
            raise ValueError(f"Missing mean or baseline: {distribution}, {patient_class}")
        if not np.isclose(frame[mean_col].sum(), 1) or not np.isclose(frame[baseline_col].sum(), 1):
            raise ValueError(f"Distribution proportions do not sum to 1: {distribution}, {patient_class}")
        centroid_js = float(jensenshannon(frame[mean_col], frame[baseline_col], base=2.0))

        original_rows = original.loc[(original.run_type == "original") &
                                     (original.variant == "optimal") &
                                     (original.distribution == distribution) &
                                     (original.patient_class == patient_class)]
        if len(original_rows) != 1:
            raise ValueError(f"Expected one original distance: {distribution}, {patient_class}")
        original_js = float(original_rows.jensen_shannon_distance.iloc[0])

        null = distances.loc[(distances.variant == "optimal") &
                             (distances.distribution == distribution) &
                             (distances.patient_class == patient_class),
                             ["seed", "js_vs_baseline"]].copy()
        if len(null) != 100 or null.seed.duplicated().any() or null.isna().any().any():
            raise ValueError(f"Expected 100 complete shuffled distances: {distribution}, {patient_class}")
        values = null.js_vs_baseline.to_numpy(dtype=float)
        n_ge = int(np.count_nonzero(values >= original_js))
        n_le = int(np.count_nonzero(values <= original_js))
        p_more_distant = (n_ge + 1) / (len(values) + 1)
        p_less_distant = (n_le + 1) / (len(values) + 1)
        nulls[(distribution, patient_class)] = values
        records.append({
            "variant": "optimal",
            "distribution": distribution,
            "patient_class": patient_class,
            "n_categories": len(frame),
            "n_shuffles": len(values),
            "shuffled_mean_js_to_original_general": centroid_js,
            "mean_individual_shuffle_js": float(np.mean(values)),
            "mean_individual_minus_shuffled_mean_js": float(np.mean(values) - centroid_js),
            "original_motif_js_to_original_general": original_js,
            "original_minus_shuffled_mean_js": original_js - centroid_js,
            "median_individual_shuffle_js": float(np.median(values)),
            "shuffles_at_least_as_distant_as_original": n_ge,
            "shuffles_at_most_as_distant_as_original": n_le,
            "empirical_p_original_more_distant": p_more_distant,
            "empirical_p_original_less_distant": p_less_distant,
            "empirical_p_two_sided": min(1.0, 2 * min(p_more_distant, p_less_distant)),
        })

    results = pd.DataFrame(records)
    results["holm_p_two_sided_across_four_tests"] = holm_adjust(
        results.empirical_p_two_sided.to_numpy(dtype=float))
    results.to_csv(root / "optimal_motif_vs_shuffled_mean_baseline_distance_tests.csv", index=False)

    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True,
                                 sharex=True, sharey=True)
        for ax, row in zip(axes.flat, results.itertuples(index=False)):
            values = nulls[(row.distribution, row.patient_class)]
            ax.hist(values, bins=np.linspace(0, 1, 21), color="#5d82ae",
                    edgecolor="white", linewidth=0.5, label="Individual shuffles")
            ax.axvline(row.shuffled_mean_js_to_original_general, color="#26846a",
                       linewidth=2, label="Distance of pooled composition")
            ax.axvline(row.mean_individual_shuffle_js, color="#8856a7",
                       linewidth=2, linestyle="--", label="Mean individual distance")
            ax.axvline(row.original_motif_js_to_original_general, color="#bd4b38",
                       linewidth=2, label="Unpermuted motif-induced")
            ax.set_title(f"{'Cell types' if row.distribution == 'cell_type' else 'Pairwise'} | {row.patient_class}")
            ax.text(0.97, 0.94, f"Permutation rank p = {row.empirical_p_two_sided:.3f}\n"
                    f"Holm p = {row.holm_p_two_sided_across_four_tests:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "#dddddd", "pad": 4})
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#d7dee8", alpha=0.6)
            ax.set_axisbelow(True)
        for ax in axes[-1]:
            ax.set_xlabel("JS distance to original general distribution (base 2)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Shuffled trials")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2,
                   frameon=False)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(root / f"optimal_motif_vs_shuffled_mean_baseline_distance_tests.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True,
                                 constrained_layout=True)
        colors = ("#26846a", "#8856a7", "#bd4b38")
        names = ("Distance of pooled\nshuffled composition",
                 "Mean of individual\nshuffle distances",
                 "Unpermuted motif-\ninduced distance")
        for ax, distribution in zip(axes, ("cell_type", "pairwise")):
            subset = results.loc[results.distribution == distribution]
            for index, row in enumerate(subset.itertuples(index=False)):
                positions = np.arange(3) + (index - 0.5) * 0.32
                heights = (row.shuffled_mean_js_to_original_general,
                           row.mean_individual_shuffle_js,
                           row.original_motif_js_to_original_general)
                bars = ax.bar(positions, heights, width=0.29,
                              color=colors, alpha=1 if index == 0 else 0.62,
                              edgecolor="#27313a" if index == 0 else "none",
                              linewidth=0.6)
                for bar, height in zip(bars, heights):
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015,
                            f"{height:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(np.arange(3), names)
            ax.set_title("Cell types" if distribution == "cell_type" else "Pairwise interactions")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", color="#d7dee8", alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("JS distance to original general distribution (base 2)")
        fig.legend([plt.Rectangle((0, 0), 1, 1, facecolor="#6b6b6b", edgecolor="#27313a"),
                    plt.Rectangle((0, 0), 1, 1, facecolor="#b0b0b0")],
                   ["NN (outlined)", "NP (lighter)"], loc="upper center",
                   bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(root / f"optimal_distance_of_mean_vs_mean_distance.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    lines = [
        "# Optimized motif-induced distance comparison",
        "",
        "All JS distances use base 2 and the original, non-motif-induced tissue distribution",
        "of the same patient class as a fixed reference. Cell-type tests use all 16 categories;",
        "pairwise tests use all 225 directed matrix entries.",
        "",
        "The previously labeled shuffled-mean distance is actually JS(normalized pooled",
        "shuffled motif counts, original general proportions). The reported pooled counts",
        "do not reproduce the arithmetic mean of the saved per-seed distribution CSVs.",
        "Accordingly, it is a descriptive pooled-composition distance, not an arithmetic",
        "mean-distribution distance or a permutation-test statistic.",
        "The mean individual distance is the arithmetic mean of the 100 JS distances",
        "from each shuffled motif-induced distribution to that same baseline. These differ",
        "because JS distance is nonlinear and pooling weights seed compositions by their",
        "motif counts. Neither summary is a p-value.",
        "The permutation-rank test asks whether the unpermuted distance is unusually",
        "far from the tissue baseline relative to individual shuffled trials. Equivalence",
        "would require a prespecified maximum meaningful JS distance and an appropriate",
        "sampling design. A separate conditional multinomial goodness-of-fit test of",
        "the pooled counts appears in optimized_pooled_and_unpermuted_vs_general_tests.csv;",
        "a trial-level bootstrap is not justified because the saved per-seed vectors do",
        "not reconstruct the pooled composition.",
        "The inferential statistic is the unpermuted motif-induced JS distance compared with",
        "the 100 distances from individual shuffled-label trials to the same original general baseline.",
        "One-sided empirical tails use (1 + number of shuffles at least as extreme) / 101.",
        "Two-sided p is twice the smaller tail, capped at 1; Holm adjusts the four tests.",
        "Mann-Whitney is not applicable: there is one unpermuted distance, not a sample of",
        "independent unpermuted distances. Categories within a composition are dependent.",
        "",
        "| Distribution | Class | JS of pooled shuffle | Mean shuffled JS | Unpermuted JS | Shuffles farther | Empirical two-sided p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.distribution} | {row.patient_class} | {row.shuffled_mean_js_to_original_general:.3f} "
            f"| {row.mean_individual_shuffle_js:.3f} | {row.original_motif_js_to_original_general:.3f} "
            f"| {row.shuffles_at_least_as_distant_as_original}/{row.n_shuffles} "
            f"| {row.empirical_p_two_sided:.3f} "
            f"| {row.holm_p_two_sided_across_four_tests:.3f} |"
        )
    (root / "optimal_motif_vs_shuffled_mean_baseline_distance_tests.md").write_text(
        "\n".join(lines) + "\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
