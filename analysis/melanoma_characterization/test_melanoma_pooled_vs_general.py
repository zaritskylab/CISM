#!/usr/bin/env python3
"""Compare optimized pooled and unpermuted motif compositions with tissue baselines."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_OUTPUT_ROOT


def monte_carlo_js_p(
    baseline: np.ndarray,
    observed_total: int,
    observed_js: float,
    rng: np.random.Generator,
    replicates: int,
    batch_size: int = 1000,
) -> tuple[int, float]:
    exceedances = 0
    for start in range(0, replicates, batch_size):
        n_batch = min(batch_size, replicates - start)
        simulated = rng.multinomial(observed_total, baseline, size=n_batch) / observed_total
        distances = jensenshannon(
            simulated, np.broadcast_to(baseline, simulated.shape), axis=1, base=2.0
        )
        exceedances += int(np.count_nonzero(distances >= observed_js))
    return exceedances, (exceedances + 1) / (replicates + 1)


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
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=20261119)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")

    root = args.output_dir
    original = pd.read_csv(root / "original_direct_motif_induced_vs_general_distribution_tests.csv")
    distances = pd.read_csv(root / "permutation_distances_to_baseline_and_original.csv")
    rng = np.random.default_rng(args.random_seed)
    records = []

    for distribution in ("cell_type", "pairwise"):
        frame = pd.read_csv(
            root / f"shuffled_mean_{distribution}_vs_original_general_distributions.csv"
        )
        count_col = (
            "shuffled_pooled_motif_node_count"
            if distribution == "cell_type" else "shuffled_pooled_motif_edge_count"
        )
        baseline_col = (
            "original_general_cell_proportion"
            if distribution == "cell_type" else "original_general_pairwise_proportion"
        )
        for patient_class in ("NN", "NP"):
            subset = frame.loc[
                (frame.variant == "optimal") & (frame.patient_class == patient_class)
            ]
            baseline = subset[baseline_col].to_numpy(dtype=float)
            counts = subset[count_col].to_numpy(dtype=float)
            if (not len(subset) or subset[[baseline_col, count_col]].isna().any().any()
                    or np.any(baseline < 0) or np.any(counts < 0)
                    or not np.isclose(baseline.sum(), 1.0)
                    or not np.allclose(counts, np.rint(counts))):
                raise ValueError(f"Invalid plotted composition: {distribution}, {patient_class}")
            pooled_total = int(round(counts.sum()))
            pooled_js = float(jensenshannon(counts, baseline, base=2.0))

            original_rows = original.loc[
                (original.run_type == "original")
                & (original.variant == "optimal")
                & (original.distribution == distribution)
                & (original.patient_class == patient_class)
            ]
            if len(original_rows) != 1:
                raise ValueError(f"Expected one unpermuted result: {distribution}, {patient_class}")
            original_row = original_rows.iloc[0]
            original_total = int(round(original_row.observed_total))
            original_js = float(original_row.jensen_shannon_distance)

            null = distances.loc[
                (distances.variant == "optimal")
                & (distances.distribution == distribution)
                & (distances.patient_class == patient_class),
                ["seed", "js_vs_baseline"],
            ]
            if len(null) != 100 or null.seed.duplicated().any() or null.isna().any().any():
                raise ValueError(f"Expected 100 shuffled distances: {distribution}, {patient_class}")
            farther = int(np.count_nonzero(null.js_vs_baseline.to_numpy() > original_js))

            pooled_exceed, pooled_p = monte_carlo_js_p(
                baseline, pooled_total, pooled_js, rng, args.replicates
            )
            original_exceed, original_p = monte_carlo_js_p(
                baseline, original_total, original_js, rng, args.replicates
            )
            records.append({
                "variant": "optimal",
                "distribution": distribution,
                "patient_class": patient_class,
                "n_categories": len(subset),
                "pooled_shuffle_count": pooled_total,
                "pooled_shuffle_js_to_general": pooled_js,
                "pooled_shuffle_gof_mc_exceedances": pooled_exceed,
                "pooled_shuffle_gof_mc_p": pooled_p,
                "unpermuted_motif_count": original_total,
                "unpermuted_js_to_general": original_js,
                "unpermuted_gof_mc_exceedances": original_exceed,
                "unpermuted_gof_mc_p": original_p,
                "n_shuffles": len(null),
                "shuffles_farther_than_unpermuted": farther,
                "fraction_shuffles_farther_than_unpermuted": farther / len(null),
                "empirical_p_unpermuted_more_distant": (farther + 1) / (len(null) + 1),
                "mc_replicates": args.replicates,
                "mc_random_seed": args.random_seed,
            })

    result = pd.DataFrame(records)
    all_p = np.concatenate([
        result.pooled_shuffle_gof_mc_p.to_numpy(),
        result.unpermuted_gof_mc_p.to_numpy(),
    ])
    adjusted = holm_adjust(all_p)
    result["pooled_shuffle_gof_mc_holm_p"] = adjusted[:len(result)]
    result["unpermuted_gof_mc_holm_p"] = adjusted[len(result):]
    result.to_csv(root / "optimized_pooled_and_unpermuted_vs_general_tests.csv", index=False)

    lines = [
        "# Optimized motif-induced compositions versus tissue baselines",
        "",
        "The original, same-class non-motif-induced tissue composition is the fixed baseline.",
        "The plotted shuffled composition is normalized *pooled* motif counts across 100",
        "label permutations; it is not the arithmetic mean of the saved per-seed vectors.",
        "The unpermuted composition uses its original selected-motif counts. All distances",
        "are base-2 Jensen-Shannon distances.",
        "",
        f"For each goodness-of-fit test, {args.replicates:,} multinomial samples of the same",
        "count total were drawn from the tissue baseline. The upper-tail p-value is",
        "(1 + number of simulated JS distances at least as large as observed) /",
        f"({args.replicates} + 1). Holm adjustment covers all eight tests. This avoids",
        "the sparse-cell chi-square approximation, but assumes independent motif-count",
        "events. Motif nodes/edges overlap and all permutations reuse the same tissue,",
        "so these are conditional model-based tests, not patient-level biological p-values.",
        "Rejecting exact equality does not establish that a composition is far in a",
        "scientifically meaningful sense; failing to reject would not prove equivalence.",
        "",
        "The farther-away fraction counts individual shuffled trials whose JS distance",
        "to the same baseline is strictly greater than the unpermuted JS distance.",
        "Its empirical upper-tail permutation p-value uses the plus-one correction.",
        "",
        "| Distribution | Class | Pooled JS | Pooled GOF p | Unpermuted JS | Unpermuted GOF p | Shuffles farther | Empirical upper-tail p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.distribution} | {row.patient_class} | {row.pooled_shuffle_js_to_general:.3f} "
            f"| {row.pooled_shuffle_gof_mc_p:.3g} | {row.unpermuted_js_to_general:.3f} "
            f"| {row.unpermuted_gof_mc_p:.3g} "
            f"| {row.shuffles_farther_than_unpermuted}/{row.n_shuffles} "
            f"| {row.empirical_p_unpermuted_more_distant:.3f} |"
        )
    (root / "optimized_pooled_and_unpermuted_vs_general_tests.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
