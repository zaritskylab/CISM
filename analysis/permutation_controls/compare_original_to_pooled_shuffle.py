#!/usr/bin/env python3
"""Exploratory count-model comparison of original motifs with pooled shuffles."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from analysis.permutation_controls.run_cism_label_permutation_test import (
    DEFAULT_ARTIFACT,
    DEFAULT_OUTPUT_ROOT,
    deserialize_graph,
)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = running
    return adjusted


def original_motif_counts(
    motif_objects: pd.Series,
    selected: pd.DataFrame,
    cell_names: dict[int, str],
    patient_class: str,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    nodes = {name: 0 for name in cell_names.values()}
    edges = {(source, target): 0 for source in cell_names.values()
             for target in cell_names.values()}
    for motif_id in selected.loc[selected.patient_class == patient_class, "ID"]:
        graph = deserialize_graph(motif_objects.loc[motif_id])
        for _, attributes in graph.nodes(data=True):
            nodes[cell_names[int(attributes["type"])]] += 1
        for source, target in nx.Graph(graph).edges():
            source_type = cell_names[int(graph.nodes[source]["type"])]
            target_type = cell_names[int(graph.nodes[target]["type"])]
            edges[(source_type, target_type)] += 1
    return nodes, edges


def compare_by_simulation(
    observed: np.ndarray,
    pooled: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float, int, int]:
    total = int(observed.sum())
    observed_vs_pooled = float(jensenshannon(observed, pooled, base=2.0))
    observed_vs_baseline = float(jensenshannon(observed, baseline, base=2.0))
    n_as_different = 0
    n_as_far = 0
    for start in range(0, replicates, 1000):
        size = min(1000, replicates - start)
        simulated = rng.multinomial(total, pooled, size=size) / total
        pooled_matrix = np.broadcast_to(pooled, simulated.shape)
        baseline_matrix = np.broadcast_to(baseline, simulated.shape)
        n_as_different += int(np.count_nonzero(
            jensenshannon(simulated, pooled_matrix, axis=1, base=2.0)
            >= observed_vs_pooled
        ))
        n_as_far += int(np.count_nonzero(
            jensenshannon(simulated, baseline_matrix, axis=1, base=2.0)
            >= observed_vs_baseline
        ))
    return observed_vs_pooled, observed_vs_baseline, n_as_different, n_as_far


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=20261119)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")

    selected = pd.read_csv(
        args.output_dir.parent / "original" / "from_artifact" / "selected_motifs_by_class.csv"
    )
    direct = pd.read_csv(
        args.output_dir / "original_direct_motif_induced_vs_general_distribution_tests.csv"
    )
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    motif_table = artifact["cism"].motifs_dataset
    motif_objects = motif_table.loc[
        motif_table.ID.isin(selected.ID)
    ].drop_duplicates("ID").set_index("ID")["motif"]
    if not set(selected.ID).issubset(motif_objects.index):
        raise ValueError("Original selected motif IDs are missing from the CISM artifact")
    cell_names = artifact["discriminator"].common_cells
    rng = np.random.default_rng(args.random_seed)
    rows = []

    for patient_class in ("NN", "NP"):
        node_counts, edge_counts = original_motif_counts(
            motif_objects, selected, cell_names, patient_class
        )
        for distribution in ("cell_type", "pairwise"):
            frame = pd.read_csv(
                args.output_dir
                / f"shuffled_mean_{distribution}_vs_original_general_distributions.csv"
            )
            frame = frame.loc[
                (frame.variant == "optimal")
                & (frame.patient_class == patient_class)
            ]
            if distribution == "cell_type":
                observed = np.asarray(
                    [node_counts[cell] for cell in frame.cell_type], dtype=float
                )
                pooled = frame.shuffled_mean_cell_type_proportion.to_numpy(dtype=float)
                baseline = frame.original_general_cell_proportion.to_numpy(dtype=float)
                pooled_count = int(frame.shuffled_pooled_motif_node_count.sum())
            else:
                observed = np.asarray(
                    [edge_counts[(source, target)] for source, target
                     in zip(frame.source_cell_type, frame.target_cell_type)], dtype=float
                )
                pooled = frame.shuffled_mean_pairwise_proportion.to_numpy(dtype=float)
                baseline = frame.original_general_pairwise_proportion.to_numpy(dtype=float)
                pooled_count = int(frame.shuffled_pooled_motif_edge_count.sum())
            if (not len(frame) or np.any(observed < 0)
                    or not np.isclose(pooled.sum(), 1.0)
                    or not np.isclose(baseline.sum(), 1.0)):
                raise ValueError(f"Invalid composition: {distribution}, {patient_class}")
            direct_rows = direct.loc[
                (direct.run_type == "original")
                & (direct.variant == "optimal")
                & (direct.distribution == distribution)
                & (direct.patient_class == patient_class)
            ]
            if len(direct_rows) != 1:
                raise ValueError(f"Missing direct original test: {distribution}, {patient_class}")
            direct_row = direct_rows.iloc[0]
            values = compare_by_simulation(
                observed, pooled, baseline, rng, args.replicates
            )
            js_to_pooled, js_to_baseline, different, farther = values
            if (not np.isclose(observed.sum(), direct_row.observed_total)
                    or not np.isclose(js_to_baseline, direct_row.jensen_shannon_distance)):
                raise ValueError(
                    f"Reconstructed motifs disagree with original test: {distribution}, {patient_class}"
                )
            rows.append({
                "variant": "optimal",
                "distribution": distribution,
                "patient_class": patient_class,
                "unpermuted_motif_count": int(observed.sum()),
                "pooled_shuffle_motif_count": pooled_count,
                "js_unpermuted_vs_pooled_shuffle": js_to_pooled,
                "p_unpermuted_vs_pooled_shuffle": (different + 1) / (args.replicates + 1),
                "js_unpermuted_vs_general": js_to_baseline,
                "js_pooled_shuffle_vs_general": float(jensenshannon(pooled, baseline, base=2.0)),
                "p_unpermuted_farther_from_general_than_pooled_model":
                    (farther + 1) / (args.replicates + 1),
                "mc_replicates": args.replicates,
                "mc_random_seed": args.random_seed,
                "mc_exceedances_difference": different,
                "mc_exceedances_farther": farther,
            })

    result = pd.DataFrame(rows)
    for test in ("p_unpermuted_vs_pooled_shuffle",
                 "p_unpermuted_farther_from_general_than_pooled_model"):
        result[f"holm_{test}"] = holm_adjust(result[test].to_numpy(dtype=float))
    output = args.output_dir / "unpermuted_vs_pooled_shuffle_direct_comparison.csv"
    result.to_csv(output, index=False)

    lines = [
        "# Unpermuted motifs versus pooled shuffled motifs",
        "",
        "Unpermuted motifs are reconstructed as unweighted node and directed edge",
        "counts from the original selected motif IDs. The counts and JS distances",
        "to the tissue baseline are checked against the existing direct-test artifact.",
        "The plotted shuffled composition is the normalized pooled count vector from",
        "100 label permutations, treated as a fixed reference in this exploratory test.",
        "",
        f"Under that fixed-reference model, {args.replicates:,} multinomial samples at the",
        "unpermuted motif count are drawn from the pooled shuffled composition.",
        "The first upper-tail Monte Carlo p-value tests whether the unpermuted",
        "composition differs from the pooled composition using base-2 JS distance.",
        "The second tests whether the unpermuted composition is farther from the",
        "original tissue baseline than simulated counts from the pooled composition.",
        "All p-values use a plus-one correction; Holm adjustment covers four classes",
        "and distributions within each test family.",
        "",
        "These are conditional count-model p-values, not patient-level permutation",
        "p-values: motif nodes/edges are dependent and the pooled reference reuses the",
        "same tissue across 100 shuffles. They cannot establish a biological effect.",
        "",
        "| Distribution | Class | Unpermuted n | JS to pooled | Direct p | JS to tissue | Pooled JS to tissue | Farther p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.distribution} | {row.patient_class} | {row.unpermuted_motif_count} "
            f"| {row.js_unpermuted_vs_pooled_shuffle:.3f} "
            f"| {row.p_unpermuted_vs_pooled_shuffle:.3g} "
            f"| {row.js_unpermuted_vs_general:.3f} "
            f"| {row.js_pooled_shuffle_vs_general:.3f} "
            f"| {row.p_unpermuted_farther_from_general_than_pooled_model:.3g} |"
        )
    (args.output_dir / "unpermuted_vs_pooled_shuffle_direct_comparison.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
