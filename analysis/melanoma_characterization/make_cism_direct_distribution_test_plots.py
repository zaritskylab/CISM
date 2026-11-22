#!/usr/bin/env python
"""Direct motif-induced-vs-general distribution tests for CISM permutations."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import ast
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, power_divergence

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.helpers import string_base64_pickle


DEFAULT_ROOT = (
    Path("melanoma_tutorials")
    / "melanoma_tutorials"
    / "runtime"
    / "cism_label_permutation_test"
)
DEFAULT_ARTIFACT = (
    Path("melanoma_tutorials")
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
VARIANTS = {"optimal": "Optimal stringency", "fixed": "Fixed stringency 0.46"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ROOT / "original" / "from_artifact")
    parser.add_argument("--shuffle-root", type=Path, default=DEFAULT_ROOT / "shuffled")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "report")
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    discriminator = artifact["discriminator"]
    cell_names = {int(k): v for k, v in discriminator.common_cells.items()}
    motif_table = discriminator.cism.motifs_dataset
    used_motifs = collect_used_motif_ids(args.original_dir, args.shuffle_root)
    motif_contributions = build_motif_contribution_cache(motif_table, cell_names, used_motifs)
    motif_occurrences = build_motif_occurrence_table(motif_table)

    rows = []
    rows.extend(
        run_direct_tests(
            run_dir=args.original_dir,
            run_type="original",
            seed=None,
            labels=args.labels,
            cell_names=cell_names,
            motif_contributions=motif_contributions,
            motif_occurrences=motif_occurrences,
        )
    )
    for seed_dir in sorted(args.shuffle_root.glob("seed_*"), key=seed_sort_key):
        seed = int(seed_dir.name.removeprefix("seed_"))
        rows.extend(
            run_direct_tests(
                run_dir=seed_dir,
                run_type="permutation",
                seed=seed,
                labels=args.labels,
                cell_names=cell_names,
                motif_contributions=motif_contributions,
                motif_occurrences=motif_occurrences,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "direct_motif_induced_vs_general_distribution_tests.csv", index=False)
    original = results.loc[results["run_type"] == "original"].copy()
    original.to_csv(args.output_dir / "original_direct_motif_induced_vs_general_distribution_tests.csv", index=False)
    plot_p_value_histograms(results, args.output_dir)
    print(original.to_string(index=False))


def run_direct_tests(
    run_dir: Path,
    run_type: str,
    seed: int | None,
    labels: list[str],
    cell_names: dict[int, str],
    motif_contributions: dict[Any, tuple[np.ndarray, np.ndarray]],
    motif_occurrences: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    patient_class_df = read_patient_classes_frame(run_dir)
    general_cell = pd.read_csv(run_dir / "general_cell_type_distribution.csv")
    general_pairwise = pd.read_csv(run_dir / "general_pairwise_distribution.csv")
    for variant in VARIANTS:
        feature_rows = pd.read_csv(run_dir / f"{variant}_motif_space_features.csv")
        selected_motifs = sorted(
            {motif_id for value in feature_rows["features"] for motif_id in parse_feature_list(value)}
        )
        motifs_by_class = assign_motifs_to_classes(
            selected_motifs=selected_motifs,
            motif_occurrences=motif_occurrences,
            patient_class_df=patient_class_df,
            labels=labels,
        )
        for label in labels:
            cell_order = list(cell_names.values())
            cell_counts, pairwise_counts = class_count_vectors(
                motifs_by_class[label],
                motif_contributions,
            )
            cell_baseline = (
                general_cell.loc[general_cell["patient_class"] == label]
                .set_index("cell_type")
                .sort_index()["proportion"]
            )
            pairwise_baseline = (
                general_pairwise.loc[general_pairwise["patient_class"] == label]
                .set_index(["source_cell_type", "target_cell_type"])
                .sort_index()["proportion"]
            )
            cell_index = sorted(cell_baseline.index)
            pairwise_index = pairwise_baseline.index.sort_values()
            cell_obs = pd.Series(cell_counts, index=cell_order).reindex(cell_index).to_numpy(dtype=float)
            pairwise_obs = (
                pd.DataFrame(pairwise_counts, index=cell_order, columns=cell_order)
                .stack()
                .reindex(pairwise_index)
                .to_numpy(dtype=float)
            )
            rows.append(
                test_row(
                    run_type,
                    seed,
                    variant,
                    "cell_type",
                    label,
                    len(motifs_by_class[label]),
                    cell_obs,
                    cell_baseline.reindex(cell_index).to_numpy(dtype=float),
                )
            )
            rows.append(
                test_row(
                    run_type,
                    seed,
                    variant,
                    "pairwise",
                    label,
                    len(motifs_by_class[label]),
                    pairwise_obs,
                    pairwise_baseline.reindex(pairwise_index).to_numpy(dtype=float),
                )
            )
    return rows


def test_row(
    run_type: str,
    seed: int | None,
    variant: str,
    distribution: str,
    patient_class: str,
    n_selected_motifs: int,
    observed_counts: np.ndarray,
    baseline_proportions: np.ndarray,
) -> dict[str, Any]:
    observed_counts = np.asarray(observed_counts, dtype=float)
    baseline_proportions = np.asarray(baseline_proportions, dtype=float)
    observed_total = float(observed_counts.sum())
    baseline_proportions = normalize(baseline_proportions)
    expected_counts = np.clip(baseline_proportions * observed_total, 1e-12, None)
    expected_counts = expected_counts / expected_counts.sum() * observed_total if observed_total > 0 else expected_counts
    if observed_total > 0:
        chi = chisquare(observed_counts, expected_counts)
        g_test = power_divergence(observed_counts, expected_counts, lambda_="log-likelihood")
        js = float(jensenshannon(normalize(observed_counts), baseline_proportions, base=2.0))
        chi_stat = float(chi.statistic)
        chi_p = float(chi.pvalue)
        g_stat = float(g_test.statistic)
        g_p = float(g_test.pvalue)
    else:
        js = np.nan
        chi_stat = np.nan
        chi_p = np.nan
        g_stat = np.nan
        g_p = np.nan
    return {
        "run_type": run_type,
        "seed": seed,
        "variant": variant,
        "distribution": distribution,
        "patient_class": patient_class,
        "n_selected_motifs": n_selected_motifs,
        "observed_total": observed_total,
        "jensen_shannon_distance": js,
        "chi_square_statistic": chi_stat,
        "chi_square_p_value": chi_p,
        "g_test_statistic": g_stat,
        "g_test_p_value": g_p,
    }


def class_count_vectors(
    motif_ids: list[Any],
    motif_contributions: dict[Any, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    first = next(iter(motif_contributions.values()))
    cell_counts = np.zeros_like(first[0], dtype=float)
    pairwise_counts = np.zeros_like(first[1], dtype=float)
    for motif_id in motif_ids:
        if motif_id not in motif_contributions:
            continue
        cell_counts += motif_contributions[motif_id][0]
        pairwise_counts += motif_contributions[motif_id][1]
    return cell_counts, pairwise_counts


def plot_p_value_histograms(results: pd.DataFrame, output_dir: Path) -> None:
    permutations = results.loc[results["run_type"] == "permutation"].copy()
    p_value_bins = np.linspace(0, 0.2, 21)
    for test, column, filename in [
        ("Chi-square", "chi_square_p_value", "permutation_chi_square_p_value_histograms.png"),
        ("G-test", "g_test_p_value", "permutation_g_test_p_value_histograms.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
        for ax, ((variant, distribution), frame) in zip(
            axes.flat,
            permutations.groupby(["variant", "distribution"], sort=True),
        ):
            for label, color in [("NN", "#2f6db3"), ("NP", "#d97824")]:
                values = frame.loc[frame["patient_class"] == label, column].dropna()
                ax.hist(values, bins=p_value_bins, alpha=0.58, color=color, label=label)
            ax.axvline(0.05, color="black", linestyle="--", linewidth=1.2)
            ax.set_xlim(0, 0.2)
            ax.set_title(f"{VARIANTS[variant]}: {distribution}")
            ax.set_xlabel("p-value")
            ax.set_ylabel("Permutation count")
            ax.legend(frameon=False)
        fig.suptitle(f"Permutation {test} p-values vs same-class general baseline", fontsize=14)
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)


def collect_used_motif_ids(original_dir: Path, shuffle_root: Path) -> set[Any]:
    used = set()
    for run_dir in [original_dir] + sorted(shuffle_root.glob("seed_*"), key=seed_sort_key):
        for variant in VARIANTS:
            path = run_dir / f"{variant}_motif_space_features.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, usecols=["features"])
            for value in frame["features"]:
                used.update(parse_feature_list(value))
    return used


def parse_feature_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    return list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]


def build_motif_contribution_cache(
    motif_table: pd.DataFrame,
    cell_names: dict[int, str],
    used_motif_ids: set[Any],
) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
    cell_types = list(cell_names.values())
    cell_to_index = {cell_type: i for i, cell_type in enumerate(cell_types)}
    cache = {}
    selected_motif_table = motif_table.loc[motif_table["ID"].isin(used_motif_ids)].drop_duplicates("ID")
    for _, row in selected_motif_table.iterrows():
        graph = deserialize_graph(row["motif"])
        cell_vector = np.zeros(len(cell_types), dtype=float)
        pairwise_matrix = np.zeros((len(cell_types), len(cell_types)), dtype=float)
        for cell_type in nx.get_node_attributes(graph, name="type").values():
            cell_vector[cell_to_index[cell_names[int(cell_type)]]] += 1.0
        for left, right in nx.Graph(graph).edges():
            left_type = cell_names[int(graph.nodes[left]["type"])]
            right_type = cell_names[int(graph.nodes[right]["type"])]
            pairwise_matrix[cell_to_index[left_type], cell_to_index[right_type]] += 1.0
        cache[row["ID"]] = (cell_vector, pairwise_matrix)
    return cache


def build_motif_occurrence_table(motif_table: pd.DataFrame) -> pd.DataFrame:
    columns = ["ID", "Patient_uId"]
    occurrence = motif_table.loc[:, columns].copy()
    occurrence["Patient_uId"] = occurrence["Patient_uId"].astype(str)
    return occurrence


def assign_motifs_to_classes(
    selected_motifs: list[Any],
    motif_occurrences: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    labels: list[str],
) -> dict[str, list[Any]]:
    result = {label: [] for label in labels}
    relevant_patients = patient_class_df.loc[patient_class_df["patient_class"].isin(labels), "patient_class"]
    selected = motif_occurrences.loc[
        motif_occurrences["ID"].isin(selected_motifs)
        & motif_occurrences["Patient_uId"].isin(relevant_patients.index)
    ].copy()
    if selected.empty:
        return result
    selected["patient_class"] = selected["Patient_uId"].map(relevant_patients)
    counts = (
        selected.drop_duplicates(["ID", "Patient_uId"])
        .groupby(["ID", "patient_class"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for label in labels:
        if label not in counts.columns:
            counts[label] = 0
    for motif_id in selected_motifs:
        if motif_id not in counts.index:
            continue
        assigned = counts.loc[motif_id, labels].astype(float).idxmax()
        result[assigned].append(motif_id)
    return result


def read_patient_classes_frame(run_dir: Path) -> pd.DataFrame:
    candidates = [run_dir / "patient_class_permuted.csv", run_dir / "patient_class_original.csv"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No patient class CSV found in {run_dir}")
    frame = pd.read_csv(path, header=None, names=["patient_id", "patient_class"])
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["patient_class"] = frame["patient_class"].astype(str)
    return frame.set_index("patient_id")


def deserialize_graph(motif_obj: Any):
    if isinstance(motif_obj, (nx.Graph, nx.DiGraph)):
        return motif_obj
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    raise TypeError(f"Unsupported motif graph representation: {type(motif_obj)!r}")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    return values / total if total > 0 else values


def seed_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.name.removeprefix("seed_")), path.name)
    except ValueError:
        return (10**12, path.name)


if __name__ == "__main__":
    main()
