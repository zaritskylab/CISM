#!/usr/bin/env python
"""Create a PDF/Markdown report for CISM label permutation results."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import ast
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, linregress, power_divergence

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
VARIANTS = {
    "optimal": "Optimal stringency",
    "fixed": "Fixed stringency 0.46",
}
CLASS_COLORS = {"NN": "#2f6db3", "NP": "#d97824"}
CLASS_CMAPS = {"NN": "Blues", "NP": "Reds"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_ROOT / "aggregate")
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ROOT / "original" / "from_artifact")
    parser.add_argument("--shuffle-root", type=Path, default=DEFAULT_ROOT / "shuffled")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "report")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    parser.add_argument("--ci-low", type=float, default=2.5)
    parser.add_argument("--ci-high", type=float, default=97.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(args.labels)

    trial_summaries = pd.read_csv(args.aggregate_dir / "all_trial_summaries.csv")
    artifact = load_artifact(args.artifact)
    motif_table = artifact["discriminator"].cism.motifs_dataset
    cell_names = {int(k): v for k, v in artifact["discriminator"].common_cells.items()}
    used_motif_ids = collect_used_motif_ids(args.original_dir, args.shuffle_root)
    motif_contributions = build_motif_contribution_cache(motif_table, cell_names, used_motif_ids)
    motif_occurrences = build_motif_occurrence_table(motif_table)

    variant_data = build_variant_distribution_data(
        original_dir=args.original_dir,
        shuffle_root=args.shuffle_root,
        cell_names=cell_names,
        labels=labels,
        motif_contributions=motif_contributions,
        motif_occurrences=motif_occurrences,
    )

    auc_summary = plot_auc_histograms(trial_summaries, args.output_dir)
    stringency_summary = plot_stringency_distribution(trial_summaries, args.output_dir)
    stats_tables = load_statistical_tables(args.aggregate_dir, trial_summaries, stringency_summary)

    cell_summaries = []
    pairwise_summaries = []
    variant_distribution_tests = []
    for variant, data in variant_data.items():
        cell_summary = summarize_distribution(
            data["cell_permutations"],
            ["patient_class", "cell_type"],
            args.ci_low,
            args.ci_high,
        )
        cell_summary.insert(0, "variant", variant)
        pairwise_summary = summarize_distribution(
            data["pairwise_permutations"],
            ["patient_class", "source_cell_type", "target_cell_type"],
            args.ci_low,
            args.ci_high,
        )
        pairwise_summary.insert(0, "variant", variant)
        cell_summaries.append(cell_summary)
        pairwise_summaries.append(pairwise_summary)
        variant_distribution_tests.extend(
            distribution_tests_for_variant(
                variant=variant,
                original=data["cell_original"],
                permutations=data["cell_permutations"],
                value_columns=["cell_type"],
                distribution_name="cell_type",
                labels=labels,
            )
        )
        variant_distribution_tests.extend(
            distribution_tests_for_variant(
                variant=variant,
                original=data["pairwise_original"],
                permutations=data["pairwise_permutations"],
                value_columns=["source_cell_type", "target_cell_type"],
                distribution_name="pairwise",
                labels=labels,
            )
        )

    cell_summary_all = pd.concat(cell_summaries, ignore_index=True)
    pairwise_summary_all = pd.concat(pairwise_summaries, ignore_index=True)
    variant_distribution_tests_df = pd.DataFrame(variant_distribution_tests)
    local_distribution_tests_df = build_local_distribution_tests(variant_data, labels)

    auc_summary.to_csv(args.output_dir / "auc_report_summary.csv", index=False)
    stringency_summary.to_csv(args.output_dir / "optimal_stringency_report_summary.csv", index=False)
    cell_summary_all.to_csv(args.output_dir / "motif_induced_cell_type_permutation_ci.csv", index=False)
    pairwise_summary_all.to_csv(args.output_dir / "motif_induced_pairwise_permutation_ci.csv", index=False)
    variant_distribution_tests_df.to_csv(
        args.output_dir / "variant_motif_induced_distribution_tests.csv",
        index=False,
    )
    local_distribution_tests_df.to_csv(
        args.output_dir / "local_motif_induced_distribution_tests.csv",
        index=False,
    )
    summarize_local_tests(local_distribution_tests_df).to_csv(
        args.output_dir / "local_motif_induced_distribution_test_summary.csv",
        index=False,
    )

    plot_cell_type_bars_by_variant(cell_summary_all, variant_data, labels, args.output_dir)
    plot_pairwise_heatmaps_by_variant(pairwise_summary_all, variant_data, labels, args.output_dir)
    plot_shuffled_mean_vs_original_general(args.output_dir, labels)
    plot_permutation_metric_scatterplots(trial_summaries, args.output_dir)
    write_report(args.output_dir, auc_summary, stringency_summary, stats_tables, len(variant_data["optimal"]["cell_permutations"]["seed"].unique()))
    write_pdf_report(args.output_dir, auc_summary, stringency_summary, stats_tables, len(variant_data["optimal"]["cell_permutations"]["seed"].unique()))


def load_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_variant_distribution_data(
    original_dir: Path,
    shuffle_root: Path,
    cell_names: dict[int, str],
    labels: list[str],
    motif_contributions: dict[Any, tuple[np.ndarray, np.ndarray]],
    motif_occurrences: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    result = {}
    for variant in VARIANTS:
        original_cell, original_pairwise = distributions_from_run_dir(
            original_dir,
            variant,
            cell_names,
            labels,
            seed=None,
            motif_contributions=motif_contributions,
            motif_occurrences=motif_occurrences,
        )
        permutation_cells = []
        permutation_pairwise = []
        for seed_dir in sorted(shuffle_root.glob("seed_*"), key=seed_sort_key):
            if not (seed_dir / f"{variant}_motif_space_features.csv").exists():
                continue
            seed = int(seed_dir.name.removeprefix("seed_"))
            cell_df, pairwise_df = distributions_from_run_dir(
                seed_dir,
                variant,
                cell_names,
                labels,
                seed=seed,
                motif_contributions=motif_contributions,
                motif_occurrences=motif_occurrences,
            )
            permutation_cells.append(cell_df)
            permutation_pairwise.append(pairwise_df)
        result[variant] = {
            "cell_original": original_cell,
            "pairwise_original": original_pairwise,
            "cell_permutations": pd.concat(permutation_cells, ignore_index=True),
            "pairwise_permutations": pd.concat(permutation_pairwise, ignore_index=True),
        }
    return result


def distributions_from_run_dir(
    run_dir: Path,
    variant: str,
    cell_names: dict[int, str],
    labels: list[str],
    seed: int | None,
    motif_contributions: dict[Any, tuple[np.ndarray, np.ndarray]],
    motif_occurrences: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_class_df = read_patient_classes_frame(run_dir)
    feature_rows = pd.read_csv(run_dir / f"{variant}_motif_space_features.csv")
    selected_motifs = sorted({motif_id for value in feature_rows["features"] for motif_id in parse_feature_list(value)})
    motifs_by_class = assign_motifs_to_classes(selected_motifs, motif_occurrences, patient_class_df, labels)
    motif_weights = motif_mean_weights(selected_motifs, motif_occurrences, patient_class_df, labels)
    cell_df = motif_aware_cell_type_distribution(motifs_by_class, motif_weights, cell_names, motif_contributions)
    pairwise_df = motif_aware_pairwise_distribution(motifs_by_class, motif_weights, cell_names, motif_contributions)
    if seed is not None:
        cell_df.insert(0, "seed", seed)
        pairwise_df.insert(0, "seed", seed)
    cell_df.insert(0, "variant", variant)
    pairwise_df.insert(0, "variant", variant)
    return cell_df, pairwise_df


def read_patient_classes_frame(run_dir: Path) -> pd.DataFrame:
    candidates = [run_dir / "patient_class_permuted.csv", run_dir / "patient_class_original.csv"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No patient class CSV found in {run_dir}")
    frame = pd.read_csv(path, header=None, names=["patient_id", "patient_class"])
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["patient_class"] = frame["patient_class"].astype(str)
    return frame.set_index("patient_id")


def collect_used_motif_ids(original_dir: Path, shuffle_root: Path) -> set[Any]:
    used = set()
    run_dirs = [original_dir] + sorted(shuffle_root.glob("seed_*"), key=seed_sort_key)
    for run_dir in run_dirs:
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


def motif_aware_cell_type_distribution(
    motifs_by_class: dict[str, list[Any]],
    motif_weights: pd.Series,
    cell_names: dict[int, str],
    motif_contributions: dict[Any, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    cell_types = list(cell_names.values())
    rows = []
    for patient_class, motif_ids in motifs_by_class.items():
        counts = np.zeros(len(cell_types), dtype=float)
        for motif_id in motif_ids:
            if motif_id not in motif_contributions:
                continue
            counts += float(motif_weights.get(motif_id, 1.0)) * motif_contributions[motif_id][0]
        total = float(counts.sum())
        for cell_type, count in zip(cell_types, counts):
            rows.append(
                {
                    "patient_class": patient_class,
                    "cell_type": cell_type,
                    "count": count,
                    "proportion": count / total if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def motif_aware_pairwise_distribution(
    motifs_by_class: dict[str, list[Any]],
    motif_weights: pd.Series,
    cell_names: dict[int, str],
    motif_contributions: dict[Any, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    cell_types = list(cell_names.values())
    rows = []
    for patient_class, motif_ids in motifs_by_class.items():
        matrix = np.zeros((len(cell_types), len(cell_types)), dtype=float)
        for motif_id in motif_ids:
            if motif_id not in motif_contributions:
                continue
            matrix += float(motif_weights.get(motif_id, 1.0)) * motif_contributions[motif_id][1]
        total = float(matrix.sum())
        proportions = matrix / total if total > 0 else matrix
        for source_i, source_cell_type in enumerate(cell_types):
            for target_i, target_cell_type in enumerate(cell_types):
                rows.append(
                    {
                        "patient_class": patient_class,
                        "source_cell_type": source_cell_type,
                        "target_cell_type": target_cell_type,
                        "count": float(matrix[source_i, target_i]),
                        "proportion": float(proportions[source_i, target_i]),
                    }
                )
    return pd.DataFrame(rows)


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
    columns = ["ID", "Patient_uId", "Patient", "Count", "Freq"]
    occurrence = motif_table.loc[:, columns].copy()
    occurrence["Patient_uId"] = occurrence["Patient_uId"].astype(str)
    occurrence["Total"] = occurrence["Count"] / occurrence["Freq"].replace(0, np.nan)
    occurrence["Total"] = occurrence["Total"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
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


def motif_mean_weights(
    selected_motifs: list[Any],
    motif_occurrences: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    labels: list[str],
) -> pd.Series:
    relevant_patients = patient_class_df.loc[patient_class_df["patient_class"].isin(labels)]
    selected = motif_occurrences.loc[
        motif_occurrences["ID"].isin(selected_motifs)
        & motif_occurrences["Patient_uId"].isin(relevant_patients.index)
    ].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    grouped = selected.groupby(["Patient", "ID"], observed=True)[["Count", "Total"]].sum().reset_index()
    grouped["Freq"] = grouped["Count"] / grouped["Total"].replace(0, np.nan)
    grouped["Freq"] = grouped["Freq"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return grouped.groupby("ID", observed=True)["Freq"].mean()


def deserialize_graph(motif_obj: Any):
    if isinstance(motif_obj, (nx.Graph, nx.DiGraph)):
        return motif_obj
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    raise TypeError(f"Unsupported motif graph representation: {type(motif_obj)!r}")


def seed_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.name.removeprefix("seed_")), path.name)
    except ValueError:
        return (math.inf, path.name)


def summarize_distribution(
    data: pd.DataFrame,
    group_columns: list[str],
    ci_low: float,
    ci_high: float,
) -> pd.DataFrame:
    grouped = data.groupby(group_columns, dropna=False)["proportion"]
    return grouped.agg(
        mean="mean",
        std="std",
        n="count",
        ci_low=lambda values: np.percentile(values, ci_low),
        ci_high=lambda values: np.percentile(values, ci_high),
    ).reset_index()


def plot_auc_histograms(trial_summaries: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    original = trial_summaries.loc[~trial_summaries["shuffle_labels"].astype(bool)].iloc[0]
    permutations = trial_summaries.loc[trial_summaries["shuffle_labels"].astype(bool)]
    metrics = [
        ("optimal_auc", "Optimal stringency AUC"),
        ("fixed_auc", "Fixed stringency 0.46 AUC"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    rows = []
    for ax, (column, title) in zip(axes, metrics):
        values = permutations[column].dropna()
        original_value = float(original[column])
        ax.hist(values, bins=16, color="#6c8ebf", edgecolor="white", alpha=0.9)
        ax.axvline(original_value, color="#b03a2e", linewidth=2.4, label=f"Original = {original_value:.3f}")
        ax.set_title(title)
        ax.set_xlabel("AUC")
        ax.set_ylabel("Permutation count")
        ax.legend(frameon=False)
        rows.append(summary_row(column, original_value, values))
    fig.suptitle("CISM label permutation AUCs", fontsize=14)
    fig.savefig(output_dir / "auc_permutation_histograms.png", dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_stringency_distribution(trial_summaries: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    original = trial_summaries.loc[~trial_summaries["shuffle_labels"].astype(bool)].iloc[0]
    permutations = trial_summaries.loc[trial_summaries["shuffle_labels"].astype(bool)]
    values = permutations["optimal_stringency"].dropna()
    original_value = float(original["optimal_stringency"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(values, bins=16, color="#7b8f62", edgecolor="white", alpha=0.9)
    axes[0].axvline(original_value, color="#b03a2e", linewidth=2.4, label=f"Original = {original_value:.3f}")
    axes[0].set_title("Optimal stringency histogram")
    axes[0].set_xlabel("Optimal stringency")
    axes[0].set_ylabel("Permutation count")
    axes[0].legend(frameon=False)
    axes[1].boxplot(values, vert=True, patch_artist=True, boxprops={"facecolor": "#c9d6b8"})
    axes[1].scatter([1], [original_value], color="#b03a2e", s=45, zorder=3, label="Original")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Permutations"])
    axes[1].set_ylabel("Optimal stringency")
    axes[1].set_title("Optimal stringency box plot")
    axes[1].legend(frameon=False)
    fig.suptitle("Optimal stringency under label permutation", fontsize=14)
    fig.savefig(output_dir / "optimal_stringency_distribution.png", dpi=300)
    plt.close(fig)
    row = summary_row("optimal_stringency", original_value, values)
    row["empirical_p_value_greater_equal"] = empirical_p_value(original_value, values.to_numpy(), greater=True)
    return pd.DataFrame([row])


def summary_row(metric: str, original_value: float, values: pd.Series) -> dict[str, Any]:
    return {
        "metric": metric,
        "original": original_value,
        "permutation_mean": values.mean(),
        "permutation_std": values.std(ddof=1),
        "permutation_min": values.min(),
        "permutation_max": values.max(),
        "n_permutations": len(values),
    }


def distribution_tests_for_variant(
    variant: str,
    original: pd.DataFrame,
    permutations: pd.DataFrame,
    value_columns: list[str],
    distribution_name: str,
    labels: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for patient_class in labels:
        original_vector = vector_for_distribution(original, value_columns, patient_class)
        permutation_vectors = [
            vector_for_distribution(seed_frame, value_columns, patient_class, index=original_vector.index)
            for _, seed_frame in permutations.groupby("seed")
        ]
        permutation_matrix = np.vstack([vector.to_numpy(dtype=float) for vector in permutation_vectors])
        expected = permutation_matrix.mean(axis=0)
        observed = original_vector.to_numpy(dtype=float)
        expected = expected / expected.sum() if expected.sum() > 0 else expected
        observed = observed / observed.sum() if observed.sum() > 0 else observed
        expected = np.clip(expected, 1e-12, None)
        expected = expected / expected.sum()
        observed = np.clip(observed, 0, None)
        observed = observed / observed.sum() if observed.sum() > 0 else observed
        null_js = np.asarray([safe_jensen_shannon(vector, expected) for vector in permutation_matrix])
        original_js = safe_jensen_shannon(observed, expected)
        chi = chisquare(f_obs=observed, f_exp=expected)
        g_test = power_divergence(f_obs=observed, f_exp=expected, lambda_="log-likelihood")
        rows.extend(
            [
                {
                    "variant": variant,
                    "distribution": distribution_name,
                    "patient_class": patient_class,
                    "test": "chi_square_goodness_of_fit_vs_mean_shuffle",
                    "statistic": float(chi.statistic),
                    "p_value": float(chi.pvalue),
                    "n_shuffle": int(len(permutation_vectors)),
                },
                {
                    "variant": variant,
                    "distribution": distribution_name,
                    "patient_class": patient_class,
                    "test": "g_test_goodness_of_fit_vs_mean_shuffle",
                    "statistic": float(g_test.statistic),
                    "p_value": float(g_test.pvalue),
                    "n_shuffle": int(len(permutation_vectors)),
                },
                {
                    "variant": variant,
                    "distribution": distribution_name,
                    "patient_class": patient_class,
                    "test": "jensen_shannon_empirical_distance",
                    "statistic": original_js,
                    "p_value": empirical_p_value(original_js, null_js, greater=True),
                    "n_shuffle": int(len(permutation_vectors)),
                },
            ]
        )
    return rows


def build_local_distribution_tests(
    variant_data: dict[str, dict[str, pd.DataFrame]],
    labels: list[str],
) -> pd.DataFrame:
    rows = []
    specs = [
        ("cell_type", ["cell_type"], "cell_original", "cell_permutations"),
        ("pairwise", ["source_cell_type", "target_cell_type"], "pairwise_original", "pairwise_permutations"),
    ]
    for variant in VARIANTS:
        for distribution, feature_columns, original_key, permutation_key in specs:
            original = variant_data[variant][original_key]
            permutations = variant_data[variant][permutation_key]
            for patient_class in labels:
                original_vector = vector_for_distribution(original, feature_columns, patient_class)
                seed_vectors = [
                    vector_for_distribution(seed_frame, feature_columns, patient_class, index=original_vector.index)
                    for _, seed_frame in permutations.groupby("seed")
                ]
                matrix = np.vstack([vector.to_numpy(dtype=float) for vector in seed_vectors])
                for feature_i, feature_id in enumerate(original_vector.index):
                    null_values = matrix[:, feature_i]
                    observed = float(original_vector.iloc[feature_i])
                    ci_low = float(np.percentile(null_values, 2.5))
                    ci_high = float(np.percentile(null_values, 97.5))
                    p_two_sided = empirical_two_sided_p_value(observed, null_values)
                    row = {
                        "variant": variant,
                        "distribution": distribution,
                        "patient_class": patient_class,
                        "feature": format_feature_id(feature_id),
                        "original": observed,
                        "permutation_mean": float(np.mean(null_values)),
                        "permutation_std": float(np.std(null_values, ddof=1)),
                        "permutation_ci_low": ci_low,
                        "permutation_ci_high": ci_high,
                        "outside_95_ci": bool(observed < ci_low or observed > ci_high),
                        "empirical_p_two_sided": p_two_sided,
                        "n_shuffle": int(len(null_values)),
                    }
                    rows.append(row)
    frame = pd.DataFrame(rows)
    fdr_frames = []
    for _, group in frame.groupby(["variant", "distribution", "patient_class"], sort=False):
        group = group.copy()
        group["fdr_bh_q_value"] = benjamini_hochberg(group["empirical_p_two_sided"].to_numpy())
        group["significant_fdr_0_05"] = group["fdr_bh_q_value"] <= 0.05
        fdr_frames.append(group)
    return pd.concat(fdr_frames, ignore_index=True)


def summarize_local_tests(local_tests: pd.DataFrame) -> pd.DataFrame:
    return (
        local_tests.groupby(["variant", "distribution", "patient_class"], as_index=False)
        .agg(
            n_features=("feature", "count"),
            n_outside_95_ci=("outside_95_ci", "sum"),
            n_fdr_0_05=("significant_fdr_0_05", "sum"),
            min_empirical_p=("empirical_p_two_sided", "min"),
            min_fdr_q=("fdr_bh_q_value", "min"),
        )
    )


def empirical_two_sided_p_value(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    lower = np.sum(null_values <= observed)
    upper = np.sum(null_values >= observed)
    return float(min(1.0, 2.0 * (min(lower, upper) + 1) / (len(null_values) + 1)))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running_min = min(running_min, ranked[i] * n / rank)
        adjusted_ranked[i] = running_min
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def format_feature_id(feature_id: Any) -> str:
    if isinstance(feature_id, tuple):
        return " -> ".join(str(part) for part in feature_id)
    return str(feature_id)


def vector_for_distribution(
    frame: pd.DataFrame,
    value_columns: list[str],
    patient_class: str,
    index: pd.Index | None = None,
) -> pd.Series:
    class_frame = frame.loc[frame["patient_class"] == patient_class]
    if len(value_columns) == 1:
        vector = class_frame.set_index(value_columns[0])["proportion"]
    else:
        vector = class_frame.set_index(value_columns)["proportion"]
    if index is not None:
        vector = vector.reindex(index, fill_value=0.0)
    return vector.sort_index()


def empirical_p_value(observed: float, null_values: np.ndarray, greater: bool) -> float:
    if greater:
        count = np.sum(null_values >= observed)
    else:
        count = np.sum(null_values <= observed)
    return float((count + 1) / (len(null_values) + 1))


def safe_jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.sum() <= 0 and right.sum() <= 0:
        return 0.0
    if left.sum() <= 0 or right.sum() <= 0:
        return 1.0
    return float(jensenshannon(left / left.sum(), right / right.sum(), base=2.0))


def load_statistical_tables(
    aggregate_dir: Path,
    trial_summaries: pd.DataFrame,
    stringency_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    auc_tests = pd.read_csv(aggregate_dir / "auc_statistical_tests.csv")
    aggregate_distribution_tests = pd.read_csv(aggregate_dir / "distribution_statistical_tests.csv")
    failure_summary = pd.read_csv(aggregate_dir / "permutation_failure_summary.csv")
    stringency_tests = pd.DataFrame(
        [
            {
                "metric": "optimal_stringency",
                "test": "empirical_permutation_greater_equal",
                "original": stringency_summary.iloc[0]["original"],
                "null_mean": stringency_summary.iloc[0]["permutation_mean"],
                "null_std": stringency_summary.iloc[0]["permutation_std"],
                "n_null": stringency_summary.iloc[0]["n_permutations"],
                "p_value": stringency_summary.iloc[0]["empirical_p_value_greater_equal"],
            }
        ]
    )
    return {
        "auc_tests": auc_tests,
        "aggregate_distribution_tests": aggregate_distribution_tests,
        "failure_summary": failure_summary,
        "stringency_tests": stringency_tests,
    }


def plot_cell_type_bars_by_variant(
    summary: pd.DataFrame,
    variant_data: dict[str, dict[str, pd.DataFrame]],
    labels: list[str],
    output_dir: Path,
) -> None:
    for variant, title in VARIANTS.items():
        variant_summary = summary.loc[summary["variant"] == variant]
        order = (
            variant_summary.groupby("cell_type")["mean"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        x = np.arange(len(order))
        width = 0.38
        for offset_i, patient_class in enumerate(labels):
            class_summary = variant_summary.loc[variant_summary["patient_class"] == patient_class].set_index("cell_type").reindex(order)
            means = class_summary["mean"].to_numpy() * 100
            lows = class_summary["ci_low"].to_numpy() * 100
            highs = class_summary["ci_high"].to_numpy() * 100
            positions = x + (offset_i - 0.5) * width
            color = CLASS_COLORS.get(patient_class, f"C{offset_i}")
            ax.bar(positions, means, width=width, color=color, alpha=0.82, label=f"{patient_class} permutation mean")
            ax.errorbar(
                positions,
                means,
                yerr=np.vstack([means - lows, highs - means]),
                fmt="none",
                ecolor="#1f2933",
                elinewidth=1.2,
                capsize=3,
            )
            original = variant_data[variant]["cell_original"]
            original_values = (
                original.loc[original["patient_class"] == patient_class]
                .set_index("cell_type")
                .reindex(order)["proportion"]
                .to_numpy()
                * 100
            )
            ax.scatter(positions, original_values, color=color, edgecolor="black", linewidth=0.4, s=42, marker="D", zorder=3, label=f"{patient_class} original")
        ax.set_title(f"{title}: motif-induced cell type distribution")
        ax.set_ylabel("Proportion (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.grid(axis="y", color="#d7dee8", linewidth=0.7, alpha=0.7)
        ax.legend(frameon=False, ncol=2)
        fig.savefig(output_dir / f"motif_induced_cell_type_distribution_bars_{variant}.png", dpi=300)
        plt.close(fig)


def plot_pairwise_heatmaps_by_variant(
    summary: pd.DataFrame,
    variant_data: dict[str, dict[str, pd.DataFrame]],
    labels: list[str],
    output_dir: Path,
) -> None:
    for variant, title in VARIANTS.items():
        for patient_class in labels:
            class_summary = summary.loc[
                (summary["variant"] == variant) & (summary["patient_class"] == patient_class)
            ].copy()
            original = variant_data[variant]["pairwise_original"]
            sources = sorted(class_summary["source_cell_type"].unique())
            targets = sorted(class_summary["target_cell_type"].unique())
            mean_matrix = pivot_pairwise(class_summary, "mean", sources, targets)
            low_matrix = pivot_pairwise(class_summary, "ci_low", sources, targets)
            high_matrix = pivot_pairwise(class_summary, "ci_high", sources, targets)
            original_matrix = pivot_pairwise(
                original.loc[original["patient_class"] == patient_class],
                "proportion",
                sources,
                targets,
            )
            fig, ax = plt.subplots(figsize=(15, 13), constrained_layout=True)
            image = ax.imshow(mean_matrix.to_numpy() * 100, cmap=CLASS_CMAPS.get(patient_class, "viridis"), aspect="auto")
            colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
            colorbar.set_label("Permutation mean proportion (%)")
            ax.set_title(f"{title}: {patient_class} motif-induced pairwise interactions")
            ax.set_xlabel("Target cell type")
            ax.set_ylabel("Source cell type")
            ax.set_xticks(np.arange(len(targets)))
            ax.set_yticks(np.arange(len(sources)))
            ax.set_xticklabels(targets, rotation=45, ha="right")
            ax.set_yticklabels(sources)
            max_value = np.nanmax(mean_matrix.to_numpy() * 100)
            for i, source in enumerate(sources):
                for j, target in enumerate(targets):
                    mean = mean_matrix.loc[source, target] * 100
                    low = low_matrix.loc[source, target] * 100
                    high = high_matrix.loc[source, target] * 100
                    orig = original_matrix.loc[source, target] * 100
                    text_color = "white" if max_value > 0 and mean > max_value * 0.55 else "black"
                    ax.text(
                        j,
                        i,
                        f"{mean:.1f}\n[{low:.1f},{high:.1f}]\nO:{orig:.1f}",
                        ha="center",
                        va="center",
                        fontsize=4.8,
                        color=text_color,
                    )
            ax.set_xticks(np.arange(-0.5, len(targets), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(sources), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.35)
            ax.tick_params(which="minor", bottom=False, left=False)
            fig.savefig(output_dir / f"motif_induced_pairwise_heatmap_{variant}_{patient_class}.png", dpi=300)
            plt.close(fig)


def pivot_pairwise(frame: pd.DataFrame, value: str, sources: list[str], targets: list[str]) -> pd.DataFrame:
    return frame.pivot(index="source_cell_type", columns="target_cell_type", values=value).reindex(index=sources, columns=targets)


def plot_shuffled_mean_vs_original_general(output_dir: Path, labels: list[str]) -> None:
    cell_path = output_dir / "shuffled_mean_cell_type_vs_original_general_distributions.csv"
    pairwise_path = output_dir / "shuffled_mean_pairwise_vs_original_general_distributions.csv"
    if cell_path.exists():
        plot_shuffled_mean_cell_type_vs_original_general(pd.read_csv(cell_path), labels, output_dir)
    if pairwise_path.exists():
        pairwise = pd.read_csv(pairwise_path)
        plot_shuffled_mean_pairwise_split_heatmaps(pairwise, labels, output_dir)
        plot_shuffled_mean_pairwise_difference_heatmaps(pairwise, labels, output_dir)


def plot_shuffled_mean_cell_type_vs_original_general(
    frame: pd.DataFrame,
    labels: list[str],
    output_dir: Path,
) -> None:
    frame = frame.loc[~frame["cell_type"].str.lower().eq("tumor")].copy()
    for variant, title in VARIANTS.items():
        variant_frame = frame.loc[frame["variant"] == variant]
        order = (
            variant_frame.groupby("cell_type")["shuffled_mean_cell_type_proportion"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        x = np.arange(len(order))
        width = 0.38
        for offset_i, patient_class in enumerate(labels):
            class_frame = (
                variant_frame.loc[variant_frame["patient_class"] == patient_class]
                .set_index("cell_type")
                .reindex(order)
            )
            positions = x + (offset_i - 0.5) * width
            color = CLASS_COLORS.get(patient_class, f"C{offset_i}")
            shuffled = class_frame["shuffled_mean_cell_type_proportion"].to_numpy() * 100.0
            original = class_frame["original_general_cell_proportion"].to_numpy() * 100.0
            ax.bar(
                positions,
                shuffled,
                width=width,
                color=color,
                alpha=0.82,
                label=f"{patient_class} shuffled motif-induced mean",
            )
            ax.scatter(
                positions,
                original,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                s=42,
                marker="D",
                zorder=3,
                label=f"{patient_class} original non-motif-induced",
            )
        ax.set_title(f"{title}: shuffled motif-induced mean vs original non-motif-induced cell type distribution")
        ax.set_ylabel("Proportion (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.grid(axis="y", color="#d7dee8", linewidth=0.7, alpha=0.7)
        ax.legend(frameon=False, ncol=2)
        fig.savefig(output_dir / f"shuffled_mean_vs_original_general_cell_type_distribution_{variant}.png", dpi=300)
        plt.close(fig)


def pairwise_cell_order(frame: pd.DataFrame) -> list[str]:
    order = []
    for cell_type in pd.concat([frame["source_cell_type"], frame["target_cell_type"]]).dropna():
        cell_type = str(cell_type)
        if cell_type.lower() == "tumor":
            continue
        if cell_type not in order:
            order.append(cell_type)
    return order


def plot_shuffled_mean_pairwise_split_heatmaps(
    frame: pd.DataFrame,
    labels: list[str],
    output_dir: Path,
) -> None:
    frame = frame.loc[
        ~frame["source_cell_type"].str.lower().eq("tumor")
        & ~frame["target_cell_type"].str.lower().eq("tumor")
    ].copy()
    order = pairwise_cell_order(frame)
    max_value = float(
        np.nanmax(
            frame[
                ["shuffled_mean_pairwise_proportion", "original_general_pairwise_proportion"]
            ].to_numpy(dtype=float)
        )
        * 100.0
    )
    if not math.isfinite(max_value) or max_value <= 0:
        max_value = 1.0
    for variant, title in VARIANTS.items():
        for patient_class in labels:
            sub = frame.loc[(frame["variant"] == variant) & (frame["patient_class"] == patient_class)]
            shuffled = pairwise_matrix(sub, "shuffled_mean_pairwise_proportion", order) * 100.0
            original = pairwise_matrix(sub, "original_general_pairwise_proportion", order) * 100.0
            cmap = plt.get_cmap(CLASS_CMAPS.get(patient_class, "viridis"))
            fig, ax = plt.subplots(figsize=(max(8.0, len(order) * 0.48), max(7.0, len(order) * 0.48)), constrained_layout=True)
            draw_upper_triangle_split_matrix(ax, shuffled, original, order, cmap, 0.0, max_value)
            ax.set_title(
                f"{title}: {patient_class} pairwise distribution\n"
                "upper cell half = shuffled motif-induced mean; lower cell half = original non-motif-induced",
                fontsize=11,
            )
            add_pairwise_axes(ax, order)
            sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=max_value), cmap=cmap)
            sm.set_array([])
            colorbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.025)
            colorbar.set_label("Pairwise proportion (%)", fontsize=9)
            colorbar.ax.tick_params(labelsize=8)
            fig.savefig(
                output_dir / f"shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_{variant}_{patient_class}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_shuffled_mean_pairwise_difference_heatmaps(
    frame: pd.DataFrame,
    labels: list[str],
    output_dir: Path,
) -> None:
    frame = frame.loc[
        ~frame["source_cell_type"].str.lower().eq("tumor")
        & ~frame["target_cell_type"].str.lower().eq("tumor")
    ].copy()
    order = pairwise_cell_order(frame)
    frame["difference_percentage_points"] = frame["difference_shuffled_minus_original_general"] * 100.0
    max_abs = float(np.nanmax(np.abs(frame["difference_percentage_points"].to_numpy(dtype=float))))
    if not math.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.get_cmap("RdBu_r")
    for variant, title in VARIANTS.items():
        for patient_class in labels:
            sub = frame.loc[(frame["variant"] == variant) & (frame["patient_class"] == patient_class)]
            matrix = pairwise_matrix(sub, "difference_percentage_points", order)
            fig, ax = plt.subplots(figsize=(max(8.0, len(order) * 0.48), max(7.0, len(order) * 0.48)), constrained_layout=True)
            draw_upper_triangle_matrix(ax, matrix, order, cmap, norm)
            ax.set_title(
                f"{title}: {patient_class} pairwise difference\n"
                "shuffled motif-induced mean - original non-motif-induced",
                fontsize=11,
            )
            add_pairwise_axes(ax, order)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            colorbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.025)
            colorbar.set_label("Difference (percentage points)", fontsize=9)
            colorbar.ax.tick_params(labelsize=8)
            fig.savefig(
                output_dir / f"shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_{variant}_{patient_class}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def pairwise_matrix(frame: pd.DataFrame, value_column: str, order: list[str]) -> pd.DataFrame:
    return (
        frame.pivot_table(
            index="source_cell_type",
            columns="target_cell_type",
            values=value_column,
            aggfunc="mean",
        )
        .reindex(index=order, columns=order)
        .fillna(0.0)
    )


def draw_upper_triangle_split_matrix(
    ax: plt.Axes,
    upper_matrix: pd.DataFrame,
    lower_matrix: pd.DataFrame,
    order: list[str],
    cmap: Any,
    vmin: float,
    vmax: float,
) -> None:
    normalizer = plt.Normalize(vmin=vmin, vmax=vmax)
    for i, source in enumerate(order):
        for j, target in enumerate(order):
            if j < i:
                continue
            upper_value = float(upper_matrix.loc[source, target])
            lower_value = float(lower_matrix.loc[source, target])
            ax.add_patch(
                Polygon(
                    [(j - 0.5, i - 0.5), (j + 0.5, i - 0.5), (j + 0.5, i + 0.5)],
                    closed=True,
                    facecolor=cmap(normalizer(upper_value)),
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
            ax.add_patch(
                Polygon(
                    [(j - 0.5, i - 0.5), (j - 0.5, i + 0.5), (j + 0.5, i + 0.5)],
                    closed=True,
                    facecolor=cmap(normalizer(lower_value)),
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
    draw_pairwise_grid(ax, len(order))


def draw_upper_triangle_matrix(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    order: list[str],
    cmap: Any,
    norm: Any,
) -> None:
    for i, source in enumerate(order):
        for j, target in enumerate(order):
            if j < i:
                continue
            value = float(matrix.loc[source, target])
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=cmap(norm(value)),
                    edgecolor="white",
                    linewidth=0.45,
                )
            )
    draw_pairwise_grid(ax, len(order))


def draw_pairwise_grid(ax: plt.Axes, size: int) -> None:
    ax.set_facecolor("white")
    for position in np.arange(-0.5, size, 1):
        ax.axvline(position, color="#eeeeee", linewidth=0.35, zorder=0)
        ax.axhline(position, color="#eeeeee", linewidth=0.35, zorder=0)
    ax.add_patch(Rectangle((-0.5, -0.5), size, size, fill=False, edgecolor="#444444", linewidth=0.8))


def add_pairwise_axes(ax: plt.Axes, order: list[str]) -> None:
    size = len(order)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Target cell type")
    ax.set_ylabel("Source cell type")
    ax.tick_params(length=0)


def plot_permutation_metric_scatterplots(trial_summaries: pd.DataFrame, output_dir: Path) -> None:
    metrics = build_permutation_scatter_metrics(trial_summaries, output_dir)
    if metrics.empty:
        return
    metrics.to_csv(output_dir / "permutation_metric_scatterplot_data.csv", index=False)
    plot_stringency_auc_colored_by_motifs(metrics, output_dir)
    plot_stringency_auc_colored_by_distance(metrics, output_dir)
    plot_motif_metric_regression(metrics, "optimal_stringency", "Optimal stringency", output_dir)
    plot_motif_metric_regression(metrics, "optimal_auc", "Optimal AUC", output_dir)
    plot_motif_metric_regression(
        metrics,
        "mean_optimal_js_distance_from_baseline",
        "Mean JS distance from non-motif-induced distribution",
        output_dir,
    )


def build_permutation_scatter_metrics(trial_summaries: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    required_columns = ["seed", "shuffle_labels", "optimal_stringency", "optimal_auc", "n_selected_unique_motifs"]
    missing = [column for column in required_columns if column not in trial_summaries.columns]
    if missing:
        raise ValueError(f"Missing required trial summary columns: {missing}")
    metrics = trial_summaries.loc[:, required_columns].copy()
    metrics["shuffle_labels"] = metrics["shuffle_labels"].astype(bool)
    metrics = metrics.rename(columns={"n_selected_unique_motifs": "n_fold_union_discriminative_motifs"})
    distance_path = output_dir / "permutation_distances_to_baseline_and_original.csv"
    metrics["mean_optimal_js_distance_from_original"] = np.nan
    metrics["mean_optimal_js_distance_from_baseline"] = np.nan
    if distance_path.exists():
        distances = pd.read_csv(distance_path)
        optimal_distances = distances.loc[distances["variant"] == "optimal"].copy()
        mean_distances = (
            optimal_distances.groupby("seed", as_index=False)[["js_vs_original", "js_vs_baseline"]]
            .mean()
            .rename(
                columns={
                    "js_vs_original": "mean_optimal_js_distance_from_original",
                    "js_vs_baseline": "mean_optimal_js_distance_from_baseline",
                }
            )
        )
        metrics = metrics.drop(
            columns=["mean_optimal_js_distance_from_original", "mean_optimal_js_distance_from_baseline"]
        ).merge(mean_distances, on="seed", how="left")
    metrics.loc[~metrics["shuffle_labels"], "mean_optimal_js_distance_from_original"] = 0.0
    original_seed = metrics.loc[~metrics["shuffle_labels"], "seed"].iloc[0]
    if distance_path.exists():
        original_baseline = (
            pd.read_csv(distance_path)
            .loc[lambda frame: (frame["variant"] == "optimal") & (frame["seed"] == original_seed), "js_vs_baseline"]
            .mean()
        )
        metrics.loc[~metrics["shuffle_labels"], "mean_optimal_js_distance_from_baseline"] = original_baseline
    return metrics


def split_original_and_permutations(metrics: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    original = metrics.loc[~metrics["shuffle_labels"]].iloc[0]
    permutations = metrics.loc[metrics["shuffle_labels"]].copy()
    return original, permutations


def plot_stringency_auc_colored_by_motifs(metrics: pd.DataFrame, output_dir: Path) -> None:
    original, permutations = split_original_and_permutations(metrics)
    fig, ax = plt.subplots(figsize=(11.5, 8.2), constrained_layout=True)
    scatter = ax.scatter(
        permutations["optimal_stringency"],
        permutations["optimal_auc"],
        c=permutations["n_fold_union_discriminative_motifs"],
        cmap="viridis",
        s=92,
        edgecolor="#394455",
        linewidth=0.8,
        alpha=0.78,
        label="Shuffled permutations",
    )
    ax.scatter(
        [original["optimal_stringency"]],
        [original["optimal_auc"]],
        c=[original["n_fold_union_discriminative_motifs"]],
        cmap="viridis",
        vmin=float(permutations["n_fold_union_discriminative_motifs"].min()),
        vmax=float(permutations["n_fold_union_discriminative_motifs"].max()),
        s=155,
        marker="D",
        edgecolor="black",
        linewidth=1.0,
        label="Original",
        zorder=5,
    )
    ax.scatter(
        [original["optimal_stringency"]],
        [original["optimal_auc"]],
        s=310,
        marker="s",
        facecolors="none",
        edgecolors="black",
        linewidth=1.8,
        zorder=6,
        label="Original boxed",
    )
    add_stringency_auc_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label("# fold-union discriminative motifs")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_dir / "optimal_stringency_vs_auc_colored_by_motif_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stringency_auc_colored_by_distance(metrics: pd.DataFrame, output_dir: Path) -> None:
    original, permutations = split_original_and_permutations(metrics)
    fig, ax = plt.subplots(figsize=(11.5, 8.2), constrained_layout=True)
    scatter = ax.scatter(
        permutations["optimal_stringency"],
        permutations["optimal_auc"],
        c=permutations["mean_optimal_js_distance_from_original"],
        cmap="magma",
        s=92,
        edgecolor="#394455",
        linewidth=0.8,
        alpha=0.78,
        label="Shuffled permutations",
    )
    ax.scatter(
        [original["optimal_stringency"]],
        [original["optimal_auc"]],
        c=[0.0],
        cmap="magma",
        vmin=0.0,
        vmax=float(permutations["mean_optimal_js_distance_from_original"].max()),
        s=155,
        marker="D",
        edgecolor="black",
        linewidth=1.0,
        label="Original",
        zorder=5,
    )
    ax.scatter(
        [original["optimal_stringency"]],
        [original["optimal_auc"]],
        s=310,
        marker="s",
        facecolors="none",
        edgecolors="black",
        linewidth=1.8,
        zorder=6,
        label="Original boxed",
    )
    add_stringency_auc_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label("Mean optimal JS distance from original")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_dir / "optimal_stringency_vs_auc_colored_by_original_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_stringency_auc_axes(ax: plt.Axes) -> None:
    ax.set_title("Optimal stringency vs AUC", fontsize=22, pad=12)
    ax.set_xlabel("Optimal stringency", fontsize=16)
    ax.set_ylabel("Optimal AUC", fontsize=16)
    ax.grid(True, color="#d7dee8", linewidth=1.0, alpha=0.8)
    ax.tick_params(axis="both", labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_motif_metric_regression(
    metrics: pd.DataFrame,
    y_column: str,
    y_label: str,
    output_dir: Path,
) -> None:
    original, permutations = split_original_and_permutations(metrics)
    plot_frame = permutations.loc[
        permutations["n_fold_union_discriminative_motifs"].notna() & permutations[y_column].notna()
    ].copy()
    x = plot_frame["n_fold_union_discriminative_motifs"].to_numpy(dtype=float)
    y = plot_frame[y_column].to_numpy(dtype=float)
    regression = linregress(x, y) if len(plot_frame) >= 2 else None
    fig, ax = plt.subplots(figsize=(10.6, 7.5), constrained_layout=True)
    ax.scatter(x, y, s=82, color="#5b84bd", edgecolor="#476f9f", linewidth=0.9, alpha=0.72, label="Shuffled permutations")
    if regression is not None:
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = regression.intercept + regression.slope * x_line
        ax.plot(
            x_line,
            y_line,
            color="#1f2933",
            linewidth=2.2,
            label=f"Linear fit (r={regression.rvalue:.3f}, p={regression.pvalue:.3g})",
        )
    ax.scatter(
        [original["n_fold_union_discriminative_motifs"]],
        [original[y_column]],
        s=150,
        marker="D",
        color="#c0392b",
        edgecolor="black",
        linewidth=1.0,
        label="Original",
        zorder=5,
    )
    ax.scatter(
        [original["n_fold_union_discriminative_motifs"]],
        [original[y_column]],
        s=300,
        marker="s",
        facecolors="none",
        edgecolors="black",
        linewidth=1.8,
        zorder=6,
    )
    ax.set_title(f"# motifs vs {y_label}", fontsize=18, pad=10)
    ax.set_xlabel("# fold-union discriminative motifs", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, color="#d7dee8", linewidth=0.9, alpha=0.75)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(axis="both", labelsize=12)
    output_name = {
        "optimal_stringency": "motif_count_vs_optimal_stringency_regression.png",
        "optimal_auc": "motif_count_vs_optimal_auc_regression.png",
        "mean_optimal_js_distance_from_baseline": "motif_count_vs_baseline_distance_regression.png",
    }[y_column]
    fig.savefig(output_dir / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    auc_summary: pd.DataFrame,
    stringency_summary: pd.DataFrame,
    stats_tables: dict[str, pd.DataFrame],
    n_permutations: int,
) -> None:
    optimal = auc_summary.loc[auc_summary["metric"] == "optimal_auc"].iloc[0]
    fixed = auc_summary.loc[auc_summary["metric"] == "fixed_auc"].iloc[0]
    stringency = stringency_summary.iloc[0]
    report = f"""# CISM label permutation report

This report summarizes {n_permutations} successful shuffled-label CISM permutations against the original tutorial artifact result.

## AUC and stringency

![AUC permutation histograms](auc_permutation_histograms.png)

![Optimal stringency distribution](optimal_stringency_distribution.png)

![Optimal stringency vs AUC colored by motif count](optimal_stringency_vs_auc_colored_by_motif_count.png)

![Optimal stringency vs AUC colored by distance from original](optimal_stringency_vs_auc_colored_by_original_distance.png)

![Motif count vs optimal stringency](motif_count_vs_optimal_stringency_regression.png)

![Motif count vs optimal AUC](motif_count_vs_optimal_auc_regression.png)

![Motif count vs distance from non-motif-induced distribution](motif_count_vs_baseline_distance_regression.png)

| Metric | Original | Permutation mean | Permutation SD | Min | Max |
|---|---:|---:|---:|---:|---:|
| Optimal stringency AUC | {optimal['original']:.3f} | {optimal['permutation_mean']:.3f} | {optimal['permutation_std']:.3f} | {optimal['permutation_min']:.3f} | {optimal['permutation_max']:.3f} |
| Fixed 0.46 AUC | {fixed['original']:.3f} | {fixed['permutation_mean']:.3f} | {fixed['permutation_std']:.3f} | {fixed['permutation_min']:.3f} | {fixed['permutation_max']:.3f} |
| Optimal stringency | {stringency['original']:.3f} | {stringency['permutation_mean']:.3f} | {stringency['permutation_std']:.3f} | {stringency['permutation_min']:.3f} | {stringency['permutation_max']:.3f} |

## Motif-induced cell type distributions

Bars show permutation means. Error bars show empirical 95% permutation intervals. Diamond markers show original values.

![Optimal motif-induced cell type distribution bars](motif_induced_cell_type_distribution_bars_optimal.png)

![Fixed motif-induced cell type distribution bars](motif_induced_cell_type_distribution_bars_fixed.png)

## Motif-induced pairwise distributions

Each heatmap cell shows permutation mean percentage, empirical 95% interval in brackets, and original value prefixed by `O:`. NN uses a blue color scheme; NP uses a red color scheme.

![Optimal NN motif-induced pairwise heatmap](motif_induced_pairwise_heatmap_optimal_NN.png)

![Optimal NP motif-induced pairwise heatmap](motif_induced_pairwise_heatmap_optimal_NP.png)

![Fixed NN motif-induced pairwise heatmap](motif_induced_pairwise_heatmap_fixed_NN.png)

![Fixed NP motif-induced pairwise heatmap](motif_induced_pairwise_heatmap_fixed_NP.png)

## Direct baseline tests

Direct tests compare motif-induced distributions against the same-class general baseline using unweighted motif node counts for cell type distributions and unweighted motif edge counts for pairwise distributions.

![Permutation chi-square p-value histograms](permutation_chi_square_p_value_histograms.png)

![Permutation G-test p-value histograms](permutation_g_test_p_value_histograms.png)

## Shuffled mean vs original non-motif-induced baseline

These figures compare the shuffled motif-induced mean distribution against the original non-motif-induced, same-class general distribution. Cell type plots show shuffled motif-induced means as bars and original non-motif-induced values as diamond markers. Pairwise split heatmaps show the shuffled motif-induced mean in the upper half of each colored upper-triangle cell and the original non-motif-induced value in the lower half. Difference heatmaps show shuffled motif-induced mean minus original non-motif-induced in percentage points.

![Optimal cell type shuffled mean vs original general](shuffled_mean_vs_original_general_cell_type_distribution_optimal.png)

![Fixed cell type shuffled mean vs original general](shuffled_mean_vs_original_general_cell_type_distribution_fixed.png)

![Optimal NN pairwise shuffled mean vs original general](shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_optimal_NN.png)

![Optimal NP pairwise shuffled mean vs original general](shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_optimal_NP.png)

![Fixed NN pairwise shuffled mean vs original general](shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_fixed_NN.png)

![Fixed NP pairwise shuffled mean vs original general](shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_fixed_NP.png)

![Optimal NN pairwise shuffled mean minus original general](shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_optimal_NN.png)

![Optimal NP pairwise shuffled mean minus original general](shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_optimal_NP.png)

![Fixed NN pairwise shuffled mean minus original general](shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_fixed_NN.png)

![Fixed NP pairwise shuffled mean minus original general](shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_fixed_NP.png)

## Optimized distribution distance comparison

Base-2 Jensen-Shannon distances use the original, same-class non-motif-induced distribution as the fixed baseline. Green lines show the distance of the pooled shuffled motif-induced composition; dashed purple lines show the mean of individual shuffled distances; red lines show the unpermuted motif-induced distance. The histograms show distances from individual shuffles. Empirical permutation-rank p-values ask whether the unpermuted distance is unusually far from the baseline relative to shuffled trials; all four two-sided comparisons are nonsignificant. They do not establish equivalence to the baseline, which requires a prespecified distance margin. The pooled-composition distance is descriptive: its pooled counts do not equal the arithmetic mean of the saved per-seed vectors, so no bootstrap p-value is attached to it.

![Optimized motif-induced distance comparison](optimal_motif_vs_shuffled_mean_baseline_distance_tests.png)

![Distance of pooled shuffled composition versus mean of shuffled distances](optimal_distance_of_mean_vs_mean_distance.png)

Full values and methods: `optimal_motif_vs_shuffled_mean_baseline_distance_tests.csv` and `optimal_motif_vs_shuffled_mean_baseline_distance_tests.md`.

## Optimized compositions versus tissue baselines

The pooled shuffled-label composition shown in the comparison plots and the unpermuted motif-induced composition are each tested against the original, same-class non-motif-induced tissue composition. Monte Carlo multinomial goodness-of-fit tests use base-2 Jensen-Shannon distance and 100,000 null draws at the observed count total. These test exact equality under an independent-count model; motif nodes and edges are not independent biological replicates, and the pooled shuffled counts reuse the same tissue 100 times. Therefore these p-values are conditional model checks, not patient-level inference or evidence of equivalence. The table also reports the fraction of individual shuffled trials farther from the tissue baseline than the unpermuted motif-induced result and its empirical upper-tail permutation p-value.

Full results and methods: `optimized_pooled_and_unpermuted_vs_general_tests.csv` and `optimized_pooled_and_unpermuted_vs_general_tests.md`.

## Fold-union discriminative motif counts

These figures summarize the number of unique discriminative motifs appearing in the union of all LOOCV fold feature lists for each trial.

![Fold-union motif count histograms](fold_union_discriminative_motif_count_histograms.png)

![Fold-union motif count boxplot](fold_union_discriminative_motif_count_boxplot.png)

## Statistical tables

- `auc_statistical_tests.csv` from the aggregate directory
- `distribution_statistical_tests.csv` from the aggregate directory
- `variant_motif_induced_distribution_tests.csv`
- `local_motif_induced_distribution_tests.csv`
- `original_direct_motif_induced_vs_general_distribution_tests.csv`
- `direct_motif_induced_vs_general_distribution_tests.csv`
- `shuffled_mean_cell_type_vs_original_general_tests.csv`
- `shuffled_mean_cell_type_vs_original_general_distributions.csv`
- `shuffled_mean_pairwise_vs_original_general_distributions.csv`
- `optimal_motif_vs_shuffled_mean_baseline_distance_tests.csv`
- `optimized_pooled_and_unpermuted_vs_general_tests.csv`
- `permutation_metric_scatterplot_data.csv`
- `fold_union_discriminative_motif_counts.csv`
- `fold_union_discriminative_motif_counts_summary.csv`
- `permutation_failure_summary.csv` from the aggregate directory
"""
    (output_dir / "README.md").write_text(report)
    stats_tables["auc_tests"].to_csv(output_dir / "auc_statistical_tests.csv", index=False)
    stats_tables["aggregate_distribution_tests"].to_csv(output_dir / "aggregate_distribution_statistical_tests.csv", index=False)
    stats_tables["failure_summary"].to_csv(output_dir / "permutation_failure_summary.csv", index=False)
    stats_tables["stringency_tests"].to_csv(output_dir / "optimal_stringency_statistical_tests.csv", index=False)


def write_pdf_report(
    output_dir: Path,
    auc_summary: pd.DataFrame,
    stringency_summary: pd.DataFrame,
    stats_tables: dict[str, pd.DataFrame],
    n_permutations: int,
) -> None:
    pdf_path = output_dir / "cism_label_permutation_report.pdf"
    image_pages = [
        ("AUC permutation histograms", output_dir / "auc_permutation_histograms.png"),
        ("Optimal stringency", output_dir / "optimal_stringency_distribution.png"),
        ("Optimal stringency vs AUC colored by motif count", output_dir / "optimal_stringency_vs_auc_colored_by_motif_count.png"),
        ("Optimal stringency vs AUC colored by distance from original", output_dir / "optimal_stringency_vs_auc_colored_by_original_distance.png"),
        ("Motif count vs optimal stringency", output_dir / "motif_count_vs_optimal_stringency_regression.png"),
        ("Motif count vs optimal AUC", output_dir / "motif_count_vs_optimal_auc_regression.png"),
        ("Motif count vs distance from non-motif-induced distribution", output_dir / "motif_count_vs_baseline_distance_regression.png"),
        ("Optimal motif-induced cell type distributions", output_dir / "motif_induced_cell_type_distribution_bars_optimal.png"),
        ("Fixed motif-induced cell type distributions", output_dir / "motif_induced_cell_type_distribution_bars_fixed.png"),
        ("Optimal NN motif-induced pairwise interactions", output_dir / "motif_induced_pairwise_heatmap_optimal_NN.png"),
        ("Optimal NP motif-induced pairwise interactions", output_dir / "motif_induced_pairwise_heatmap_optimal_NP.png"),
        ("Fixed NN motif-induced pairwise interactions", output_dir / "motif_induced_pairwise_heatmap_fixed_NN.png"),
        ("Fixed NP motif-induced pairwise interactions", output_dir / "motif_induced_pairwise_heatmap_fixed_NP.png"),
        ("Direct chi-square p-values", output_dir / "permutation_chi_square_p_value_histograms.png"),
        ("Direct G-test p-values", output_dir / "permutation_g_test_p_value_histograms.png"),
        ("Optimal cell type: shuffled mean vs original general", output_dir / "shuffled_mean_vs_original_general_cell_type_distribution_optimal.png"),
        ("Fixed cell type: shuffled mean vs original general", output_dir / "shuffled_mean_vs_original_general_cell_type_distribution_fixed.png"),
        ("Optimal NN pairwise: shuffled mean vs original general", output_dir / "shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_optimal_NN.png"),
        ("Optimal NP pairwise: shuffled mean vs original general", output_dir / "shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_optimal_NP.png"),
        ("Fixed NN pairwise: shuffled mean vs original general", output_dir / "shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_fixed_NN.png"),
        ("Fixed NP pairwise: shuffled mean vs original general", output_dir / "shuffled_mean_vs_original_general_pairwise_upper_triangle_split_heatmap_fixed_NP.png"),
        ("Optimal NN pairwise: shuffled mean minus original general", output_dir / "shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_optimal_NN.png"),
        ("Optimal NP pairwise: shuffled mean minus original general", output_dir / "shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_optimal_NP.png"),
        ("Fixed NN pairwise: shuffled mean minus original general", output_dir / "shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_fixed_NN.png"),
        ("Fixed NP pairwise: shuffled mean minus original general", output_dir / "shuffled_mean_minus_original_general_pairwise_upper_triangle_heatmap_fixed_NP.png"),
        ("Optimized motif-induced distances to original general baseline", output_dir / "optimal_motif_vs_shuffled_mean_baseline_distance_tests.png"),
        ("Distance of pooled composition versus average distance", output_dir / "optimal_distance_of_mean_vs_mean_distance.png"),
        ("Fold-union discriminative motif count histograms", output_dir / "fold_union_discriminative_motif_count_histograms.png"),
        ("Fold-union discriminative motif count boxplot", output_dir / "fold_union_discriminative_motif_count_boxplot.png"),
    ]
    with PdfPages(pdf_path) as pdf:
        add_title_page(pdf, auc_summary, stringency_summary, n_permutations)
        for title, image_path in image_pages:
            if image_path.exists():
                add_image_page(pdf, title, image_path)
        add_table_pages(pdf, output_dir, stats_tables)


def add_title_page(
    pdf: PdfPages,
    auc_summary: pd.DataFrame,
    stringency_summary: pd.DataFrame,
    n_permutations: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    optimal = auc_summary.loc[auc_summary["metric"] == "optimal_auc"].iloc[0]
    fixed = auc_summary.loc[auc_summary["metric"] == "fixed_auc"].iloc[0]
    stringency = stringency_summary.iloc[0]
    lines = [
        f"Summary of {n_permutations} successful shuffled-label CISM permutations",
        "compared with the original tutorial artifact result.",
        "",
        "AUC and stringency summary",
        f"Optimal AUC: original {optimal['original']:.3f}, permutation mean {optimal['permutation_mean']:.3f} (SD {optimal['permutation_std']:.3f})",
        f"Fixed 0.46 AUC: original {fixed['original']:.3f}, permutation mean {fixed['permutation_mean']:.3f} (SD {fixed['permutation_std']:.3f})",
        f"Optimal stringency: original {stringency['original']:.3f}, permutation mean {stringency['permutation_mean']:.3f} (SD {stringency['permutation_std']:.3f})",
        "",
        "Variant distribution plots are motif-aware.",
        "Unique selected motifs are assigned to NN/NP by class prevalence and weighted by motif mean frequency.",
        "Direct baseline tests use unweighted motif node/edge counts against same-class general baselines.",
    ]
    ax.text(0.08, 0.92, "CISM Label Permutation Report", fontsize=24, weight="bold", va="top")
    ax.text(0.08, 0.84, "\n".join(lines), fontsize=12, va="top", linespacing=1.55)
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    image = plt.imread(image_path)
    height, width = image.shape[:2]
    landscape = width >= height
    fig_size = (11.69, 8.27) if landscape else (8.27, 11.69)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, y=0.965)
    pdf.savefig(fig)
    plt.close(fig)


def add_table_pages(pdf: PdfPages, output_dir: Path, stats_tables: dict[str, pd.DataFrame]) -> None:
    variant_tests = pd.read_csv(output_dir / "variant_motif_induced_distribution_tests.csv")
    local_tests = pd.read_csv(output_dir / "local_motif_induced_distribution_tests.csv")
    local_top = local_tests.sort_values(
        ["significant_fdr_0_05", "outside_95_ci", "empirical_p_two_sided"],
        ascending=[False, False, True],
    ).head(60)
    add_text_table_page(pdf, "AUC and stringency statistical tests", pd.concat([stats_tables["auc_tests"], stats_tables["stringency_tests"]], ignore_index=True))
    add_text_table_page(pdf, "Aggregate distribution statistical tests", stats_tables["aggregate_distribution_tests"])
    add_text_table_page(pdf, "Fixed/optimal motif-induced distribution tests", variant_tests)
    add_text_table_page(pdf, "Top local motif-induced deviations", local_top)
    optional_tables = [
        ("Original direct motif-induced vs general tests", output_dir / "original_direct_motif_induced_vs_general_distribution_tests.csv"),
        ("Permutation direct motif-induced vs general tests", output_dir / "direct_motif_induced_vs_general_distribution_tests.csv"),
        ("Shuffled mean cell type vs original general tests", output_dir / "shuffled_mean_cell_type_vs_original_general_tests.csv"),
        ("Shuffled mean cell type vs original general distributions", output_dir / "shuffled_mean_cell_type_vs_original_general_distributions.csv"),
        ("Shuffled mean pairwise vs original general distributions", output_dir / "shuffled_mean_pairwise_vs_original_general_distributions.csv"),
        ("Optimized motif-induced distance tests", output_dir / "optimal_motif_vs_shuffled_mean_baseline_distance_tests.csv"),
        ("Optimized compositions vs tissue baseline (conditional GOF)", output_dir / "optimized_pooled_and_unpermuted_vs_general_tests.csv"),
        ("Permutation metric scatterplot data", output_dir / "permutation_metric_scatterplot_data.csv"),
        ("Fold-union discriminative motif count summary", output_dir / "fold_union_discriminative_motif_counts_summary.csv"),
        ("Fold-union discriminative motif counts", output_dir / "fold_union_discriminative_motif_counts.csv"),
    ]
    for title, path in optional_tables:
        if path.exists():
            frame = pd.read_csv(path)
            if title == "Permutation direct motif-induced vs general tests":
                frame = summarize_direct_test_permutations(frame)
            elif title == "Optimized motif-induced distance tests":
                frame = frame[["distribution", "patient_class", "shuffled_mean_js_to_original_general",
                               "mean_individual_shuffle_js",
                               "original_motif_js_to_original_general", "median_individual_shuffle_js",
                               "shuffles_at_least_as_distant_as_original",
                               "empirical_p_two_sided", "holm_p_two_sided_across_four_tests"]]
            elif title == "Optimized compositions vs tissue baseline (conditional GOF)":
                frame = frame[["distribution", "patient_class", "pooled_shuffle_js_to_general",
                               "pooled_shuffle_gof_mc_p", "unpermuted_js_to_general",
                               "unpermuted_gof_mc_p", "fraction_shuffles_farther_than_unpermuted",
                               "empirical_p_unpermuted_more_distant"]]
                frame.columns = ["Distribution", "Class", "Pooled JS", "Pooled GOF p",
                                 "Original JS", "Original GOF p", "Farther fraction", "Rank p"]
            elif title == "Fold-union discriminative motif counts":
                frame = frame.head(80)
            add_text_table_page(pdf, title, frame)
    add_text_table_page(pdf, "Permutation no-feature/failure summary", stats_tables["failure_summary"])


def summarize_direct_test_permutations(frame: pd.DataFrame) -> pd.DataFrame:
    permutations = frame.loc[frame["run_type"] == "permutation"].copy()
    if permutations.empty:
        return permutations
    return (
        permutations.groupby(["variant", "distribution", "patient_class"], as_index=False)
        .agg(
            n_permutations=("seed", "count"),
            n_chi_square_p_lt_0_05=("chi_square_p_value", lambda values: int((values < 0.05).sum())),
            n_g_test_p_lt_0_05=("g_test_p_value", lambda values: int((values < 0.05).sum())),
            chi_square_p_median=("chi_square_p_value", "median"),
            g_test_p_median=("g_test_p_value", "median"),
        )
    )


def add_text_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    rows = dataframe_to_lines(frame)
    chunks = [rows[i : i + 36] for i in range(0, len(rows), 36)] or [[]]
    for page_i, chunk in enumerate(chunks, start=1):
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        page_title = title if len(chunks) == 1 else f"{title} ({page_i}/{len(chunks)})"
        ax.text(0.04, 0.95, page_title, fontsize=15, weight="bold", va="top")
        ax.text(0.04, 0.9, "\n".join(chunk), family="monospace", fontsize=6.8, va="top")
        pdf.savefig(fig)
        plt.close(fig)


def dataframe_to_lines(frame: pd.DataFrame) -> list[str]:
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_float_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: f"{value:.4g}")
    return printable.to_string(index=False).splitlines()


if __name__ == "__main__":
    main()
