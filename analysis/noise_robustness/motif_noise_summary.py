#!/usr/bin/env python3
"""Summarize motif-induced distributions and noise sensitivity across CISM runs."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path("/groups/assafza_group/assafza/Omri/golan_output")
DEFAULT_OUTPUT_DIR = Path("motif_noise_summary_outputs")
DEFAULT_PATIENT_CLASS = DEFAULT_INPUT_ROOT / "Melanoma" / "patient_class.csv"

CELL_ORDER = [
    "B cell",
    "CD4 APC",
    "CD4 T cell",
    "CD4 Treg",
    "CD8 T cell",
    "DCs",
    "Germinal Center cells",
    "HEVs",
    "MAC",
    "Memory CD4 T cell",
    "NK cell",
    "Neutrophil",
    "Stroma",
    "Unidentified",
    "Vessels",
]

CELL_ALIASES = {
    "b cell": "B cell",
    "cd4 apc": "CD4 APC",
    "cd4 t cell": "CD4 T cell",
    "cd4 treg": "CD4 Treg",
    "treg": "CD4 Treg",
    "cd8 t cell": "CD8 T cell",
    "dc": "DCs",
    "dcs": "DCs",
    "gc": "Germinal Center cells",
    "germinal center cell": "Germinal Center cells",
    "germinal center cells": "Germinal Center cells",
    "hev": "HEVs",
    "hevs": "HEVs",
    "mac": "MAC",
    "memory cd4 t cell": "Memory CD4 T cell",
    "memory%cd4 t cell": "Memory CD4 T cell",
    "nk cell": "NK cell",
    "neutrophil": "Neutrophil",
    "stroma": "Stroma",
    "unidentified": "Unidentified",
    "vessels": "Vessels",
    "blood vessels": "Vessels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average motif-induced cell/pairwise distributions by noise condition "
            "and estimate how noise affects optimal stringency and AUC."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient-class", type=Path, default=DEFAULT_PATIENT_CLASS)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    parser.add_argument(
        "--panel-conditions",
        default="",
        help=(
            "Comma-separated condition names to show in replicated cell/heatmap panels. "
            "Default: raw plus the highest non-matrix and highest matrix noise conditions that have patient counts."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default="0.2,0.3,0.4,0.42,0.44,0.46,0.48,0.5,0.6,0.7,0.72,0.74,0.76,0.8",
        help="Comma-separated shared_percentage/stringency thresholds.",
    )
    parser.add_argument("--max-class-features", type=int, default=30)
    return parser.parse_args()


def condition_name_from_output_dir(output_dir: Path) -> Tuple[str, float]:
    name = output_dir.name
    if name == "output":
        return "raw", 0.0
    match = re.search(r"(\d+(?:\.\d+)?)", name)
    if not match:
        return name, math.nan
    noise_rate = float(match.group(1))
    suffix = "_matrix" if "matrix" in name.lower() else ""
    return f"noise_{noise_rate:g}{suffix}", noise_rate


def condition_paths(input_root: Path) -> List[Tuple[str, float, Path]]:
    discovered = []
    seen = set()
    for motifs_dir in sorted(input_root.glob("output*/Melanoma/Motifs")):
        files = motif_files(motifs_dir)
        if not files:
            continue
        output_dir = motifs_dir.parents[1]
        condition, noise_rate = condition_name_from_output_dir(output_dir)
        key = (condition, str(motifs_dir))
        if key in seen:
            continue
        seen.add(key)
        discovered.append((condition, noise_rate, motifs_dir))
    return sorted(
        discovered,
        key=lambda item: (
            math.inf if math.isnan(item[1]) else item[1],
            "matrix" in item[0],
            item[0],
        ),
    )


def select_panel_conditions(summary: pd.DataFrame, requested: str) -> Optional[List[str]]:
    if summary.empty:
        return None
    available = summary[["condition", "noise_rate"]].drop_duplicates()
    if requested.strip():
        requested_names = [item.strip() for item in requested.split(",") if item.strip()]
        return [name for name in requested_names if name in set(available["condition"])]

    selected: List[str] = []
    if "raw" in set(available["condition"]):
        selected.append("raw")
    non_raw = available[available["condition"] != "raw"].copy()
    non_matrix = non_raw[~non_raw["condition"].str.contains("matrix", case=False, na=False)]
    matrix = non_raw[non_raw["condition"].str.contains("matrix", case=False, na=False)]
    for block in [non_matrix, matrix]:
        if block.empty:
            continue
        row = block.sort_values(["noise_rate", "condition"], ascending=[False, True]).iloc[0]
        condition = str(row["condition"])
        if condition not in selected:
            selected.append(condition)
    return selected


def parse_thresholds(value: str) -> List[float]:
    thresholds = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def read_patient_classes(path: Path, labels: List[str]) -> Dict[str, str]:
    df = pd.read_csv(path, names=["patient", "class"], dtype=str)
    df = df[df["class"].isin(labels)].copy()
    return dict(zip(df["patient"], df["class"]))


def motif_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    files = []
    for csv_path in path.glob("discriminative_motifs*.csv"):
        match = re.fullmatch(r"discriminative_motifs(\d+)\.csv", csv_path.name)
        if match:
            files.append(csv_path)
    return sorted(files, key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))


def run_id_from_path(path: Path) -> int:
    match = re.search(r"discriminative_motifs(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Cannot parse run id from {path}")
    return int(match.group(1))


def normalize_cell(cell: str) -> str:
    cell = re.sub(r"#\d+$", "", str(cell)).strip().lower()
    cell = re.sub(r"\s+", " ", cell)
    return CELL_ALIASES.get(cell, cell.title())


def parse_nodes(nodes_text: object) -> List[str]:
    if pd.isna(nodes_text):
        return []
    cells = []
    for part in str(nodes_text).split(","):
        part = part.strip()
        if part:
            cells.append(normalize_cell(part))
    return cells


def parse_edges(edges_text: object) -> List[Tuple[str, str]]:
    if pd.isna(edges_text):
        return []
    pairs = []
    for edge in str(edges_text).split(";"):
        if " - " not in edge:
            continue
        left, right = edge.split(" - ", 1)
        left_cell = normalize_cell(left)
        right_cell = normalize_cell(right)
        if left_cell and right_cell:
            pairs.append((left_cell, right_cell))
    return pairs


def parse_counts(counts_text: object) -> Dict[str, float]:
    if pd.isna(counts_text):
        return {}
    counts = {}
    for part in str(counts_text).split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        patient, value = part.split("=", 1)
        try:
            counts[patient.strip()] = float(value.strip())
        except ValueError:
            counts[patient.strip()] = 0.0
    return counts


def load_run_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "motif_id" not in df.columns:
        raise ValueError(f"Missing motif_id column in {path}")
    return df


def class_weighted_distributions(
    df: pd.DataFrame, patient_classes: Dict[str, str], labels: List[str]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    needed = {"nodes", "edges", "counts_by_patient"}
    if not needed.issubset(df.columns):
        return None, None

    cell_counts = {label: Counter() for label in labels}
    edge_counts = {label: Counter() for label in labels}

    for _, row in df.iterrows():
        nodes = parse_nodes(row["nodes"])
        edges = parse_edges(row["edges"])
        counts_by_patient = parse_counts(row["counts_by_patient"])
        label_weights = {label: 0.0 for label in labels}
        for patient, count in counts_by_patient.items():
            label = patient_classes.get(patient)
            if label in label_weights:
                label_weights[label] += count
        for label, weight in label_weights.items():
            if weight <= 0:
                continue
            for cell in nodes:
                cell_counts[label][cell] += weight
            for left, right in edges:
                edge_counts[label][(left, right)] += weight

    cell_records = []
    for label in labels:
        total = sum(cell_counts[label].values())
        for cell, count in cell_counts[label].items():
            cell_records.append(
                {
                    "class": label,
                    "cell_type": cell,
                    "fraction": count / total if total else 0.0,
                    "raw_count": count,
                }
            )

    edge_records = []
    for label in labels:
        total = sum(edge_counts[label].values())
        for (left, right), count in edge_counts[label].items():
            edge_records.append(
                {
                    "class": label,
                    "source": left,
                    "target": right,
                    "probability": count / total if total else 0.0,
                    "raw_count": count,
                }
            )

    return pd.DataFrame(cell_records), pd.DataFrame(edge_records)


def unlabelled_edge_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter()
    for _, row in df.iterrows():
        for left, right in parse_edges(row.get("edges", "")):
            counts[(left, right)] += 1
    total = sum(counts.values())
    return pd.DataFrame(
        [
            {
                "source": left,
                "target": right,
                "probability": count / total if total else 0.0,
                "raw_count": count,
            }
            for (left, right), count in counts.items()
        ]
    )


def counts_matrix(df: pd.DataFrame, patient_classes: Dict[str, str], labels: List[str]) -> Optional[pd.DataFrame]:
    if "counts_by_patient" not in df.columns:
        return None
    records = []
    for _, row in df.iterrows():
        motif_id = str(row["motif_id"])
        counts = parse_counts(row["counts_by_patient"])
        records.append(
            {
                "motif_id": motif_id,
                **{patient: counts.get(patient, 0.0) for patient, label in patient_classes.items() if label in labels},
            }
        )
    if not records:
        return None
    matrix = pd.DataFrame(records).groupby("motif_id", as_index=True).sum()
    return matrix


def average_distribution(
    run_frames: List[pd.DataFrame],
    group_cols: List[str],
    value_col: str,
    fill_index: Optional[pd.MultiIndex] = None,
) -> pd.DataFrame:
    if not run_frames:
        return pd.DataFrame()
    normalized = []
    for i, frame in enumerate(run_frames):
        if frame.empty:
            continue
        local = frame[group_cols + [value_col]].copy()
        local["run_ix"] = i
        normalized.append(local)
    if not normalized:
        return pd.DataFrame()
    merged = pd.concat(normalized, ignore_index=True)
    per_run = merged.groupby(["run_ix"] + group_cols, as_index=False)[value_col].sum()
    if fill_index is not None:
        filled = []
        for run_ix in per_run["run_ix"].unique():
            local = per_run[per_run["run_ix"] == run_ix].set_index(group_cols)
            local = local.reindex(fill_index, fill_value=0).reset_index()
            local["run_ix"] = run_ix
            filled.append(local)
        per_run = pd.concat(filled, ignore_index=True)
    summary = (
        per_run.groupby(group_cols)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"mean_{value_col}", "std": f"sd_{value_col}", "count": "n_runs"})
    )
    return summary


def auc_score(y_true: List[str], scores: List[float], positive_label: str) -> float:
    y = np.array([label == positive_label for label in y_true])
    s = np.array(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan

    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = rank
        i = j + 1

    sum_pos = ranks[y].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_stringency_auc(
    matrix: pd.DataFrame,
    patient_classes: Dict[str, str],
    labels: List[str],
    thresholds: List[float],
    max_class_features: int,
) -> pd.DataFrame:
    patients = [patient for patient in matrix.columns if patient_classes.get(patient) in labels]
    patient_label_values = np.array([patient_classes[patient] for patient in patients])
    values = matrix[patients].fillna(0.0).to_numpy(dtype=float)
    label_masks = {label: patient_label_values == label for label in labels}

    records = []
    for threshold in thresholds:
        y_true = []
        scores = []
        feature_counts = []

        for heldout_ix, heldout in enumerate(patients):
            train_mask = np.ones(len(patients), dtype=bool)
            train_mask[heldout_ix] = False
            selected_by_label = {label: [] for label in labels}

            stats = {}
            for label in labels:
                class_mask = train_mask & label_masks[label]
                class_values = values[:, class_mask]
                if class_values.shape[1] == 0:
                    stats[label] = {
                        "pct": np.zeros(values.shape[0], dtype=float),
                        "total": np.zeros(values.shape[0], dtype=float),
                    }
                else:
                    stats[label] = {
                        "pct": (class_values > 0).mean(axis=1),
                        "total": class_values.sum(axis=1),
                    }

            label0_valid = (stats[labels[0]]["pct"] > threshold) & (stats[labels[1]]["pct"] == 0)
            label1_valid = (stats[labels[1]]["pct"] > threshold) & (stats[labels[0]]["pct"] == 0)
            selected_ix = {}
            for label, valid in [(labels[0], label0_valid), (labels[1], label1_valid)]:
                candidate_ix = np.flatnonzero(valid)
                if len(candidate_ix) == 0:
                    selected_ix[label] = np.array([], dtype=int)
                    continue
                order = np.lexsort((stats[label]["pct"][candidate_ix], stats[label]["total"][candidate_ix]))
                selected_ix[label] = candidate_ix[order[::-1]][:max_class_features]

            label0_score = float(values[selected_ix[labels[0]], heldout_ix].sum()) if len(selected_ix[labels[0]]) else 0.0
            label1_score = float(values[selected_ix[labels[1]], heldout_ix].sum()) if len(selected_ix[labels[1]]) else 0.0
            y_true.append(patient_classes[heldout])
            scores.append(label0_score - label1_score)
            feature_counts.append(len(selected_ix[labels[0]]) + len(selected_ix[labels[1]]))

        records.append(
            {
                "stringency": threshold,
                "auc": auc_score(y_true, scores, labels[0]),
                "mean_selected_features": float(np.mean(feature_counts)) if feature_counts else 0.0,
            }
        )

    return pd.DataFrame(records)


def plot_cell_fractions(
    cell_summary: pd.DataFrame, out_path: Path, labels: List[str], panel_conditions: Optional[List[str]] = None
) -> None:
    if panel_conditions is None:
        available = sorted(cell_summary["condition"].unique())
    else:
        available = [condition for condition in panel_conditions if condition in set(cell_summary["condition"])]
    if not available:
        return
    cell_order = [cell for cell in CELL_ORDER if cell in set(cell_summary["cell_type"])]
    extra = sorted(set(cell_summary["cell_type"]) - set(cell_order))
    cell_order.extend(extra)

    fig, axes = plt.subplots(len(available), 1, figsize=(10, max(4, 3.8 * len(available))), sharex=True)
    if len(available) == 1:
        axes = [axes]

    y = np.arange(len(cell_order))
    bar_h = 0.36
    colors = {labels[0]: "#5B9BD5", labels[1]: "#C9565A"}
    offsets = {labels[0]: -bar_h / 2, labels[1]: bar_h / 2}

    for ax, condition in zip(axes, available):
        local = cell_summary[cell_summary["condition"] == condition]
        for label in labels:
            values = []
            errors = []
            for cell in cell_order:
                row = local[(local["class"] == label) & (local["cell_type"] == cell)]
                values.append(float(row["mean_fraction"].iloc[0]) if not row.empty else 0.0)
                errors.append(float(row["sd_fraction"].fillna(0).iloc[0]) if not row.empty else 0.0)
            ax.barh(y + offsets[label], values, height=bar_h, label=label, color=colors.get(label), xerr=errors)
        ax.set_yticks(y)
        ax.set_yticklabels(cell_order)
        ax.invert_yaxis()
        ax.set_title(condition)
        ax.set_xlabel("Fraction")
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="upper right")
    fig.suptitle("Average Motif-Induced Cell Type Fraction Across Runs", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def matrix_from_edges(edge_summary: pd.DataFrame, condition: str, label: str, value_col: str) -> np.ndarray:
    mat = np.zeros((len(CELL_ORDER), len(CELL_ORDER)), dtype=float)
    index = {cell: i for i, cell in enumerate(CELL_ORDER)}
    local = edge_summary[(edge_summary["condition"] == condition) & (edge_summary["class"] == label)]
    for _, row in local.iterrows():
        source = row["source"]
        target = row["target"]
        if source not in index or target not in index:
            continue
        mat[index[source], index[target]] += row[value_col]
    return mat


def plot_pairwise_heatmaps(
    edge_summary: pd.DataFrame, out_path: Path, labels: List[str], panel_conditions: Optional[List[str]] = None
) -> None:
    if panel_conditions is None:
        available = sorted(edge_summary["condition"].unique())
    else:
        available = [condition for condition in panel_conditions if condition in set(edge_summary["condition"])]
    if not available:
        return
    vmax = float(edge_summary["mean_probability"].max()) if not edge_summary.empty else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, axes = plt.subplots(
        len(available),
        len(labels),
        figsize=(7.2 * len(labels), max(4.8, 4.7 * len(available))),
        squeeze=False,
    )
    cmaps = {labels[0]: "Blues", labels[1]: "Reds"}
    for row_ix, condition in enumerate(available):
        for col_ix, label in enumerate(labels):
            ax = axes[row_ix][col_ix]
            mat = matrix_from_edges(edge_summary, condition, label, "mean_probability")
            im = ax.imshow(mat, cmap=cmaps.get(label, "viridis"), vmin=0, vmax=vmax)
            ax.set_title(f"{condition} - {label}")
            ax.set_xticks(np.arange(len(CELL_ORDER)))
            ax.set_yticks(np.arange(len(CELL_ORDER)))
            ax.set_xticklabels(CELL_ORDER, rotation=90, fontsize=8)
            ax.set_yticklabels(CELL_ORDER, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Average Motif-Induced Pairwise Interaction Probability", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_unlabelled_pair_bars(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    summary = summary.copy()
    summary["pair"] = summary["source"] + " - " + summary["target"]
    top_pairs = (
        summary.groupby("pair")["mean_probability"].mean().sort_values(ascending=False).head(18).index.tolist()
    )
    local = summary[summary["pair"].isin(top_pairs)].copy()
    pivot = local.pivot_table(index="pair", columns="condition", values="mean_probability", fill_value=0.0)
    pivot = pivot.loc[top_pairs]
    ax = pivot.plot(kind="barh", figsize=(11, 7), width=0.82)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_ylabel("")
    ax.set_title("Average Unlabelled Motif-Induced Pair Distribution")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_auc_curve(curves: pd.DataFrame, out_path: Path) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in sorted(curves["condition"].unique()):
        local = curves[curves["condition"] == condition].sort_values("stringency")
        ax.plot(local["stringency"], local["mean_auc"], marker="o", label=condition)
        lower = local["mean_auc"] - local["sem_auc"].fillna(0)
        upper = local["mean_auc"] + local["sem_auc"].fillna(0)
        ax.fill_between(local["stringency"], lower, upper, alpha=0.15)
    ax.set_xlabel("Stringency / shared_percentage")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.set_title("AUC Across Stringency Thresholds")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_noise_summary(summary: pd.DataFrame, value: str, ylabel: str, title: str, out_path: Path) -> None:
    if summary.empty:
        return
    local = summary.sort_values(["noise_rate", "condition"])
    x = np.arange(len(local))
    fig, ax = plt.subplots(figsize=(8, 5))
    y = local[f"mean_{value}"].astype(float)
    err = local[f"sd_{value}"].fillna(0).astype(float)
    ax.errorbar(x, y, yerr=err, fmt="o-", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(local["condition"], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if value == "auc":
        ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_by_run(records: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return ""
    columns = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rows.append(values)
    widths = [
        max(len(str(col)), *(len(values[ix]) for values in rows))
        for ix, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[ix]) for ix, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[ix] for ix in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(values[ix].ljust(widths[ix]) for ix in range(len(columns))) + " |"
        for values in rows
    ]
    return "\n".join([header, sep] + body)


def write_report(
    out_dir: Path,
    run_inventory: pd.DataFrame,
    auc_summary: pd.DataFrame,
    skipped: List[str],
    labels: List[str],
    panel_conditions: Optional[List[str]],
) -> None:
    lines = [
        "# Motif Noise Summary",
        "",
        "This report averages motif-induced distributions across runs within each noise condition.",
        f"Disease-state plots and AUC/stringency estimates use patient-count columns for {labels[0]} vs {labels[1]}.",
        "",
        "## Input Inventory",
        "",
        dataframe_to_markdown(run_inventory) if not run_inventory.empty else "No motif files were found.",
        "",
    ]
    if not auc_summary.empty:
        lines.extend(
            [
                "## Noise Effect on Optimal Stringency and AUC",
                "",
                dataframe_to_markdown(auc_summary, floatfmt=".4f"),
                "",
            ]
        )
    if skipped:
        lines.extend(["## Data Limitations", ""])
        lines.extend(f"- {item}" for item in skipped)
        lines.append("")
    if panel_conditions:
        lines.extend(
            [
                "## Replicated Panel Conditions",
                "",
                "The screenshot-style cell-fraction and heatmap panels are shown only for:",
                "",
            ]
        )
        lines.extend(f"- {condition}" for condition in panel_conditions)
        lines.append("")
    lines.extend(
        [
            "## Generated Files",
            "",
            "- `average_cell_type_fraction_by_noise.png`",
            "- `average_pairwise_heatmaps_by_noise.png`",
            "- `average_unlabelled_pair_distribution_by_noise.png`",
            "- `auc_by_noise.png`",
            "- `optimal_stringency_by_noise.png`",
            "- `auc_vs_stringency_by_noise.png`",
            "- CSV tables in this output directory",
        ]
    )
    (out_dir / "motif_noise_summary_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_classes = read_patient_classes(args.patient_class, args.labels)

    cell_run_frames: List[pd.DataFrame] = []
    edge_run_frames: List[pd.DataFrame] = []
    unlabelled_edge_run_frames: List[pd.DataFrame] = []
    auc_records: List[pd.DataFrame] = []
    inventory_records: List[Dict[str, object]] = []
    skipped: List[str] = []

    class_cell_index = pd.MultiIndex.from_product([args.labels, CELL_ORDER], names=["class", "cell_type"])
    class_edge_index = pd.MultiIndex.from_product(
        [args.labels, CELL_ORDER, CELL_ORDER], names=["class", "source", "target"]
    )
    unlabelled_edge_index = pd.MultiIndex.from_product([CELL_ORDER, CELL_ORDER], names=["source", "target"])

    for condition, noise_rate, motifs_dir in condition_paths(args.input_root):
        files = motif_files(motifs_dir)
        if not files:
            skipped.append(f"{condition}: no discriminative motif CSVs found at {motifs_dir}")
            continue

        condition_cell_frames = []
        condition_edge_frames = []
        condition_unlabelled_frames = []
        condition_auc_frames = []
        with_patient_counts = 0

        for csv_path in files:
            run_id = run_id_from_path(csv_path)
            df = load_run_table(csv_path)
            has_patient_counts = {"nodes", "counts_by_patient"}.issubset(df.columns)
            if has_patient_counts:
                with_patient_counts += 1

            cell_df, edge_df = class_weighted_distributions(df, patient_classes, args.labels)
            if cell_df is not None and edge_df is not None:
                cell_df["condition"] = condition
                cell_df["noise_rate"] = noise_rate
                cell_df["run_id"] = run_id
                edge_df["condition"] = condition
                edge_df["noise_rate"] = noise_rate
                edge_df["run_id"] = run_id
                condition_cell_frames.append(cell_df)
                condition_edge_frames.append(edge_df)

            unlabelled = unlabelled_edge_distribution(df)
            if not unlabelled.empty:
                unlabelled["condition"] = condition
                unlabelled["noise_rate"] = noise_rate
                unlabelled["run_id"] = run_id
                condition_unlabelled_frames.append(unlabelled)

            matrix = counts_matrix(df, patient_classes, args.labels)
            if matrix is not None:
                auc_df = evaluate_stringency_auc(
                    matrix=matrix,
                    patient_classes=patient_classes,
                    labels=args.labels,
                    thresholds=thresholds,
                    max_class_features=args.max_class_features,
                )
                auc_df["condition"] = condition
                auc_df["noise_rate"] = noise_rate
                auc_df["run_id"] = run_id
                condition_auc_frames.append(auc_df)

        inventory_records.append(
            {
                "condition": condition,
                "noise_rate": noise_rate,
                "motif_files": len(files),
                "files_with_patient_counts": with_patient_counts,
                "path": str(motifs_dir),
            }
        )
        if files and with_patient_counts == 0:
            skipped.append(
                f"{condition}: files contain edges only, so NN/NP weighted panels and AUC/stringency were not computed."
            )

        if condition_cell_frames:
            cell_run_frames.extend(condition_cell_frames)
        if condition_edge_frames:
            edge_run_frames.extend(condition_edge_frames)
        if condition_unlabelled_frames:
            unlabelled_edge_run_frames.extend(condition_unlabelled_frames)
        if condition_auc_frames:
            auc_records.extend(condition_auc_frames)

    cell_by_run = summarize_by_run(cell_run_frames)
    edge_by_run = summarize_by_run(edge_run_frames)
    unlabelled_by_run = summarize_by_run(unlabelled_edge_run_frames)
    auc_by_run = pd.concat(auc_records, ignore_index=True) if auc_records else pd.DataFrame()
    inventory = pd.DataFrame.from_records(inventory_records)

    inventory.to_csv(out_dir / "motif_input_inventory.csv", index=False)
    if not cell_by_run.empty:
        cell_by_run.to_csv(out_dir / "cell_type_fraction_by_run.csv", index=False)
    if not edge_by_run.empty:
        edge_by_run.to_csv(out_dir / "pairwise_probability_by_run.csv", index=False)
    if not unlabelled_by_run.empty:
        unlabelled_by_run.to_csv(out_dir / "unlabelled_pairwise_probability_by_run.csv", index=False)
    if not auc_by_run.empty:
        auc_by_run.to_csv(out_dir / "auc_by_stringency_by_run.csv", index=False)

    panel_conditions: Optional[List[str]] = None
    cell_summary = pd.DataFrame()
    if not cell_by_run.empty:
        summaries = []
        for condition in sorted(cell_by_run["condition"].unique()):
            local_frames = []
            for _, run_frame in cell_by_run[cell_by_run["condition"] == condition].groupby("run_id"):
                local_frames.append(run_frame)
            summary = average_distribution(local_frames, ["class", "cell_type"], "fraction", class_cell_index)
            summary["condition"] = condition
            summary["noise_rate"] = float(cell_by_run[cell_by_run["condition"] == condition]["noise_rate"].iloc[0])
            summaries.append(summary)
        cell_summary = pd.concat(summaries, ignore_index=True)
        cell_summary.to_csv(out_dir / "average_cell_type_fraction_by_noise.csv", index=False)
        panel_conditions = select_panel_conditions(cell_summary, args.panel_conditions)
        plot_cell_fractions(
            cell_summary,
            out_dir / "average_cell_type_fraction_by_noise.png",
            args.labels,
            panel_conditions=panel_conditions,
        )

    edge_summary = pd.DataFrame()
    if not edge_by_run.empty:
        summaries = []
        for condition in sorted(edge_by_run["condition"].unique()):
            local_frames = []
            for _, run_frame in edge_by_run[edge_by_run["condition"] == condition].groupby("run_id"):
                local_frames.append(run_frame)
            summary = average_distribution(local_frames, ["class", "source", "target"], "probability", class_edge_index)
            summary["condition"] = condition
            summary["noise_rate"] = float(edge_by_run[edge_by_run["condition"] == condition]["noise_rate"].iloc[0])
            summaries.append(summary)
        edge_summary = pd.concat(summaries, ignore_index=True)
        edge_summary.to_csv(out_dir / "average_pairwise_probability_by_noise.csv", index=False)
        if panel_conditions is None:
            panel_conditions = select_panel_conditions(edge_summary, args.panel_conditions)
        plot_pairwise_heatmaps(
            edge_summary,
            out_dir / "average_pairwise_heatmaps_by_noise.png",
            args.labels,
            panel_conditions=panel_conditions,
        )

    if not unlabelled_by_run.empty:
        summaries = []
        for condition in sorted(unlabelled_by_run["condition"].unique()):
            local_frames = []
            for _, run_frame in unlabelled_by_run[unlabelled_by_run["condition"] == condition].groupby("run_id"):
                local_frames.append(run_frame)
            summary = average_distribution(
                local_frames, ["source", "target"], "probability", unlabelled_edge_index
            )
            summary["condition"] = condition
            summary["noise_rate"] = float(
                unlabelled_by_run[unlabelled_by_run["condition"] == condition]["noise_rate"].iloc[0]
            )
            summaries.append(summary)
        unlabelled_summary = pd.concat(summaries, ignore_index=True)
        unlabelled_summary.to_csv(out_dir / "average_unlabelled_pairwise_probability_by_noise.csv", index=False)
        plot_unlabelled_pair_bars(
            unlabelled_summary, out_dir / "average_unlabelled_pair_distribution_by_noise.png"
        )

    auc_summary = pd.DataFrame()
    if not auc_by_run.empty:
        curve_summary = (
            auc_by_run.groupby(["condition", "noise_rate", "stringency"])["auc"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_auc", "std": "sd_auc", "count": "n_runs"})
        )
        curve_summary["sem_auc"] = curve_summary["sd_auc"] / np.sqrt(curve_summary["n_runs"])
        curve_summary.to_csv(out_dir / "average_auc_by_stringency_and_noise.csv", index=False)
        plot_auc_curve(curve_summary, out_dir / "auc_vs_stringency_by_noise.png")

        optimal_rows = []
        for (condition, run_id), local in auc_by_run.groupby(["condition", "run_id"]):
            local = local.sort_values(["auc", "stringency"], ascending=[False, True])
            best = local.iloc[0]
            optimal_rows.append(
                {
                    "condition": condition,
                    "noise_rate": float(best["noise_rate"]),
                    "run_id": int(run_id),
                    "optimal_stringency": float(best["stringency"]),
                    "auc": float(best["auc"]),
                    "mean_selected_features": float(best["mean_selected_features"]),
                }
            )
        optimal = pd.DataFrame.from_records(optimal_rows)
        optimal.to_csv(out_dir / "optimal_stringency_and_auc_by_run.csv", index=False)

        auc_summary = (
            optimal.groupby(["condition", "noise_rate"], as_index=False)
            .agg(
                mean_optimal_stringency=("optimal_stringency", "mean"),
                sd_optimal_stringency=("optimal_stringency", "std"),
                mean_auc=("auc", "mean"),
                sd_auc=("auc", "std"),
                n_runs=("run_id", "count"),
            )
            .sort_values(["noise_rate", "condition"])
        )
        auc_summary.to_csv(out_dir / "noise_effect_on_stringency_and_auc.csv", index=False)
        plot_noise_summary(
            auc_summary,
            "optimal_stringency",
            "Optimal stringency",
            "Optimal Stringency by Noise Condition",
            out_dir / "optimal_stringency_by_noise.png",
        )
        plot_noise_summary(auc_summary, "auc", "AUC", "Best AUC by Noise Condition", out_dir / "auc_by_noise.png")

    write_report(out_dir, inventory, auc_summary, skipped, args.labels, panel_conditions)
    print(f"Wrote motif noise summary outputs to {out_dir}")


if __name__ == "__main__":
    main()
