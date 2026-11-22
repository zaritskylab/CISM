from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.helpers import string_base64_pickle


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "Melanoma_NN_NP_absent_cism_ready.pkl"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "class_absent_motifs_90pct"
    / "class_absent_motifs_90pct.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "class_depleted_pairwise_heatmaps"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pairwise interaction heatmaps for class-depleted absent motif sets."
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--class-absent-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)

    motifs = artifact["cism"].motifs_dataset.copy()
    patient_class_df = artifact["discriminator"].patient_class_df.copy()
    report = pd.read_csv(args.class_absent_report)
    labels = list(args.labels)

    class_sets = {
        label: set(report.loc[report["patient_class"] == label, "ID"].astype("int64"))
        for label in labels
    }
    intersection = set.intersection(*(class_sets[label] for label in labels))

    set_specs = [
        {
            "name": f"{labels[0]}_depleted",
            "title": f"{labels[0]} depleted motifs",
            "motif_ids": class_sets[labels[0]],
            "patient_classes": [labels[0]],
            "cmap": "Blues",
        },
        {
            "name": f"{labels[1]}_depleted",
            "title": f"{labels[1]} depleted motifs",
            "motif_ids": class_sets[labels[1]],
            "patient_classes": [labels[1]],
            "cmap": "Reds",
        },
        {
            "name": f"{labels[0]}_{labels[1]}_intersection_depleted",
            "title": f"{labels[0]} and {labels[1]} depleted motif intersection",
            "motif_ids": intersection,
            "patient_classes": labels,
            "cmap": "Purples",
        },
    ]

    matrices = {}
    top_rows = []
    summary_sets = {}
    for spec in set_specs:
        patient_ids = patient_class_df[
            patient_class_df["patient_class"].isin(spec["patient_classes"])
        ].index
        weights = calculate_motif_mean_weights(motifs, spec["motif_ids"], patient_ids)
        matrix = build_pairwise_matrix(
            motifs=motifs,
            motif_ids=spec["motif_ids"],
            motif_weights=weights,
            cells_type=artifact["common_cells_type"],
        )
        matrices[spec["name"]] = matrix
        matrix.to_csv(args.output_dir / f"{spec['name']}_pairwise.csv")
        plot_heatmap(
            matrix=matrix,
            path=args.output_dir / f"{spec['name']}_pairwise.png",
            title=spec["title"],
            cmap=spec["cmap"],
        )
        top = top_interactions(matrix, spec["name"])
        top.to_csv(args.output_dir / f"{spec['name']}_top_interactions.csv", index=False)
        top_rows.append(top)
        summary_sets[spec["name"]] = {
            "motif_count": int(len(spec["motif_ids"])),
            "patient_classes_used_for_weights": spec["patient_classes"],
            "nonzero_weighted_motifs": int(weights.gt(0).sum()),
            "matrix_sum": float(matrix.to_numpy().sum()),
        }

    intersection_name = f"{labels[0]}_{labels[1]}_intersection_depleted"
    subtraction_pairs = [
        (f"{labels[0]}_depleted", intersection_name),
        (f"{labels[1]}_depleted", intersection_name),
    ]
    for minuend_name, subtrahend_name in subtraction_pairs:
        diff_name = f"{minuend_name}_minus_intersection"
        diff = matrices[minuend_name] - matrices[subtrahend_name]
        diff.to_csv(args.output_dir / f"{diff_name}_pairwise.csv")
        plot_difference_heatmap(
            matrix=diff,
            path=args.output_dir / f"{diff_name}_pairwise.png",
            title=f"{minuend_name.replace('_', ' ')} minus shared depleted intersection",
        )
        top_diff = top_interactions(diff.abs(), diff_name)
        top_diff.to_csv(args.output_dir / f"{diff_name}_top_abs_interactions.csv", index=False)
        summary_sets[diff_name] = {
            "matrix_sum": float(diff.to_numpy().sum()),
            "min": float(diff.to_numpy().min()),
            "max": float(diff.to_numpy().max()),
            "max_abs": float(np.abs(diff.to_numpy()).max()),
        }

    all_top = pd.concat(top_rows, ignore_index=True)
    all_top.to_csv(args.output_dir / "top_interactions_all_sets.csv", index=False)

    summary = {
        "artifact": str(args.artifact),
        "class_absent_report": str(args.class_absent_report),
        "labels": labels,
        "significance_mode": artifact.get("absence_summary", {}).get("significance_mode", "normal_two_tailed"),
        "set_summary": summary_sets,
    }
    (args.output_dir / "class_depleted_pairwise_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_readme(args.output_dir, summary)
    print(json.dumps(summary, indent=2))


def calculate_motif_mean_weights(
    motifs: pd.DataFrame,
    motif_ids: set[int],
    patient_ids: pd.Index,
) -> pd.Series:
    if not motif_ids:
        return pd.Series(dtype=float)
    data = motifs[
        motifs["ID"].isin(motif_ids)
        & motifs["Patient_uId"].isin(patient_ids)
        & (motifs["Freq"] > 0)
    ].copy()
    if data.empty:
        return pd.Series(dtype=float)

    data["_total_subgraphs_for_weight"] = data["Count"] / data["Freq"]
    grouped = (
        data.groupby(["Patient_uId", "ID"], observed=True)
        .agg(
            total_count=("Count", "sum"),
            total_subgraphs=("_total_subgraphs_for_weight", "sum"),
        )
        .reset_index()
    )
    grouped["motif_depletion_frequency"] = grouped["total_count"] / grouped["total_subgraphs"]
    return grouped.groupby("ID", observed=True)["motif_depletion_frequency"].mean()


def build_pairwise_matrix(
    motifs: pd.DataFrame,
    motif_ids: set[int],
    motif_weights: pd.Series,
    cells_type: dict[int, str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=cells_type.values(), columns=cells_type.values(), dtype=float).fillna(0.0)
    if not motif_ids:
        return matrix

    motif_lookup = motifs.drop_duplicates("ID").set_index("ID")["motif"]
    for motif_id in sorted(motif_ids):
        if motif_id not in motif_lookup.index or motif_id not in motif_weights.index:
            continue
        graph = deserialize_graph(motif_lookup.loc[motif_id])
        weight = float(motif_weights.loc[motif_id])
        for left, right in nx.Graph(graph).edges():
            left_type = int(graph.nodes[left]["type"])
            right_type = int(graph.nodes[right]["type"])
            matrix.loc[cells_type[left_type], cells_type[right_type]] += weight

    total = float(np.asarray(matrix).sum())
    if total > 0:
        matrix = matrix / total
    return matrix


def deserialize_graph(motif_obj):
    if isinstance(motif_obj, (nx.Graph, nx.DiGraph)):
        return motif_obj
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    raise TypeError(f"Unsupported motif graph representation: {type(motif_obj)!r}")


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str, cmap: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_difference_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def top_interactions(matrix: pd.DataFrame, motif_set: str, top_n: int = 25) -> pd.DataFrame:
    stacked = matrix.stack().reset_index()
    stacked.columns = ["source_cell_type", "target_cell_type", "weight"]
    stacked.insert(0, "motif_set", motif_set)
    return stacked.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)


def write_readme(output_dir: Path, summary: dict) -> None:
    lines = [
        "# Class-Depleted Motif Pairwise Heatmaps",
        "",
        "These matrices use motifs depleted in at least 90% of patients in the specified class.",
        "Edges are weighted by mean CISM depletion frequency within the listed patient classes.",
        "",
        f"- Significance mode: {summary['significance_mode']}",
    ]
    for name, values in summary["set_summary"].items():
        if "motif_count" in values:
            lines.append(
                f"- {name}: {values['motif_count']:,} motifs, "
                f"weights from {', '.join(values['patient_classes_used_for_weights'])}"
            )
        else:
            lines.append(f"- {name}: max absolute difference {values['max_abs']:.4g}")
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
