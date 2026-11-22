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
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.cism import GetMotifsResult
from cism.helpers import string_base64_pickle
from cism.optimization import _clone_feature_conf


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "Melanoma_NN_NP_absent_cism_ready.pkl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "motif_induced_pairwise_stringency0"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build motif-induced pairwise interactions for two-tailed absent CISM motifs."
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    parser.add_argument("--shared-percentage", type=float, default=0.0)
    parser.add_argument("--max-class-features", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--prefer", choices=["threads", "processes"], default="threads")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)

    discriminator = artifact["discriminator"]
    cism = artifact["cism"]
    common_cells = artifact["common_cells_type"]
    feature_conf = _clone_feature_conf(
        artifact["feature_conf"],
        labels=list(args.labels),
        shared_percentage=args.shared_percentage,
        max_class_features=args.max_class_features,
    )

    motif_table = cism.motifs_dataset
    if "p_value_direction" in motif_table.columns:
        directions = motif_table["p_value_direction"].value_counts(dropna=False).to_dict()
        non_absence = motif_table[motif_table["p_value_direction"] != "absence"]
        if not non_absence.empty:
            raise ValueError(f"Expected only absence motif rows, found directions: {directions}")

    motif_space_features = discriminator.get_features(
        feature_conf=feature_conf,
        exclude_patients=[],
        n_jobs=args.n_jobs,
        prefer=args.prefer,
    )
    motif_space_features.to_csv(args.output_dir / "stringency0_motif_space_features.csv", index=False)

    motif_result = GetMotifsResult(
        motif_space_features=motif_space_features,
        discriminator=discriminator,
        classes=list(args.labels),
    )
    motif_mean_weight = motif_result.get_motifs_mean_weight()
    motif_mean_weight.to_csv(args.output_dir / "stringency0_motif_mean_weight.csv")

    motifs_by_class = motif_result.get_motifs_by_class()
    write_motifs_by_class(motifs_by_class, args.output_dir / "stringency0_motifs_by_class.csv")

    pairwise_by_class = {}
    for label in args.labels:
        matrix = create_motif_induced_pairwise_matrix(
            discriminator=discriminator,
            cells_type=common_cells,
            cell_identity_to_motif_hash=motif_result.cell_identity_to_motif_hash,
            motifs_ids=motifs_by_class[label],
            motifs_weight=motif_mean_weight,
        )
        pairwise_by_class[label] = matrix
        matrix.to_csv(args.output_dir / f"motif_induced_pairwise_{label}.csv")

    if len(args.labels) == 2:
        diff = pairwise_by_class[args.labels[0]] - pairwise_by_class[args.labels[1]]
        diff.to_csv(args.output_dir / f"motif_induced_pairwise_{args.labels[0]}_minus_{args.labels[1]}.csv")

    summary = {
        "artifact": str(args.artifact),
        "labels": args.labels,
        "shared_percentage": args.shared_percentage,
        "max_class_features": args.max_class_features,
        "significance_mode": artifact.get("absence_summary", {}).get("significance_mode", "normal_two_tailed"),
        "active_p_value": artifact.get("absence_summary", {}).get("active_p_value", "p_value_normal_two_tailed"),
        "motif_rows": int(len(motif_table)),
        "unique_absence_motifs_in_table": int(motif_table["ID"].nunique()),
        "feature_rows": int(len(motif_space_features)),
        "all_selected_unique_motifs": int(len(motif_result.all_motif_features)),
        "selected_motifs_by_class": {label: int(len(motifs_by_class[label])) for label in args.labels},
        "mean_cfeatures": float(motif_space_features["features"].map(len).mean()),
        "median_cfeatures": float(motif_space_features["features"].map(len).median()),
    }
    (args.output_dir / "motif_induced_pairwise_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_readme(summary, args.output_dir)
    plot_heatmaps(pairwise_by_class, args.output_dir)

    print(json.dumps(summary, indent=2))


def create_motif_induced_pairwise_matrix(
    discriminator,
    cells_type: dict[int, str],
    cell_identity_to_motif_hash: dict[Any, list[int]],
    motifs_ids: list[int],
    motifs_weight: pd.Series,
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=cells_type.values(), columns=cells_type.values(), dtype=float).fillna(0.0)
    motif_set = set(motifs_ids)
    motifs_dataset = discriminator.cism.motifs_dataset

    for hash_id, motif_ids_for_hash in cell_identity_to_motif_hash.items():
        for motif_id in motif_ids_for_hash:
            if motif_id not in motif_set:
                continue
            motif_obj = motifs_dataset[motifs_dataset.ID == motif_id].iloc[0].motif
            graph = deserialize_graph(motif_obj)
            weight = float(motifs_weight.loc[motif_id])
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


def write_motifs_by_class(motifs_by_class: dict[str, list[int]], path: Path) -> None:
    rows = []
    for label, motif_ids in motifs_by_class.items():
        rows.extend({"patient_class": label, "ID": motif_id} for motif_id in motif_ids)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_heatmaps(pairwise_by_class: dict[str, pd.DataFrame], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for label, matrix in pairwise_by_class.items():
        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
        cmap = "Blues" if label == "NN" else "Reds" if label == "NP" else "viridis"
        draw_heatmap(ax, matrix, cmap=cmap)
        ax.set_title(f"Motif-induced pairwise interactions: {label}")
        fig.tight_layout()
        fig.savefig(output_dir / f"motif_induced_pairwise_{label}.png")
        plt.close(fig)

    labels = list(pairwise_by_class)
    if len(labels) == 2:
        diff = pairwise_by_class[labels[0]] - pairwise_by_class[labels[1]]
        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
        vmax = float(np.nanmax(np.abs(diff.to_numpy())))
        draw_heatmap(ax, diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Motif-induced pairwise difference: {labels[0]} - {labels[1]}")
        fig.tight_layout()
        fig.savefig(output_dir / f"motif_induced_pairwise_{labels[0]}_minus_{labels[1]}.png")
        plt.close(fig)


def draw_heatmap(ax, matrix: pd.DataFrame, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def write_readme(summary: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Two-tailed absent motif-induced pairwise interactions",
        "",
        f"- Labels: {', '.join(summary['labels'])}",
        f"- Shared-percentage stringency: {summary['shared_percentage']}",
        f"- Max class features: {summary['max_class_features']}",
        f"- Significance mode: {summary['significance_mode']}",
        f"- Motif table rows: {summary['motif_rows']:,}",
        f"- Unique absent motifs in table: {summary['unique_absence_motifs_in_table']:,}",
        f"- Selected unique motifs across LOOCV folds: {summary['all_selected_unique_motifs']:,}",
        f"- Mean fold feature count: {summary['mean_cfeatures']:.3g}",
        "",
        "Outputs include class-specific motif-induced pairwise matrices and an NN-minus-NP difference matrix.",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
