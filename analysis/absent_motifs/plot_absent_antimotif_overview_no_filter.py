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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


REPO_ROOT = _REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

from cism.helpers import string_base64_pickle  # noqa: E402


DEFAULT_BASE_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
)
DEFAULT_ARTIFACT = DEFAULT_BASE_DIR / "Melanoma_NN_NP_absent_cism_ready.pkl"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "absent_antimotif_overview_no_filter"

CELL_TYPE_ORDER = [
    "b cell",
    "cd4 apc",
    "cd4 t cell",
    "treg",
    "cd8 t cell",
    "DC",
    "gc",
    "hevs",
    "mac",
    "memory%cd4 t cell",
    "nk cell",
    "neutrophil",
    "stroma",
    "unidentified",
    "blood vessels",
]

DISPLAY_LABELS = {
    "b cell": "B cell",
    "cd4 apc": "CD4 APC",
    "cd4 t cell": "CD4 T cell",
    "treg": "CD4 Treg",
    "cd8 t cell": "CD8 T cell",
    "DC": "DCs",
    "gc": "Germinal Center cell",
    "hevs": "HEVs",
    "mac": "MAC",
    "memory%cd4 t cell": "Memory CD4 T cell",
    "nk cell": "NK cell",
    "neutrophil": "Neutrophil",
    "stroma": "Stroma",
    "unidentified": "Unidentified",
    "blood vessels": "Vessels",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Figure S3-style overview for no-filter absent anti-motifs."
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    parser.add_argument("--p-bin-width", type=float, default=0.001)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)

    motifs = artifact["cism"].motifs_dataset.copy()
    motifs = motifs[motifs["patient_class"].isin(args.labels)].copy()
    if "p_value_direction" in motifs.columns:
        motifs = motifs[motifs["p_value_direction"] == "absence"].copy()
    motifs["ID"] = motifs["ID"].astype("int64")
    p_col = "p_value_normal_two_tailed" if "p_value_normal_two_tailed" in motifs.columns else "p_value"
    weight_col = "absence_depletion" if "absence_depletion" in motifs.columns else "Freq"

    cell_type_by_code = {str(code): label for code, label in artifact["common_cells_type"].items()}
    motif_lookup = motifs.drop_duplicates("ID").set_index("ID")["motif"]
    graph_cache = {int(motif_id): deserialize_graph(motif_obj) for motif_id, motif_obj in motif_lookup.items()}

    pvalue_counts = create_pvalue_count_table(motifs, p_col, args.p_bin_width)
    cell_fractions = create_cell_fraction_table(motifs, graph_cache, cell_type_by_code, args.labels)
    matrices = {
        label: create_pairwise_matrix(motifs, graph_cache, cell_type_by_code, label, weight_col)
        for label in args.labels
    }

    pvalue_counts.to_csv(args.output_dir / "panel_a_motif_count_vs_p_value.csv", index=False)
    cell_fractions.to_csv(args.output_dir / "panel_b_cell_type_fractions.csv", index=False)
    for label, matrix in matrices.items():
        matrix.to_csv(args.output_dir / f"panel_c_{label}_no_filter_pairwise_matrix.csv")

    plot_panel_a(pvalue_counts, args.output_dir / "panel_a_motif_count_vs_p_value.png")
    plot_panel_b(cell_fractions, args.output_dir / "panel_b_cell_type_fractions.png")
    plot_panel_c(matrices, args.output_dir / "panel_c_no_filter_pairwise_distribution.png", args.labels)
    plot_composite(
        pvalue_counts=pvalue_counts,
        cell_fractions=cell_fractions,
        matrices=matrices,
        labels=args.labels,
        output=args.output_dir / "absent_antimotif_no_filter_overview_figure.png",
    )

    summary = {
        "artifact": str(args.artifact),
        "labels": args.labels,
        "p_value_column": p_col,
        "weight_column": weight_col,
        "significance_filter": "p_value_direction == absence; no discriminative/stringency/prevalence filter",
        "motif_rows": int(len(motifs)),
        "unique_motifs_aggregated": int(motifs["ID"].nunique()),
        "unique_motifs_by_class": {
            label: int(motifs.loc[motifs["patient_class"] == label, "ID"].nunique())
            for label in args.labels
        },
    }
    (args.output_dir / "absent_antimotif_no_filter_overview_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def deserialize_graph(motif_obj):
    if isinstance(motif_obj, (nx.Graph, nx.DiGraph)):
        return motif_obj
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    raise TypeError(f"Unsupported motif graph representation: {type(motif_obj)!r}")


def create_pvalue_count_table(motifs: pd.DataFrame, p_col: str, bin_width: float) -> pd.DataFrame:
    motif_min_p = motifs.groupby("ID", observed=True)[p_col].min().clip(lower=0, upper=0.05)
    edges = np.arange(0, 0.05 + bin_width, bin_width)
    counts, bin_edges = np.histogram(motif_min_p.to_numpy(), bins=edges)
    return pd.DataFrame(
        {
            "p_value_left": bin_edges[:-1],
            "p_value_right": bin_edges[1:],
            "p_value_mid": (bin_edges[:-1] + bin_edges[1:]) / 2,
            "unique_motif_count": counts.astype(int),
        }
    )


def create_cell_fraction_table(
    motifs: pd.DataFrame,
    graph_cache: dict[int, nx.Graph],
    cell_type_by_code: dict[str, str],
    labels: list[str],
) -> pd.DataFrame:
    rows = []
    for label in labels:
        motif_ids = sorted(motifs.loc[motifs["patient_class"] == label, "ID"].unique())
        counts = {cell_type: 0 for cell_type in CELL_TYPE_ORDER}
        for motif_id in motif_ids:
            graph = graph_cache[int(motif_id)]
            for _node, data in graph.nodes(data=True):
                cell_type = cell_type_by_code.get(str(data.get("type")), str(data.get("type")))
                if cell_type in counts:
                    counts[cell_type] += 1
        denominator = sum(counts.values())
        for cell_type in CELL_TYPE_ORDER:
            rows.append(
                {
                    "patient_class": label,
                    "cell_type": cell_type,
                    "cell_type_label": DISPLAY_LABELS[cell_type],
                    "count": int(counts[cell_type]),
                    "fraction": float(counts[cell_type] / denominator) if denominator else 0.0,
                    "unique_motifs": int(len(motif_ids)),
                    "total_motif_nodes": int(denominator),
                }
            )
    return pd.DataFrame(rows)


def create_pairwise_matrix(
    motifs: pd.DataFrame,
    graph_cache: dict[int, nx.Graph],
    cell_type_by_code: dict[str, str],
    label: str,
    weight_col: str,
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=CELL_TYPE_ORDER, columns=CELL_TYPE_ORDER, dtype=float).fillna(0.0)
    class_motifs = motifs[motifs["patient_class"] == label]
    weights = class_motifs.groupby("ID", observed=True)[weight_col].mean()

    for motif_id, weight in weights.items():
        graph = graph_cache[int(motif_id)]
        for left, right in nx.Graph(graph).edges():
            left_type = cell_type_by_code.get(str(graph.nodes[left].get("type")), str(graph.nodes[left].get("type")))
            right_type = cell_type_by_code.get(str(graph.nodes[right].get("type")), str(graph.nodes[right].get("type")))
            if left_type in matrix.index and right_type in matrix.columns:
                matrix.loc[left_type, right_type] += float(weight)

    total = float(matrix.to_numpy().sum())
    if total > 0:
        matrix = matrix / total
    matrix.index = [DISPLAY_LABELS[cell_type] for cell_type in matrix.index]
    matrix.columns = [DISPLAY_LABELS[cell_type] for cell_type in matrix.columns]
    return matrix


def plot_panel_a(counts: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=220)
    draw_panel_a(ax, counts)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_panel_b(fractions: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=220)
    draw_panel_b(ax, fractions)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_panel_c(matrices: dict[str, pd.DataFrame], output: Path, labels: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=220, gridspec_kw={"wspace": 0.42})
    draw_pairwise_heatmap(fig, axes[0], matrices[labels[0]], f"{labels[0]} disease state", "Blues")
    draw_pairwise_heatmap(fig, axes[1], matrices[labels[1]], f"{labels[1]} disease state", "Reds")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_composite(
    pvalue_counts: pd.DataFrame,
    cell_fractions: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    labels: list[str],
    output: Path,
) -> None:
    fig = plt.figure(figsize=(11.2, 8.6), dpi=220)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18], width_ratios=[0.92, 1.35], hspace=0.45, wspace=0.36)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    subgrid = grid[1, :].subgridspec(1, 2, wspace=0.34)
    ax_c1 = fig.add_subplot(subgrid[0, 0])
    ax_c2 = fig.add_subplot(subgrid[0, 1])

    draw_panel_a(ax_a, pvalue_counts)
    draw_panel_b(ax_b, cell_fractions)
    draw_pairwise_heatmap(fig, ax_c1, matrices[labels[0]], f"{labels[0]} disease state", "Blues")
    draw_pairwise_heatmap(fig, ax_c2, matrices[labels[1]], f"{labels[1]} disease state", "Reds")

    ax_a.text(-0.24, 1.08, "A", transform=ax_a.transAxes, fontsize=13, fontweight="bold")
    ax_b.text(-0.18, 1.08, "B", transform=ax_b.transAxes, fontsize=13, fontweight="bold")
    ax_c1.text(-0.20, 1.08, "C", transform=ax_c1.transAxes, fontsize=13, fontweight="bold")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def draw_panel_a(ax, counts: pd.DataFrame) -> None:
    nonzero = counts[counts["unique_motif_count"] > 0]
    gap = (nonzero["p_value_right"] - nonzero["p_value_left"]) * 0.08
    ax.hlines(
        nonzero["unique_motif_count"],
        nonzero["p_value_left"] + gap,
        nonzero["p_value_right"] - gap,
        color="#6d6d6d",
        linewidth=1.1,
    )
    ax.set_yscale("log")
    ax.set_xlabel("P-value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 0.05)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_b(ax, fractions: pd.DataFrame) -> None:
    order = (
        fractions.groupby("cell_type", observed=True)["fraction"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    display_order = [cell_type for cell_type in order if cell_type in CELL_TYPE_ORDER]
    pivot = (
        fractions.pivot(index="cell_type", columns="patient_class", values="fraction")
        .reindex(display_order)
        .fillna(0.0)
    )
    y = np.arange(len(display_order))
    height = 0.32
    ax.barh(y - height / 2, pivot.get("NN", 0), height=height, color="#5b9bd5", label="NN")
    ax.barh(y + height / 2, pivot.get("NP", 0), height=height, color="#c95757", label="NP")
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_LABELS[cell_type] for cell_type in display_order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction", fontsize=8)
    ax.set_xlim(0, max(0.30, float(pivot.max().max()) * 1.08))
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def draw_pairwise_heatmap(fig, ax, matrix: pd.DataFrame, title: str, cmap: str) -> None:
    data = matrix.to_numpy(dtype=float)
    vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 0.0
    image = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax or None)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Probability", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


if __name__ == "__main__":
    main()
