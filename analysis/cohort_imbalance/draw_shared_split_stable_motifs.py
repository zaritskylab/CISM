#!/usr/bin/env python3
"""Draw motif graphs shared between native and split-FOV analyses."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.helpers import string_base64_pickle  # noqa: E402
from multi_class.data_artifacts import COMMON_CELLS_TYPE  # noqa: E402


RUNTIME = REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
ARTIFACT_PATH = RUNTIME / "serialized" / "Melanoma_cism_ready.pkl"
STABILITY_DIR = RUNTIME / "split_fov_scale_experiment" / "fold_union_stringency_046_stability"
OUTPUT_DIR = STABILITY_DIR / "motif_drawings"


CELL_COLORS = {
    "DC": "#4E79A7",
    "b cell": "#A0CBE8",
    "blood vessels": "#F28E2B",
    "cd4 apc": "#FFBE7D",
    "cd4 t cell": "#59A14F",
    "cd8 t cell": "#8CD17D",
    "gc": "#B6992D",
    "hevs": "#F1CE63",
    "mac": "#499894",
    "memory%cd4 t cell": "#86BCB6",
    "neutrophil": "#E15759",
    "nk cell": "#FF9D9A",
    "stroma": "#79706E",
    "treg": "#BAB0AC",
    "tumor": "#D37295",
    "unidentified": "#B07AA1",
}


def load_motif_lookup() -> pd.DataFrame:
    with ARTIFACT_PATH.open("rb") as handle:
        artifact = pickle.load(handle)
    motifs = artifact["cism"].motifs_dataset.copy()
    return motifs.drop_duplicates("ID").set_index("ID")


def motif_graph(motif_obj):
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    return motif_obj


def motif_labels_and_colors(graph: nx.Graph) -> tuple[dict, list[str]]:
    node_types = nx.get_node_attributes(graph, name="type")
    labels = {}
    colors = []
    for node in graph.nodes():
        cell_label = COMMON_CELLS_TYPE[int(node_types[node])]
        labels[node] = cell_label
        colors.append(CELL_COLORS.get(cell_label, "#CCCCCC"))
    return labels, colors


def draw_one_motif(ax, motif_id: int, graph: nx.Graph, direction: str, title_suffix: str = "") -> None:
    graph = graph.to_undirected()
    labels, colors = motif_labels_and_colors(graph)
    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#444444", width=2.0)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=colors,
        node_size=2500,
        edgecolors="#222222",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=7, font_color="black")
    ax.set_title(f"{motif_id}\n{direction}{title_suffix}", fontsize=9)
    ax.axis("off")


def draw_panel(motif_ids: list[int], motif_lookup: pd.DataFrame, motif_metadata: pd.DataFrame, path: Path, title: str) -> None:
    cols = 4
    rows = math.ceil(len(motif_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, motif_id in zip(axes_list, motif_ids):
        direction = motif_metadata.loc[motif_metadata["ID"].eq(motif_id), "direction"].iloc[0]
        draw_one_motif(ax, motif_id, motif_graph(motif_lookup.loc[motif_id, "motif"]), direction)
    for ax in axes_list[len(motif_ids) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_individuals(motif_ids: list[int], motif_lookup: pd.DataFrame, motif_metadata: pd.DataFrame, subdir: Path) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    for motif_id in motif_ids:
        direction = motif_metadata.loc[motif_metadata["ID"].eq(motif_id), "direction"].iloc[0]
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        draw_one_motif(ax, motif_id, motif_graph(motif_lookup.loc[motif_id, "motif"]), direction)
        fig.tight_layout()
        fig.savefig(subdir / f"motif_{motif_id}_{direction}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    motif_lookup = load_motif_lookup()
    motif_metadata = pd.read_csv(STABILITY_DIR / "native_fold_union_selected_motifs.csv")
    motif_metadata["ID"] = motif_metadata["ID"].astype(int)
    motif_metadata = motif_metadata.rename(columns={"label": "direction"})

    stable_all_splits = [
        2954952172,
        2968124908,
        4040914412,
        20168457548,
        24496979308,
        25563453932,
        25571793292,
        33076491692,
    ]
    shared_split2_split4 = [
        2954952172,
        2968124908,
        3503957420,
        4040914412,
        20168457548,
        20181555532,
        20181555628,
        24496979308,
        25563453932,
        25571793292,
        33076491692,
    ]

    manifest = motif_metadata[motif_metadata["ID"].isin(shared_split2_split4)].copy()
    manifest["stable_all_split_levels"] = manifest["ID"].isin(stable_all_splits)
    manifest["shared_split2_split4"] = True
    manifest = manifest.sort_values(["stable_all_split_levels", "direction", "ID"], ascending=[False, True, True])
    manifest.to_csv(OUTPUT_DIR / "drawn_motif_manifest.csv", index=False)

    draw_panel(
        stable_all_splits,
        motif_lookup,
        motif_metadata,
        OUTPUT_DIR / "stable_all_split_levels_8_motifs.png",
        "Stable shared motifs across all split levels",
    )
    draw_panel(
        shared_split2_split4,
        motif_lookup,
        motif_metadata,
        OUTPUT_DIR / "shared_split2_split4_11_motifs.png",
        "Motifs shared by native analysis and split-2/split-4",
    )
    draw_individuals(stable_all_splits, motif_lookup, motif_metadata, OUTPUT_DIR / "individual_stable_8")
    draw_individuals(shared_split2_split4, motif_lookup, motif_metadata, OUTPUT_DIR / "individual_shared_11")
    print(f"Wrote motif drawings to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
