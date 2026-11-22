from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


REPO_ROOT = _REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

from cism.helpers import string_base64_pickle  # noqa: E402


RUNTIME = REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
ENRICHED_ARTIFACT = RUNTIME / "serialized" / "Melanoma_cism_ready.pkl"
DEPLETED_ARTIFACT = (
    RUNTIME / "absent_discriminative_cism_nn_np" / "Melanoma_NN_NP_absent_cism_ready.pkl"
)
DEFAULT_OUTPUT = (
    RUNTIME
    / "absent_discriminative_cism_nn_np"
    / "enriched_depleted_four_motif_comparison.png"
)

MOTIFS = [
    ("Enriched", 2968124908, "#26708e"),
    ("Depleted", 1090111980, "#b64c50"),
    ("Depleted", 3501312492, "#b64c50"),
    ("Depleted", 2426531308, "#b64c50"),
]
POSITIONS = {
    "b cell": (0.0, -0.25),
    "DC": (-1.0, 0.55),
    "hevs": (1.0, 0.55),
    "unidentified": (0.0, -1.35),
}
NODE_LABELS = {
    "b cell": "B cell",
    "DC": "DC",
    "hevs": "HEV",
    "unidentified": "Unident.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare four cell-composition matched melanoma motifs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    graphs = {}
    for status, artifact_path, motif_ids in [
        ("Enriched", ENRICHED_ARTIFACT, [2968124908]),
        ("Depleted", DEPLETED_ARTIFACT, [1090111980, 3501312492, 2426531308]),
    ]:
        with artifact_path.open("rb") as handle:
            artifact = pickle.load(handle)
        motifs = artifact["cism"].motifs_dataset
        lookup = motifs[motifs["ID"].isin(motif_ids)].drop_duplicates("ID").set_index("ID")["motif"]
        if len(lookup) != len(motif_ids):
            raise ValueError(f"Could not find every {status.lower()} motif graph")
        for motif_id, obj in lookup.items():
            graph = string_base64_pickle(obj) if isinstance(obj, str) else obj
            undirected = nx.Graph(graph)
            labels = {
                node: artifact["common_cells_type"][int(data["type"])]
                for node, data in graph.nodes(data=True)
            }
            if set(labels.values()) != set(POSITIONS):
                raise ValueError(f"Unexpected cell composition for motif {motif_id}")
            nx.set_node_attributes(undirected, labels, "cell_type")
            graphs[int(motif_id)] = undirected

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7), dpi=220)
    fig.patch.set_facecolor("white")
    fig.suptitle("Composition-matched enriched and depleted motifs", y=0.985, fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.945,
        "NN/NP melanoma  |  Same four cell types; different pairwise connections",
        ha="center",
        fontsize=10,
        color="#52565b",
    )
    fig.text(0.5, 0.025, "Unident. = unidentified cell", ha="center", fontsize=8, color="#666666")

    for ax, (status, motif_id, color) in zip(axes.flat, MOTIFS):
        graph = graphs[motif_id]
        names = nx.get_node_attributes(graph, "cell_type")
        positions = {node: POSITIONS[cell_type] for node, cell_type in names.items()}
        ax.set_axis_off()
        nx.draw_networkx_edges(graph, positions, ax=ax, edge_color=color, width=3.2, alpha=0.9)
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_size=2500,
            node_color="#f7f8f8",
            edgecolors=color,
            linewidths=2.3,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={node: NODE_LABELS[cell_type] for node, cell_type in names.items()},
            ax=ax,
            font_size=9,
            font_weight="medium",
            font_color="#25292c",
        )
        ax.set_title(
            f"{status.upper()}  |  Motif {motif_id}  |  {graph.number_of_edges()} edges",
            color=color,
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlim(-1.58, 1.58)
        ax.set_ylim(-1.78, 1.32)

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.91), h_pad=0.35, w_pad=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
