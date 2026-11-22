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
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
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
DEFAULT_PAIRWISE_DIR = DEFAULT_BASE_DIR / "motif_induced_pairwise_best_stringency"
DEFAULT_CISM_PATH = DEFAULT_BASE_DIR / "Melanoma_NN_NP_absent_cism_ready.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot optimal-stringency discriminative underrepresented motif subgraphs."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--pairwise-dir", type=Path, default=DEFAULT_PAIRWISE_DIR)
    parser.add_argument("--cism-path", type=Path, default=DEFAULT_CISM_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PAIRWISE_DIR / "optimal_stringency_discriminative_motif_subgraphs.png",
    )
    parser.add_argument(
        "--legend-output",
        type=Path,
        default=DEFAULT_PAIRWISE_DIR / "optimal_stringency_discriminative_motifs.csv",
    )
    args = parser.parse_args()

    selected = pd.read_csv(args.pairwise_dir / "stringency0_motifs_by_class.csv")
    selected = selected.sort_values(["patient_class", "ID"]).reset_index(drop=True)

    with args.cism_path.open("rb") as handle:
        payload = pickle.load(handle)
    cism = payload["cism"]
    cell_type_by_code = {str(code): label for code, label in payload["common_cells_type"].items()}
    motif_lookup = cism.motifs_dataset.drop_duplicates("ID").set_index("ID")["motif"]

    rows = []
    motifs = []
    for row in selected.itertuples(index=False):
        motif = deserialize_graph(motif_lookup.loc[row.ID])
        node_labels = {
            node: cell_type_by_code.get(str(data.get("type")), str(data.get("type")))
            for node, data in motif.nodes(data=True)
        }
        motifs.append((row.patient_class, int(row.ID), motif, node_labels))
        rows.append(
            {
                "patient_class": row.patient_class,
                "ID": int(row.ID),
                "node_cell_types": "; ".join(node_labels[node] for node in sorted(node_labels)),
                "n_nodes": motif.number_of_nodes(),
                "n_directed_edges": motif.number_of_edges(),
                "n_undirected_edges": motif.to_undirected().number_of_edges(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.legend_output, index=False)
    plot_motif_panel(motifs, args.output)
    print(args.output)
    print(args.legend_output)


def deserialize_graph(motif_obj):
    if isinstance(motif_obj, (nx.Graph, nx.DiGraph)):
        return motif_obj
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    raise TypeError(f"Unsupported motif graph representation: {type(motif_obj)!r}")


def plot_motif_panel(motifs: list[tuple[str, int, nx.Graph, dict]], output: Path) -> None:
    group_colors = {"NN": "#1f77b4", "NP": "#c9252d"}
    group_fills = {"NN": "#d6e9ff", "NP": "#ffd8d8"}
    n_cols = 5
    n_rows = -(-len(motifs) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 13.5), dpi=220)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    fig.suptitle(
        "Optimal-stringency discriminative underrepresented motifs",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        "Blue titles: NN-associated depleted motifs. Red titles: NP-associated depleted motifs.",
        ha="center",
        va="top",
        fontsize=9.5,
    )

    for ax, (patient_class, motif_id, motif, node_labels) in zip(axes, motifs):
        ax.set_axis_off()
        graph = motif.to_undirected()
        pos = nx.circular_layout(graph)
        color = group_colors.get(patient_class, "#333333")
        fill = group_fills.get(patient_class, "#eeeeee")
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#616161", width=1.4, alpha=0.9)
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=620,
            node_color=fill,
            edgecolors=color,
            linewidths=1.8,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={node: compact_label(label) for node, label in node_labels.items()},
            ax=ax,
            font_size=4.5,
            font_color="#171717",
        )
        ax.set_title(f"{patient_class}  |  {motif_id}", fontsize=6.6, color=color, pad=1.5)
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.42, 1.42)

    for ax in axes[len(motifs) :]:
        ax.set_axis_off()

    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.94), h_pad=0.9, w_pad=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def compact_label(label: str) -> str:
    replacements = {
        "blood vessels": "blood\nvessels",
        "memory%cd4 t cell": "memory\ncd4 T",
        "cd4 t cell": "cd4 T",
        "cd8 t cell": "cd8 T",
        "b cell": "B cell",
        "nk cell": "NK",
        "unidentified": "unident.",
    }
    text = replacements.get(label, label)
    if "\n" in text:
        return text
    return "\n".join(textwrap.wrap(text, width=9))


if __name__ == "__main__":
    main()
