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
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


ROOT = _REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cism.cism import GetMotifsResult  # noqa: E402


BASE = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np_3node"
GROUP_COLORS = {"NN": "#3979a8", "NP": "#c85058"}
GROUP_FILLS = {"NN": "#e8f3f9", "NP": "#fbecee"}
DISPLAY_LABELS = {
    "DC": "DC",
    "b cell": "B cell",
    "blood vessels": "Vessels",
    "cd4 apc": "CD4 APC",
    "cd4 t cell": "CD4 T",
    "cd8 t cell": "CD8 T",
    "gc": "Germinal\ncenter",
    "hevs": "HEVs",
    "mac": "MAC",
    "memory%cd4 t cell": "Memory\nCD4 T",
    "neutrophil": "Neutrophil",
    "nk cell": "NK cell",
    "stroma": "Stroma",
    "treg": "CD4 Treg",
    "tumor": "Tumor",
    "unidentified": "Unident.",
}


def load_motifs(artifact_path: Path, n_jobs: int) -> tuple[list[dict], float, int]:
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    cism = artifact["cism"]
    if cism.motif_size != 3:
        raise ValueError(f"Expected 3-node CISM artifact, got motif size {cism.motif_size}")
    feature_conf = artifact["feature_conf"]
    discriminator = artifact["discriminator"]
    folds = discriminator.get_features(
        feature_conf=feature_conf, exclude_patients=[], n_jobs=n_jobs, prefer="threads"
    )
    result = GetMotifsResult(folds, discriminator, ["NN", "NP"])
    motifs_by_class = result.get_motifs_by_class()
    fold_counts = Counter(motif_id for features in folds.features for motif_id in set(features))
    lookup = cism.motifs_dataset.drop_duplicates("ID").set_index("ID")["motif"]
    type_names = {str(code): name for code, name in artifact["common_cells_type"].items()}

    rows = []
    for cohort in ("NN", "NP"):
        for motif_id in sorted(motifs_by_class[cohort], key=lambda value: (-fold_counts[value], value)):
            graph = lookup.loc[motif_id]
            if not isinstance(graph, nx.Graph):
                raise TypeError(f"Motif {motif_id} is not a graph")
            undirected = graph.to_undirected()
            if (graph.number_of_nodes() != 3 or undirected.number_of_edges() != 2
                    or any(not graph.has_edge(v, u) for u, v in graph.edges())):
                raise ValueError(f"Motif {motif_id} is not a reciprocal 3-node path")
            center = next(node for node, degree in undirected.degree() if degree == 2)
            leaves = sorted(
                (node for node in undirected if node != center),
                key=lambda node: (type_names.get(str(graph.nodes[node]["type"]), ""), node),
            )
            names = {
                node: type_names.get(str(graph.nodes[node]["type"]), str(graph.nodes[node]["type"]))
                for node in graph.nodes
            }
            rows.append(
                {
                    "patient_class": cohort,
                    "ID": int(motif_id),
                    "selected_folds": int(fold_counts[motif_id]),
                    "total_folds": len(folds),
                    "left_cell_type": names[leaves[0]],
                    "center_cell_type": names[center],
                    "right_cell_type": names[leaves[1]],
                    "graph": graph,
                    "ordered_nodes": (leaves[0], center, leaves[1]),
                }
            )
    return rows, float(feature_conf.shared_percentage), len(folds)


def draw_panel(ax: plt.Axes, row: dict) -> None:
    cohort = row["patient_class"]
    graph = row["graph"].to_undirected()
    nodes = row["ordered_nodes"]
    positions = {nodes[0]: (-1.1, 0), nodes[1]: (0, 0), nodes[2]: (1.1, 0)}
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#666b70", width=2.1)
    nx.draw_networkx_nodes(
        graph, positions, nodelist=list(nodes), ax=ax, node_size=2100,
        node_color=GROUP_FILLS[cohort], edgecolors=GROUP_COLORS[cohort], linewidths=2,
    )
    labels = {
        node: DISPLAY_LABELS.get(row[field], row[field])
        for node, field in zip(nodes, ("left_cell_type", "center_cell_type", "right_cell_type"))
    }
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=7.4,
                            font_color="#202327", font_weight="medium")
    ax.set_title(f"Motif {row['ID']}", color=GROUP_COLORS[cohort], fontsize=10,
                 fontweight="bold", pad=4)
    ax.text(0, -0.68, f"Selected in {row['selected_folds']}/{row['total_folds']} folds",
            ha="center", va="center", fontsize=8, color="#555b61")
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-0.85, 0.65)
    ax.set_aspect("equal")
    ax.axis("off")


def plot(rows: list[dict], stringency: float, output: Path) -> None:
    nn = [row for row in rows if row["patient_class"] == "NN"]
    np_rows = [row for row in rows if row["patient_class"] == "NP"]
    if len(nn) > 4 or len(np_rows) > 12:
        raise ValueError("The 4-column panel needs expansion for this motif set")
    fig, axes = plt.subplots(4, 4, figsize=(16, 11.4), dpi=220)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Discriminative 3-node anti-motifs at stringency {stringency:.3f}",
        fontsize=17, fontweight="bold", y=0.985,
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes[0], nn):
        draw_panel(ax, row)
    for ax, row in zip(axes[1:].flat, np_rows):
        draw_panel(ax, row)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.88, bottom=0.04,
                        hspace=0.72, wspace=0.12)
    fig.text(0.045, 0.917, f"NN-associated ({len(nn)})", color=GROUP_COLORS["NN"],
             fontsize=12, fontweight="bold")
    fig.text(0.045, 0.696, f"NP-associated ({len(np_rows)})", color=GROUP_COLORS["NP"],
             fontsize=12, fontweight="bold")
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the 3-node anti-motif LOOCV fold union.")
    parser.add_argument("--artifact", type=Path, default=BASE / "Melanoma_NN_NP_absent_cism_ready.pkl")
    parser.add_argument("--output-dir", type=Path, default=BASE / "figures")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, stringency, folds = load_motifs(args.artifact, args.n_jobs)
    if folds != 38 or len(rows) != 15:
        raise ValueError(f"Expected 15 motifs across 38 folds, got {len(rows)} across {folds}")
    table = pd.DataFrame([{key: value for key, value in row.items()
                           if key not in {"graph", "ordered_nodes"}} for row in rows])
    table.to_csv(args.output_dir / "optimal_3node_fold_union_motifs.csv", index=False)
    output = args.output_dir / "optimal_3node_fold_union_motif_graphs.png"
    plot(rows, stringency, output)
    print(output)
    print(output.with_suffix(".pdf"))
    print(args.output_dir / "optimal_3node_fold_union_motifs.csv")


if __name__ == "__main__":
    main()
