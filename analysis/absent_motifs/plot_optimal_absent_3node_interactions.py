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
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = _REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cism.cism import GetMotifsResult  # noqa: E402
from cism.helpers import string_base64_pickle  # noqa: E402


BASE = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np_3node"
COLORS = {"NN": "#3979a8", "NP": "#c85058"}
LABELS = {
    "DC": "DCs", "b cell": "B cell", "blood vessels": "Vessels",
    "cd4 apc": "CD4 APC", "cd4 t cell": "CD4 T cell", "cd8 t cell": "CD8 T cell",
    "gc": "Germinal center", "hevs": "HEVs", "mac": "MAC",
    "memory%cd4 t cell": "Memory CD4 T cell", "neutrophil": "Neutrophil",
    "nk cell": "NK cell", "stroma": "Stroma", "treg": "CD4 Treg",
    "tumor": "Tumor", "unidentified": "Unidentified",
}
PLOT_ORDER = [
    "b cell", "cd4 apc", "cd4 t cell", "treg", "cd8 t cell", "DC", "gc",
    "hevs", "mac", "memory%cd4 t cell", "nk cell", "neutrophil",
    "stroma", "unidentified", "blood vessels", "tumor",
]


def build_distributions(artifact: dict, n_jobs: int):
    cism = artifact["cism"]
    if cism.motif_size != 3:
        raise ValueError(f"Expected 3-node artifact, got {cism.motif_size}")
    table = cism.motifs_dataset
    if "p_value_direction" in table and not table.p_value_direction.eq("absence").all():
        raise ValueError("Motif table includes non-absence rows")
    conf = artifact["feature_conf"]
    discriminator = artifact["discriminator"]
    folds = discriminator.get_features(
        feature_conf=conf, exclude_patients=[], n_jobs=n_jobs, prefer="threads"
    )
    result = GetMotifsResult(folds, discriminator, ["NN", "NP"])
    by_class = result.get_motifs_by_class()
    weights = result.get_motifs_mean_weight()
    selected_folds = Counter(motif for features in folds.features for motif in set(features))
    names = {int(code): name for code, name in artifact["common_cells_type"].items()}
    cell_types = [name for name in LABELS if name in set(names.values())]
    if set(cell_types) != set(names.values()):
        raise ValueError("Cell type display order is incomplete")
    lookup = table.drop_duplicates("ID").set_index("ID")["motif"]
    rows = []
    cells = {}
    pairs = {}

    for cohort in ("NN", "NP"):
        node_counts = pd.Series(0.0, index=cell_types)
        edge_counts = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
        for motif_id in sorted(by_class[cohort]):
            graph = lookup.loc[motif_id]
            if isinstance(graph, str):
                graph = string_base64_pickle(graph)
            if not isinstance(graph, nx.Graph) or graph.number_of_nodes() != 3:
                raise ValueError(f"Invalid 3-node graph for motif {motif_id}")
            undirected = nx.Graph(graph)
            if undirected.number_of_edges() != 2:
                raise ValueError(f"Expected a 3-node path for motif {motif_id}")
            weight = float(weights.loc[motif_id])
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"Invalid frequency weight for motif {motif_id}: {weight}")
            for node in graph.nodes:
                node_counts[names[int(graph.nodes[node]["type"])]] += weight
            for left, right in undirected.edges:
                left_type = names[int(graph.nodes[left]["type"])]
                right_type = names[int(graph.nodes[right]["type"])]
                i, j = sorted((cell_types.index(left_type), cell_types.index(right_type)))
                edge_counts.iloc[i, j] += weight
            rows.append({
                "patient_class": cohort, "ID": int(motif_id),
                "selected_folds": int(selected_folds[motif_id]),
                "total_folds": int(len(folds)), "mean_frequency_weight": weight,
            })
        cells[cohort] = node_counts / node_counts.sum()
        # Each undirected edge is shown in both orientations without doubling its mass.
        symmetric_edges = (edge_counts + edge_counts.T) / 2
        pairs[cohort] = symmetric_edges / symmetric_edges.to_numpy().sum()
    return pd.DataFrame(rows), cells, pairs, float(conf.shared_percentage)


def plot_cells(cells: dict, counts: dict, stringency: float, output: Path) -> None:
    fractions = pd.DataFrame(cells)
    fractions = fractions.loc[fractions.sum(axis=1).sort_values(ascending=False).index]
    y = np.arange(len(fractions))
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.barh(y - 0.19, fractions.NN, height=0.36, color=COLORS["NN"],
            label=f"NN ({counts['NN']} motifs)")
    ax.barh(y + 0.19, fractions.NP, height=0.36, color=COLORS["NP"],
            label=f"NP ({counts['NP']} motifs)")
    ax.set_yticks(y, [LABELS[name] for name in fractions.index])
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of motif nodes (frequency-weighted)")
    ax.set_title(f"3-node anti-motif cell types | optimal stringency {stringency:.3f}")
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_pairs(pairs: dict, counts: dict, stringency: float, output: Path) -> None:
    active = [name for name in PLOT_ORDER if any(
        pairs[cohort].loc[name].sum() > 0 for cohort in ("NN", "NP")
    )]
    vmax = max(float(pairs[cohort].loc[active, active].to_numpy().max())
               for cohort in ("NN", "NP"))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.9), sharex=True, sharey=True)
    for ax, cohort in zip(axes, ("NN", "NP")):
        matrix = pairs[cohort].loc[active, active]
        cmap = "Blues" if cohort == "NN" else "Reds"
        image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, vmax=vmax,
                          interpolation="nearest", aspect="equal")
        labels = [LABELS[name] for name in matrix.index]
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.7)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.7)
        ax.tick_params(length=0)
        ax.set_title(f"{cohort} disease state ({counts[cohort]} motifs)",
                     color="#202020", fontsize=10, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.018, shrink=0.8,
                           aspect=35)
        bar.set_label("Probability", fontsize=8)
        bar.locator = MaxNLocator(nbins=5)
        bar.formatter = FormatStrFormatter("%.2f")
        bar.update_ticks()
        bar.ax.tick_params(labelsize=7, length=2)
        bar.outline.set_linewidth(0.5)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot optimal-stringency 3-node anti-motif interactions")
    parser.add_argument("--artifact", type=Path, default=BASE / "Melanoma_NN_NP_absent_cism_ready.pkl")
    parser.add_argument("--output-dir", type=Path, default=BASE / "figures/optimal_3node_interactions")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    motifs, cells, pairs, stringency = build_distributions(artifact, args.n_jobs)
    counts = motifs.patient_class.value_counts().to_dict()
    if len(motifs) != 15 or counts != {"NP": 12, "NN": 3}:
        raise ValueError(f"Unexpected fold union: {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motifs.to_csv(args.output_dir / "selected_motifs.csv", index=False)
    pd.DataFrame(cells).to_csv(args.output_dir / "cell_type_fractions.csv")
    for cohort in ("NN", "NP"):
        pairs[cohort].to_csv(args.output_dir / f"pairwise_edge_fractions_{cohort}.csv")
    plot_cells(cells, counts, stringency, args.output_dir / "cell_type_distribution.png")
    plot_pairs(pairs, counts, stringency, args.output_dir / "pairwise_interactions.png")
    summary = {
        "artifact": str(args.artifact), "stringency": stringency,
        "motif_count_by_class": counts,
        "motifs_selected_in_one_fold_by_class": {
            cohort: int(((motifs.patient_class == cohort) & (motifs.selected_folds == 1)).sum())
            for cohort in ("NN", "NP")
        },
        "weighting": "Each motif in the LOOCV fold union contributes once, weighted by its mean observed frequency across patients; nodes and undirected edges are normalized within class.",
        "pairwise_layout": "Symmetric full matrix; each undirected edge contributes half its probability to each off-diagonal orientation. All-zero cell types remain in the CSV but are omitted from the plot.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
