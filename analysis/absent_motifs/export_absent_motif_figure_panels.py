from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter


BASE = (
    _REPO_ROOT
    / "melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np"
)
OPTIMUM = 0.3642694428844888
BLUE = "#3979a8"
RED = "#c85058"
PURPLE = "#654393"
LABELS = {
    "DC": "DCs",
    "b cell": "B cell",
    "blood vessels": "Vessels",
    "cd4 apc": "CD4 APC",
    "cd4 t cell": "CD4 T cell",
    "cd8 t cell": "CD8 T cell",
    "gc": "Germinal center cells",
    "hevs": "HEVs",
    "mac": "MAC",
    "memory%cd4 t cell": "Memory CD4 T cell",
    "neutrophil": "Neutrophil",
    "nk cell": "NK cell",
    "stroma": "Stroma",
    "treg": "CD4 Treg",
    "tumor": "Tumor",
    "unidentified": "Unidentified",
}


def finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def tidy(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#e7e7e7", linewidth=0.7)
        ax.set_axisbelow(True)


def plot_auc(base: Path, output: Path) -> None:
    trials = pd.read_csv(base / "absent_stringency_tradeoff.csv")
    trials = trials.loc[trials.stringency.le(0.4)].sort_values("stringency")
    # Retain one observed trial per nearby cluster; always show the chosen optimum.
    groups: list[list[pd.Series]] = []
    for _, row in trials.iterrows():
        if groups and row.stringency - groups[-1][-1].stringency <= 0.02:
            groups[-1].append(row)
        else:
            groups.append([row])
    selected = pd.DataFrame(
        [
            max(group, key=lambda row: (row.roc_auc, -abs(row.stringency - OPTIMUM)))
            for group in groups
        ]
    ).sort_values("stringency")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(selected.stringency, selected.roc_auc, color=PURPLE, linewidth=1.8)
    ax.scatter(selected.stringency, selected.roc_auc, color=PURPLE, s=27, zorder=3)
    ax.set(xlim=(0, 0.4), ylim=(0.35, 0.72), xlabel="Discrimination stringency",
           ylabel="LOOCV ROC AUC")
    tidy(ax)
    fig.tight_layout()
    finish(fig, output / "auc_vs_stringency.png")


def plot_counts(base: Path, output: Path) -> None:
    curve = pd.read_csv(base / "absent_4node_motif_count_by_class_vs_stringency.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for cohort, style in (("NN", "solid"), ("NP", (0, (1.2, 1.8)))):
        values = curve[f"{cohort}_motifs"]
        anchors = curve.loc[values.gt(0) & values.ne(values.shift(fill_value=-1))].copy()
        value = int(values.iloc[np.searchsorted(curve.stringency, OPTIMUM, side="right") - 1])
        anchors = pd.concat(
            [anchors, pd.DataFrame({"stringency": [OPTIMUM], f"{cohort}_motifs": [value]})],
            ignore_index=True,
        ).sort_values("stringency").drop_duplicates("stringency")
        ax.plot(anchors.stringency, anchors[f"{cohort}_motifs"], color=PURPLE,
                linewidth=2.3, linestyle=style, label=f"4 nodes: {cohort}")
        ax.scatter([OPTIMUM], [value], color=PURPLE, s=34, zorder=4)
        ax.annotate(f"{cohort}: {value}", (OPTIMUM, value),
                    xytext=(6, 9 if cohort == "NP" else -15), textcoords="offset points",
                    color=BLUE if cohort == "NN" else RED, fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set(xlim=(0, 0.5), ylim=(0.8, 10000),
           xlabel="Fraction of patients sharing the motif (stringency)",
           ylabel="Number of discriminative anti-motifs")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    tidy(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    finish(fig, output / "discriminative_4node_motifs_vs_stringency.png")


def plot_cell_types(motifs: pd.DataFrame, output: Path) -> None:
    counts: dict[str, pd.Series] = {}
    for cohort in ("NN", "NP"):
        rows = motifs.loc[motifs.patient_class.eq(cohort)]
        nodes = rows.node_cell_types.str.split("; ").explode().str.strip()
        if not rows.n_nodes.eq(4).all() or len(nodes) != 4 * len(rows):
            raise ValueError(f"Invalid 4-node motif composition for {cohort}")
        counts[cohort] = nodes.value_counts().div(len(nodes))
    fractions = pd.DataFrame(counts).fillna(0)
    fractions = fractions.loc[fractions.sum(axis=1).sort_values(ascending=False).index]
    y = np.arange(len(fractions))
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    ax.barh(y - 0.19, fractions.NN, height=0.36, color=BLUE, label="NN (19 motifs)")
    ax.barh(y + 0.19, fractions.NP, height=0.36, color=RED, label="NP (26 motifs)")
    ax.set_yticks(y, [LABELS.get(name, name) for name in fractions.index])
    ax.invert_yaxis()
    ax.set(xlabel="Fraction of motif nodes", xlim=(0, max(fractions.max()) * 1.1))
    ax.xaxis.grid(color="#e7e7e7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    finish(fig, output / "motif_induced_cell_type_distribution.png")


def plot_pairwise(base: Path, output: Path) -> None:
    source = base / "motif_induced_pairwise_best_stringency"
    matrices = {
        cohort: pd.read_csv(source / f"motif_induced_pairwise_{cohort}.csv", index_col=0)
        for cohort in ("NN", "NP")
    }
    if not matrices["NN"].index.equals(matrices["NP"].index) or not matrices["NN"].columns.equals(matrices["NP"].columns):
        raise ValueError("NN and NP pairwise matrices must share axes")
    vmax = max(float(matrix.to_numpy().max()) for matrix in matrices.values())
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 6.7), sharex=True, sharey=True)
    for ax, (cohort, color) in zip(axes, (("NN", BLUE), ("NP", RED))):
        matrix = matrices[cohort]
        cmap = LinearSegmentedColormap.from_list(f"{cohort}_white", ["#ffffff", color])
        image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, vmax=vmax,
                          interpolation="nearest", aspect="equal")
        names = [LABELS.get(name, name) for name in matrix.index]
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(names)), names, fontsize=7.5)
        ax.tick_params(length=0)
        ax.set_title(f"{cohort} disease state ({19 if cohort == 'NN' else 26} motifs)",
                     color=color, fontsize=11, fontweight="bold", pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.025)
        bar.ax.tick_params(labelsize=8)
        bar.set_label("Pairwise probability", fontsize=9)
    fig.tight_layout(w_pad=2.5)
    finish(fig, output / "motif_induced_pairwise_interactions.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export melanoma NN/NP anti-motif figure panels.")
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.base_dir / "figure_panels_fold_union_45"
    output.mkdir(parents=True, exist_ok=True)
    motifs = pd.read_csv(
        args.base_dir / "motif_induced_pairwise_best_stringency/optimal_stringency_discriminative_motifs.csv"
    )
    if motifs.groupby("patient_class").size().to_dict() != {"NN": 19, "NP": 26}:
        raise ValueError("Expected the 45-motif LOOCV fold union (19 NN, 26 NP)")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10})
    plot_auc(args.base_dir, output)
    plot_counts(args.base_dir, output)
    plot_cell_types(motifs, output)
    plot_pairwise(args.base_dir, output)


if __name__ == "__main__":
    main()
