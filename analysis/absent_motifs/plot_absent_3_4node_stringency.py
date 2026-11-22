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


ROOT = _REPO_ROOT
RUNTIME = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime"
DEFAULT_FOUR = RUNTIME / "absent_discriminative_cism_nn_np"
DEFAULT_THREE = RUNTIME / "absent_discriminative_cism_nn_np_3node"
COLORS = {3: "#399c34", 4: "#68449c"}
STYLES = {"NN": "solid", "NP": (0, (1.3, 2.1))}


def motif_count_curve(path: Path, size: int) -> pd.DataFrame:
    selected = pd.read_csv(path, usecols=["patient_class", "ID", "patient_percentage"])
    selected = selected[selected.patient_class.isin(STYLES)].copy()
    prevalence = (
        selected.groupby(["patient_class", "ID"], observed=True).patient_percentage
        .max()
        .reset_index()
    )
    thresholds = np.r_[0.0, np.sort(prevalence.patient_percentage.unique())]
    curve = pd.DataFrame({"stringency": thresholds})
    for cohort in STYLES:
        class_values = prevalence.loc[
            prevalence.patient_class.eq(cohort), "patient_percentage"
        ].to_numpy()
        curve[cohort] = [int(np.count_nonzero(class_values > x)) for x in thresholds]
    curve.insert(0, "motif_size", size)
    return curve


def plot_counts(curves: dict[int, pd.DataFrame], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for size in (4, 3):
        curve = curves[size]
        for cohort, style in STYLES.items():
            values = curve[cohort]
            anchors = curve.loc[values.gt(0) & values.ne(values.shift(fill_value=-1))]
            ax.plot(
                anchors.stringency,
                anchors[cohort],
                color=COLORS[size],
                linestyle=style,
                linewidth=2.2,
                label=f"{size} nodes  {cohort}",
            )
    ax.set(xlim=(0, 0.7), xlabel="Fraction of patients sharing the motif (stringency)",
           ylabel="Number of discriminative anti-motifs")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 10000)
    ax.grid(axis="y", which="major", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def plot_four_node_counts_paper(curve: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for cohort, style in (
        ("NN", (0, (6, 3))),
        ("NP", (0, (1, 2.2))),
    ):
        values = curve[cohort]
        anchors = curve.loc[values.gt(0) & values.ne(values.shift(fill_value=-1))]
        ax.plot(
            anchors.stringency, anchors[cohort],
            color=COLORS[4], linestyle=style, linewidth=2.6, label=cohort,
        )
    ax.set(
        xlim=(0, 0.5), ylim=(0.8, 10000),
        xlabel="Discrimination stringency",
        ylabel="Number of discriminative anti-motifs",
    )
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(axis="y", which="major", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def thin_trials(trials: pd.DataFrame, spacing: float = 0.02) -> pd.DataFrame:
    groups: list[list[pd.Series]] = []
    for _, row in trials.sort_values("stringency").iterrows():
        if groups and row.stringency - groups[-1][-1].stringency <= spacing:
            groups[-1].append(row)
        else:
            groups.append([row])
    return pd.DataFrame(
        [max(group, key=lambda row: (row.roc_auc, row.stringency)) for group in groups]
    ).sort_values("stringency")


def plot_auc(trials_by_size: dict[int, pd.DataFrame], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for size in (4, 3):
        trials = trials_by_size[size]
        displayed = thin_trials(trials.loc[trials.stringency.le(0.4)])
        if displayed.empty:
            raise ValueError(f"No {size}-node tuning trials at stringency <= 0.4")
        ax.plot(
            displayed.stringency,
            displayed.roc_auc,
            color=COLORS[size],
            marker="o",
            markersize=4.5,
            linewidth=2,
            label=f"{size}-node anti-motifs",
        )
    ax.set(xlim=(0, 0.4), ylim=(0.3, 0.85), xlabel="Discrimination stringency",
           ylabel="NN-NP LOOCV ROC AUC")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def plot_four_node_only(curve: pd.DataFrame, trials: pd.DataFrame, output: Path) -> None:
    fig, (counts_ax, auc_ax) = plt.subplots(1, 2, figsize=(14.8, 5.0))
    for cohort, style in STYLES.items():
        values = curve[cohort]
        anchors = curve.loc[values.gt(0) & values.ne(values.shift(fill_value=-1))]
        counts_ax.plot(
            anchors.stringency, anchors[cohort], color=COLORS[4], linestyle=style,
            linewidth=2.2, label=f"4 nodes  {cohort}",
        )
    counts_ax.set(
        xlim=(0, 0.7), ylim=(0.8, 10000),
        xlabel="Fraction of patients sharing the motif (stringency)",
        ylabel="Number of discriminative anti-motifs",
    )
    counts_ax.set_yscale("log")
    counts_ax.grid(axis="y", which="major", color="#e6e6e6", linewidth=0.7)
    counts_ax.set_axisbelow(True)
    counts_ax.spines[["top", "right"]].set_visible(False)
    counts_ax.legend(frameon=False, loc="upper right", fontsize=9)

    draw_four_node_auc(auc_ax, trials)
    fig.tight_layout(w_pad=3.2)
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def draw_four_node_auc(ax: plt.Axes, trials: pd.DataFrame) -> None:
    valid = trials.loc[
        trials.stringency.between(0.2, 0.5) & trials.roc_auc.gt(0)
    ]
    displayed = valid.sort_values("stringency")
    if displayed.empty:
        raise ValueError("No valid 4-node tuning trials between stringency 0.2 and 0.5")
    ax.plot(
        displayed.stringency, displayed.roc_auc, color=COLORS[4],
        marker="o", markersize=4.5, linewidth=2, label="4-node anti-motifs",
    )
    ax.set(
        xlim=(0.24, 0.42), ylim=(0.25, 0.66),
        xlabel="Discrimination stringency", ylabel="NN-NP LOOCV ROC AUC",
    )
    ax.set_xticks(np.arange(0.25, 0.401, 0.05))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")


def plot_four_node_auc_only(trials: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    draw_four_node_auc(ax, trials)
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare melanoma NN/NP 3- and 4-node anti-motif stringency.")
    parser.add_argument("--three-node-dir", type=Path, default=DEFAULT_THREE)
    parser.add_argument("--four-node-dir", type=Path, default=DEFAULT_FOUR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.three_node_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {3: args.three_node_dir, 4: args.four_node_dir}
    curves = {
        size: motif_count_curve(path / "absent_discriminative_motifs.csv", size)
        for size, path in paths.items()
    }
    pd.concat(curves.values(), ignore_index=True).to_csv(
        output_dir / "motif_counts_by_size_class_stringency.csv", index=False
    )
    trials = {
        size: pd.read_csv(path / "absent_stringency_optuna_trials.csv")
        .rename(columns={"params_shared_percentage": "stringency", "value": "roc_auc"})
        for size, path in paths.items()
    }
    for size, table in trials.items():
        if "max_class_features" in table.columns and not table.max_class_features.eq(30).all():
            raise ValueError(f"{size}-node trials did not all use max_class_features=30")
        if "params_max_class_features" in table.columns and not table.params_max_class_features.eq(30).all():
            raise ValueError(f"{size}-node trials did not all use max_class_features=30")
    plot_counts(curves, output_dir / "discriminative_motifs_vs_stringency_3_and_4_node.png")
    plot_auc(trials, output_dir / "auc_vs_stringency_3_and_4_node.png")
    four_node_output = args.four_node_dir / "figures/stringency_4node_only.png"
    four_node_output.parent.mkdir(parents=True, exist_ok=True)
    plot_four_node_only(curves[4], trials[4], four_node_output)
    four_node_auc_output = args.four_node_dir / "figures/auc_vs_stringency_4node_focused.png"
    plot_four_node_auc_only(trials[4], four_node_auc_output)
    four_node_counts_output = args.four_node_dir / "figures/discriminative_4node_antimotifs_vs_stringency_paper.png"
    plot_four_node_counts_paper(curves[4], four_node_counts_output)
    print(output_dir / "discriminative_motifs_vs_stringency_3_and_4_node.png")
    print(output_dir / "auc_vs_stringency_3_and_4_node.png")
    print(four_node_output)
    print(four_node_auc_output)
    print(four_node_counts_output)


if __name__ == "__main__":
    main()
