#!/usr/bin/env python3
"""Summarize top-three motif appearances at each nearby run's optimum."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = _REPO_ROOT
RUNS = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/cism_label_permutation_test"
ARTIFACT = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/serialized/Melanoma_cism_ready.pkl"


def save_figure(fig, report: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(report / f"{name}.{extension}", dpi=250)
    plt.close(fig)


def style_axis(ax, summary: pd.DataFrame, xlabel: str) -> np.ndarray:
    y = np.arange(len(summary))
    ax.set_yticks(y, ["Original" if run == "original" else run.replace("_", " ").title()
                      for run in summary.run])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#e1e5e9", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.get_yticklabels()[0].set_color("#b54435")
    return y


def color_scale(summary: pd.DataFrame):
    shuffled = summary.loc[summary.run != "original", "n_fold_union_motifs"]
    norm = mcolors.Normalize(vmin=shuffled.min(), vmax=shuffled.max())
    cmap = plt.get_cmap("viridis")
    return cmap, norm


def add_colorbar(fig, ax, cmap, norm) -> None:
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.025)
    bar.set_label("Fold-union discriminative motifs")


def plot_variants(summary: pd.DataFrame, details: pd.DataFrame, report: Path, nearest: int) -> None:
    top3 = details.loc[details["rank"] <= 3]
    ranks = top3.pivot(index="run", columns="rank", values="mean_appearances_among_present_patients")
    ranks = ranks.reindex(summary.run)
    if ranks.isna().any().any():
        raise ValueError("Every run must have three selected motifs")
    cmap, norm = color_scale(summary)
    positions = np.arange(len(summary))
    offsets = np.array([-0.25, 0, 0.25])
    bar_height = 0.22
    xlabel = "Mean appearances per motif-positive patient"
    original_outline = "#b54435"

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.1, top=0.92)
    rank_colors = ["#376da8", "#4ca68d", "#e4a343"]
    for rank, offset, color in zip((1, 2, 3), offsets, rank_colors):
        ax.barh(positions + offset, ranks[rank], height=bar_height,
                color=color, edgecolor=[original_outline if run == "original" else color
                                      for run in summary.run],
                linewidth=[2.2 if run == "original" else 0.4 for run in summary.run])
    style_axis(ax, summary, xlabel)
    ax.set_title("Top three motif appearances at each run's optimum")
    ax.set_xlim(0, ranks.max().max() * 1.12)
    ax.legend(handles=[patches.Patch(facecolor=color, edgecolor="none", label=f"Motif {rank}")
                       for rank, color in zip((1, 2, 3), rank_colors)],
              ncol=3, frameon=False, loc="lower right")
    save_figure(fig, report, f"nearest_{nearest}_top3_appearance_individual_bars")

    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    sum_colors = [original_outline if run == "original" else cmap(norm(count))
                  for run, count in zip(summary.run, summary.n_fold_union_motifs)]
    ax.barh(positions, summary.top3_mean_appearances_sum, height=0.72, color=sum_colors)
    style_axis(ax, summary, "Sum of top 3 mean motif appearances per motif-positive patient")
    ax.set_title("Top-three motif appearance score at each run's optimum")
    ax.set_xlim(0, summary.top3_mean_appearances_sum.max() * 1.13)
    add_colorbar(fig, ax, cmap, norm)
    save_figure(fig, report, f"nearest_{nearest}_top3_appearance_sums_colored_by_motifs")

    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    hatches = ["", "///", "xx"]
    for rank, offset, hatch in zip((1, 2, 3), offsets, hatches):
        ax.barh(positions + offset, ranks[rank], height=bar_height,
                color=sum_colors, edgecolor="#333333", linewidth=0.5, hatch=hatch,
                label=f"Motif {rank}")
    style_axis(ax, summary, xlabel)
    ax.set_title("Top three motif appearances, colored by motif count")
    ax.set_xlim(0, ranks.max().max() * 1.12)
    ax.legend(handles=[patches.Patch(facecolor="white", edgecolor="#333333", hatch=hatch,
                                    label=f"Motif {rank}")
                       for rank, hatch in zip((1, 2, 3), hatches)],
              ncol=3, frameon=False, loc="lower right")
    add_colorbar(fig, ax, cmap, norm)
    save_figure(fig, report, f"nearest_{nearest}_top3_appearance_individual_bars_colored_by_motifs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--nearest", type=int, default=10)
    args = parser.parse_args()

    report = args.runs_dir / "report"
    sweeps = pd.read_csv(report / f"nearest_{args.nearest}_fold_union_motif_sweeps.csv")
    optimum = sweeps.loc[sweeps.at_saved_optimum].copy()
    nearest = pd.read_csv(report / f"nearest_{args.nearest}_optimal_stringency_shuffles.csv")
    run_order = ["original", *(f"seed_{int(seed)}" for seed in nearest.seed)]
    if set(optimum.run) != set(run_order) or optimum.run.duplicated().any():
        raise ValueError("Expected one saved optimum per run")

    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    discriminator = artifact["discriminator"]
    patients = set(discriminator.get_patients_class(["NN", "NP"]).index)
    selected = {
        run: {int(motif_id) for motif_id in ids.split(";") if motif_id}
        for run, ids in zip(optimum.run, optimum.selected_motif_ids)
    }
    all_ids = set().union(*selected.values())
    motifs = discriminator.cism.motifs_dataset
    relevant = motifs.loc[
        motifs.Patient_uId.isin(patients) & motifs.ID.isin(all_ids),
        ["ID", "Patient_uId", "Count"],
    ]
    patient_counts = relevant.groupby(["ID", "Patient_uId"], observed=True).Count.sum()
    motif_stats = patient_counts.groupby(level="ID").agg(
        mean_appearances="mean", patients_with_motif="size", total_appearances="sum"
    )

    detail_rows = []
    summary_rows = []
    for run in run_order:
        run_stats = motif_stats.loc[sorted(selected[run])].sort_values(
            ["mean_appearances", "patients_with_motif"], ascending=False
        )
        if len(run_stats) != len(selected[run]):
            raise ValueError(f"Some selected motifs have no observed instances: {run}")
        for rank, (motif_id, stats) in enumerate(run_stats.iterrows(), start=1):
            detail_rows.append({
                "run": run,
                "motif_id": int(motif_id),
                "rank": rank,
                "mean_appearances_among_present_patients": float(stats.mean_appearances),
                "patients_with_motif": int(stats.patients_with_motif),
                "total_appearances": float(stats.total_appearances),
            })
        top3 = run_stats.head(3)
        summary_rows.append({
            "run": run,
            "optimal_stringency": float(optimum.set_index("run").loc[run, "stringency"]),
            "n_fold_union_motifs": len(selected[run]),
            "top3_mean_appearances_sum": float(top3.mean_appearances.sum()),
            "top3_motif_ids": ";".join(str(motif_id) for motif_id in top3.index),
            "top3_mean_appearances": ";".join(f"{value:.6f}" for value in top3.mean_appearances),
        })

    details = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    details.to_csv(report / f"nearest_{args.nearest}_optimum_motif_appearances.csv", index=False)
    summary.to_csv(report / f"nearest_{args.nearest}_top3_appearance_sums.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    y = np.arange(len(summary))
    colors = ["#b54435" if run == "original" else "#e09b42" for run in summary.run]
    ax.barh(y, summary.top3_mean_appearances_sum, color=colors, height=0.72)
    ax.set_yticks(y, ["Original" if run == "original" else run.replace("_", " ").title()
                      for run in summary.run])
    ax.invert_yaxis()
    ax.set_xlabel("Sum of top 3 mean motif appearances per motif-positive patient")
    ax.set_title("Top-three motif appearance score at each run's optimum")
    ax.grid(axis="x", color="#e1e5e9", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for position, value in zip(y, summary.top3_mean_appearances_sum):
        ax.annotate(f"{value:,.1f}", (value, position), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlim(0, summary.top3_mean_appearances_sum.max() * 1.14)
    for extension in ("png", "pdf"):
        fig.savefig(report / f"nearest_{args.nearest}_top3_appearance_sums.{extension}", dpi=250)
    plt.close(fig)
    plot_variants(summary, details, report, args.nearest)
    print(summary[["run", "n_fold_union_motifs", "top3_mean_appearances_sum"]].to_string(index=False))
    print(f"Saved top-three results to {report}")


if __name__ == "__main__":
    main()
