#!/usr/bin/env python
"""Make a three-panel summary of the fixed-0.46 CISM label permutations."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.image import AxesImage
import numpy as np
import pandas as pd


ROOT = Path("melanoma_tutorials/melanoma_tutorials/runtime/cism_label_permutation_test")
COLORS = {"NN": "#3b78af", "NP": "#dc7b37"}


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trials = pd.read_csv(root / "aggregate/all_trial_summaries.csv")
    cells = pd.read_csv(root / "report/motif_induced_cell_type_permutation_ci.csv")
    pairs = pd.read_csv(root / "report/motif_induced_pairwise_permutation_ci.csv")
    cell_baseline = pd.read_csv(root / "original/from_artifact/general_cell_type_distribution.csv")
    pair_baseline = pd.read_csv(root / "original/from_artifact/general_pairwise_distribution.csv")
    cells = cells.loc[cells["variant"].eq("fixed") & ~cells["cell_type"].str.lower().eq("tumor")].copy()
    pairs = pairs.loc[pairs["variant"].eq("fixed")].copy()

    expected_n = int(trials["shuffle_labels"].astype(bool).sum())
    if cells["n"].nunique() != 1 or pairs["n"].nunique() != 1:
        raise ValueError("The distribution tables contain inconsistent shuffle counts")
    if int(cells["n"].iloc[0]) != expected_n or int(pairs["n"].iloc[0]) != expected_n:
        raise ValueError("The distribution tables do not match the AUC shuffle count")
    cells = cells.merge(
        cell_baseline[["patient_class", "cell_type", "proportion"]].rename(
            columns={"proportion": "original_general_proportion"}
        ),
        on=["patient_class", "cell_type"],
        how="left",
        validate="one_to_one",
    )
    if cells["original_general_proportion"].isna().any():
        raise ValueError("Original general cell-type baseline is incomplete")
    return trials, cells, make_pairwise_differences(pairs, pair_baseline)


def make_pairwise_differences(pairs: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.loc[
        ~pairs["source_cell_type"].str.lower().eq("tumor")
        & ~pairs["target_cell_type"].str.lower().eq("tumor")
    ].copy()
    baseline = baseline.loc[
        ~baseline["source_cell_type"].str.lower().eq("tumor")
        & ~baseline["target_cell_type"].str.lower().eq("tumor")
    ].copy()
    cell_types = sorted(set(pairs["source_cell_type"]) | set(pairs["target_cell_type"]))
    rows = []
    for patient_class in ("NN", "NP"):
        motif = pairs.loc[pairs["patient_class"].eq(patient_class)].pivot(
            index="source_cell_type", columns="target_cell_type", values="mean"
        ).reindex(index=cell_types, columns=cell_types, fill_value=0.0)
        general = baseline.loc[baseline["patient_class"].eq(patient_class)].pivot(
            index="source_cell_type", columns="target_cell_type", values="proportion"
        ).reindex(index=cell_types, columns=cell_types, fill_value=0.0)
        if motif.isna().any().any() or general.isna().any().any():
            raise ValueError(f"Incomplete pairwise matrix for {patient_class}")
        for i, source in enumerate(cell_types):
            for j in range(i, len(cell_types)):
                target = cell_types[j]
                motif_mean = float(motif.loc[source, target])
                general_value = float(general.loc[source, target])
                if i != j:
                    motif_mean += float(motif.loc[target, source])
                    general_value += float(general.loc[target, source])
                rows.append(
                    {
                        "patient_class": patient_class,
                        "source_cell_type": source,
                        "target_cell_type": target,
                        "mean_motif_induced_proportion": motif_mean,
                        "original_general_proportion": general_value,
                        "difference_percentage_points": 100.0 * (motif_mean - general_value),
                    }
                )
    return pd.DataFrame(rows)


def plot_auc(ax: plt.Axes, trials: pd.DataFrame) -> dict[str, float | int]:
    original = trials.loc[~trials["shuffle_labels"].astype(bool), "fixed_auc"]
    shuffled = trials.loc[trials["shuffle_labels"].astype(bool)].dropna(subset=["fixed_auc"])
    if len(original) != 1 or shuffled.empty:
        raise ValueError("Expected one original run and at least one shuffle")
    reference = float(original.iloc[0])
    skipped = shuffled["fixed_skipped_folds_no_motifs"]
    if skipped.isna().any():
        raise ValueError("Missing fixed-0.46 no-motif fold counts")
    has_empty_fold = skipped.gt(0)
    affected = int(has_empty_fold.sum())
    all_empty = int(skipped.eq(shuffled["n_loocv_folds"]).sum())
    ax.hist(
        [shuffled.loc[~has_empty_fold, "fixed_auc"], shuffled.loc[has_empty_fold, "fixed_auc"]],
        bins=np.linspace(0, 1, 21),
        stacked=True,
        color=["#5b84bd", "#dc7b37"],
        edgecolor="white",
        label=[f"No empty folds (n={len(shuffled) - affected})", f">=1 no-motif fold (n={affected})"],
    )
    ax.axvline(reference, color="#bd4234", lw=2.4, label=f"Original AUC = {reference:.3f}")
    ax.set(xlabel="AUC at stringency 0.46", ylabel="Number of shuffles", xlim=(0, 1))
    ax.set_title(f"A  Fixed 0.46 AUC: {affected}/{len(shuffled)} with no-motif folds", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis="y", alpha=0.25)
    return {
        "stringency": 0.46,
        "original_auc": reference,
        "n_shuffles": len(shuffled),
        "with_at_least_one_no_motif_fold": affected,
        "without_no_motif_folds": len(shuffled) - affected,
        "with_all_folds_no_motifs": all_empty,
    }


def plot_cells(ax: plt.Axes, cells: pd.DataFrame) -> None:
    classes = ("NN", "NP")
    pivot = cells.pivot(index="cell_type", columns="patient_class", values="mean")
    order = pivot.loc[:, list(classes)].mean(axis=1).sort_values(ascending=False).index
    x = np.arange(len(order))
    width = 0.38
    for offset, patient_class in zip((-0.5, 0.5), classes):
        class_cells = cells.loc[cells["patient_class"].eq(patient_class)].set_index("cell_type").reindex(order)
        positions = x + offset * width
        ax.bar(positions, class_cells["mean"].to_numpy() * 100.0, width=width,
               color=COLORS[patient_class], alpha=0.87,
               label=f"{patient_class} shuffled motif-induced mean")
        ax.scatter(positions, class_cells["original_general_proportion"].to_numpy() * 100.0,
                   marker="D", s=35, color=COLORS[patient_class], edgecolors="#263238",
                   linewidths=0.7, zorder=4, label=f"{patient_class} original tissue baseline")
    ax.set_xticks(x, order, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("C  Motif-induced cell types vs original tissue baseline", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_pairs(ax: plt.Axes, pairs: pd.DataFrame, patient_class: str, order: list[str], norm: TwoSlopeNorm) -> AxesImage:
    matrix = pairs.loc[pairs["patient_class"].eq(patient_class)].pivot(
        index="source_cell_type", columns="target_cell_type", values="difference_percentage_points"
    ).reindex(index=order, columns=order)
    masked = np.ma.masked_invalid(np.triu(matrix.to_numpy(dtype=float)))
    masked[np.tril_indices(len(order), -1)] = np.ma.masked
    image = ax.imshow(masked, cmap="RdBu_r", norm=norm, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(len(order)), order, rotation=55, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(order)), order, fontsize=6.5)
    ax.set_title(patient_class, fontsize=10, fontweight="bold")
    ax.tick_params(length=0)
    return image


def make_figure(trials: pd.DataFrame, cells: pd.DataFrame, pairs: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(16, 13.5), constrained_layout=True)
    layout = fig.add_gridspec(3, 2, height_ratios=(0.95, 0.10, 1.45), width_ratios=(1.0, 1.45))
    auc_ax = fig.add_subplot(layout[0, 0])
    cell_ax = fig.add_subplot(layout[0, 1])
    summary = plot_auc(auc_ax, trials)
    plot_cells(cell_ax, cells)

    heading = fig.add_subplot(layout[1, :])
    heading.axis("off")
    heading.text(0, 0.5, "D  Pairwise interactions vs original tissue baseline at fixed stringency 0.46",
                 ha="left", va="center", fontsize=13, fontweight="bold")
    lower = layout[2, :].subgridspec(1, 2, wspace=0.08)
    pair_axes = [fig.add_subplot(lower[0, i]) for i in range(2)]
    order = sorted(set(pairs["source_cell_type"]) | set(pairs["target_cell_type"]))
    bound = float(pairs["difference_percentage_points"].abs().max())
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    for ax, patient_class in zip(pair_axes, ("NN", "NP")):
        image = plot_pairs(ax, pairs, patient_class, order, norm)
    fig.colorbar(image, ax=pair_axes, location="bottom", shrink=0.5, pad=0.02,
                 label="Mean motif-induced minus original general (percentage points)")
    fig.supxlabel("Pairwise heatmaps show unordered cell-type pairs; tumor excluded", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fixed_046_three_panel_figure"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame([summary]).to_csv(output_dir / "fixed_046_three_panel_summary.csv", index=False)
    cells.to_csv(output_dir / "fixed_046_cell_type_mean_vs_original_general.csv", index=False)
    pairs.to_csv(output_dir / "fixed_046_pairwise_difference_unordered.csv", index=False)
    print(f"{stem}.png")
    print(f"{stem}.pdf")
    print(summary)


def load_reassessed_auc(path: Path) -> pd.DataFrame:
    trials = pd.read_csv(path)
    if len(trials) != 101 or trials.seed.notna().sum() != 100:
        raise ValueError("Expected one original run and 100 completed shuffles")
    if trials.seed.dropna().astype(int).tolist() != list(range(100)):
        raise ValueError("Expected unique seeds 0 through 99")
    return trials.rename(columns={
        "no_feature_folds": "fixed_skipped_folds_no_motifs",
        "n_patients": "n_loocv_folds",
        "fixed_auc": "fixed_auc",
    }).assign(shuffle_labels=trials.seed.notna())


def make_separate_panels(trials: pd.DataFrame, cells: pd.DataFrame,
                         pairs: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.3), layout="constrained")
    summary = plot_auc(ax, trials)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"panel_A_fixed_046_auc.{extension}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6.3), layout="constrained")
    plot_cells(ax, cells)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"panel_C_fixed_046_cell_types.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), layout="constrained")
    order = sorted(set(pairs["source_cell_type"]) | set(pairs["target_cell_type"]))
    bound = float(pairs["difference_percentage_points"].abs().max())
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    for ax, patient_class in zip(axes, ("NN", "NP")):
        image = plot_pairs(ax, pairs, patient_class, order, norm)
    fig.colorbar(image, ax=axes, location="bottom", shrink=0.5, pad=0.03,
                 label="Mean motif-induced minus original general (percentage points)")
    fig.suptitle("D  Pairwise interactions vs original tissue baseline at fixed stringency 0.46",
                 fontweight="bold")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"panel_D_fixed_046_pairwise.{extension}", dpi=300)
    plt.close(fig)

    pd.DataFrame([summary]).to_csv(output_dir / "panel_A_summary.csv", index=False)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--auc-reassessment", type=Path,
                        help="Completed all_fixed_046_aucs.csv from the CRC/TNBC-style reassessment")
    parser.add_argument("--separate-panels", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / "report"
    trials, cells, pairs = load_data(args.root)
    if args.auc_reassessment:
        trials = load_reassessed_auc(args.auc_reassessment)
    if args.separate_panels:
        make_separate_panels(trials, cells, pairs, output_dir)
    else:
        make_figure(trials, cells, pairs, output_dir)


if __name__ == "__main__":
    main()
