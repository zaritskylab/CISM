#!/usr/bin/env python3
"""Generate publication-ready figures for the imbalance bootstrap analysis."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

try:
    import seaborn as sns
except ImportError:
    sns = None


SCRIPT_DIR = ANALYSIS_DIR
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

PALETTE = {"NN": "#2F6C9E", "NP": "#D77A3D"}
CLASS_PALETTE = {"NN": "#2F6C9E", "NP": "#D77A3D"}


def configure_style() -> None:
    if sns is not None:
        sns.set_theme(style="white", context="paper")
    else:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(False)
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def line_with_ci(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    low: str | None,
    high: str | None,
    hue: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    for label, group in data.groupby(hue, observed=True):
        group = group.sort_values(x)
        color = PALETTE.get(label, None)
        ax.plot(group[x], group[y], marker="o", linewidth=1.8, label=f"sampled {label}", color=color)
        if low is not None and high is not None:
            ax.fill_between(group[x], group[low], group[high], alpha=0.18, color=color)
    ax.set_xlabel("Sampled-class fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(data[x].unique()))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=True)


def plot_auc_and_stringency(ci: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    line_with_ci(
        axes[0],
        ci,
        x="bootstrap_fraction",
        y="mean_auc",
        low="auc_sd_low",
        high="auc_sd_high",
        hue="sampled_label",
        ylabel="RF LOOCV AUC",
        title="Classification performance (mean +/- SD)",
        ylim=(0, 1.02),
    )
    line_with_ci(
        axes[1],
        ci,
        x="bootstrap_fraction",
        y="mean_optimal_stringency",
        low=None,
        high=None,
        hue="sampled_label",
        ylabel="Mean optimized stringency",
        title="Stringency selected by Optuna",
        ylim=(0, 1.02),
    )
    fig.tight_layout()
    save_figure(fig, "imbalance_auc_and_stringency")


def plot_motif_pool(results: pd.DataFrame) -> None:
    summary = (
        results.groupby(["sampled_label", "bootstrap_fraction"], as_index=False, observed=True)
        .agg(
            raw_NN=("n_discriminative_motifs_NN", "mean"),
            raw_NP=("n_discriminative_motifs_NP", "mean"),
            pass_NN=("optimal_full_bootstrap_discriminative_motifs_NN", "mean"),
            pass_NP=("optimal_full_bootstrap_discriminative_motifs_NP", "mean"),
        )
        .melt(
            id_vars=["sampled_label", "bootstrap_fraction"],
            var_name="metric",
            value_name="motif_count",
        )
    )
    summary["threshold"] = np.where(summary["metric"].str.startswith("raw"), "Hard-discriminative", "Pass optimized stringency")
    summary["motif_label"] = summary["metric"].str.extract(r"_(NN|NP)$")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True, sharey="row")
    for row_idx, threshold in enumerate(["Hard-discriminative", "Pass optimized stringency"]):
        for col_idx, sampled_label in enumerate(["NN", "NP"]):
            ax = axes[row_idx, col_idx]
            subset = summary[(summary["threshold"] == threshold) & (summary["sampled_label"] == sampled_label)]
            for motif_label, group in subset.groupby("motif_label", observed=True):
                group = group.sort_values("bootstrap_fraction")
                ax.plot(
                    group["bootstrap_fraction"],
                    group["motif_count"],
                    marker="o",
                    linewidth=1.7,
                    label=f"{motif_label}-associated",
                    color=CLASS_PALETTE.get(motif_label),
                )
            ax.set_yscale("log")
            ax.set_title(f"{threshold}; sampled {sampled_label}")
            ax.set_xlabel("Sampled-class fraction")
            ax.set_ylabel("Mean motif count")
            ax.set_xticks(sorted(results["bootstrap_fraction"].unique()))
            ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, "imbalance_motif_pool_by_class")


def plot_original_reuse(usage: pd.DataFrame) -> None:
    summary = (
        usage.groupby(["sampled_label", "bootstrap_fraction"], as_index=False, observed=True)
        .agg(
            fraction_original_NN_motifs_used=("fraction_original_NN_motifs_used", "mean"),
            fraction_original_NP_motifs_used=("fraction_original_NP_motifs_used", "mean"),
            fraction_all_original_motifs_used=("fraction_all_original_motifs_used", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    panels = [
        ("fraction_original_NN_motifs_used", "Original NN-associated motifs"),
        ("fraction_original_NP_motifs_used", "Original NP-associated motifs"),
        ("fraction_all_original_motifs_used", "All original motifs"),
    ]
    for ax, (column, title) in zip(axes, panels):
        for sampled_label, group in summary.groupby("sampled_label", observed=True):
            group = group.sort_values("bootstrap_fraction")
            ax.plot(
                group["bootstrap_fraction"],
                group[column],
                marker="o",
                linewidth=1.8,
                label=f"sampled {sampled_label}",
                color=PALETTE.get(sampled_label),
            )
        ax.set_title(title)
        ax.set_xlabel("Sampled-class fraction")
        ax.set_ylabel("Mean fraction reused")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(sorted(summary["bootstrap_fraction"].unique()))
        ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, "imbalance_original_motif_reuse")


def plot_motifs_vs_stringency(motifs_vs_stringency: pd.DataFrame) -> None:
    summary = (
        motifs_vs_stringency.groupby(
            ["sampled_label", "bootstrap_fraction", "stringency", "motif_label"],
            as_index=False,
            observed=True,
        )
        .agg(mean_passing=("discriminative_motifs_passing", "mean"))
    )
    for sampled_label in ["NN", "NP"]:
        subset = summary[summary["sampled_label"] == sampled_label]
        fractions = sorted(subset["bootstrap_fraction"].unique())
        cmap = plt.cm.viridis
        line_styles = {"NN": "--", "NP": ":"}

        fig, ax = plt.subplots(figsize=(7.4, 5.1))
        fraction_handles = []
        fraction_labels = []
        style_handles = []
        style_labels = []

        for idx, fraction in enumerate(fractions):
            color = cmap(idx / max(1, len(fractions) - 1))
            for motif_label in ["NN", "NP"]:
                group = subset[
                    (subset["bootstrap_fraction"] == fraction)
                    & (subset["motif_label"] == motif_label)
                ].sort_values("stringency")
                if group.empty:
                    continue
                ax.plot(
                    group["stringency"],
                    group["mean_passing"],
                    linestyle=line_styles[motif_label],
                    linewidth=1.5,
                    color=color,
                )
            fraction_handles.append(plt.Line2D([0], [0], color=color, linewidth=1.8))
            fraction_labels.append(f"{fraction:.1f}")

        for motif_label in ["NN", "NP"]:
            style_handles.append(
                plt.Line2D([0], [0], color="#333333", linestyle=line_styles[motif_label], linewidth=1.8)
            )
            style_labels.append(f"{motif_label}-associated")

        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(f"Motif counts vs stringency when sampling {sampled_label}")
        ax.set_xlabel("Stringency")
        ax.set_ylabel("Mean motifs passing")
        ax.set_xticks(sorted(summary["stringency"].unique()))
        ax.set_xlim(0, 1.0)

        first_legend = ax.legend(
            fraction_handles,
            fraction_labels,
            title="Sampled fraction",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=True,
        )
        ax.add_artist(first_legend)
        ax.legend(
            style_handles,
            style_labels,
            title="Motif class",
            bbox_to_anchor=(1.02, 0.34),
            loc="upper left",
            frameon=True,
        )
        fig.tight_layout()
        save_figure(fig, f"imbalance_motifs_vs_stringency_sampled_{sampled_label}")


def plot_aggregated_motifs_vs_stringency(motifs_vs_stringency: pd.DataFrame) -> None:
    summary = (
        motifs_vs_stringency.groupby(
            ["sampled_label", "bootstrap_fraction", "stringency", "motif_label"],
            as_index=False,
            observed=True,
        )
        .agg(mean_passing=("discriminative_motifs_passing", "mean"))
    )

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.4), sharex=True, sharey=True)
    fractions = sorted(summary["bootstrap_fraction"].unique())
    line_styles = {"NN": "--", "NP": ":"}

    for ax, fraction in zip(axes.ravel(), fractions):
        subset = summary[summary["bootstrap_fraction"] == fraction]
        for sampled_label in ["NN", "NP"]:
            for motif_label in ["NN", "NP"]:
                group = subset[
                    (subset["sampled_label"] == sampled_label)
                    & (subset["motif_label"] == motif_label)
                ].sort_values("stringency")
                if group.empty:
                    continue
                ax.plot(
                    group["stringency"],
                    group["mean_passing"].clip(lower=1),
                    color=PALETTE.get(sampled_label),
                    linestyle=line_styles[motif_label],
                    linewidth=1.9,
                    label=f"sampled {sampled_label}, {motif_label} motifs",
                )
        ax.set_title(f"Sampled fraction {fraction:.1f}")
        ax.set_yscale("log")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(bottom=1)
        ax.set_xticks(sorted(summary["stringency"].unique()))

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean # discriminative motifs")
    for ax in axes[-1, :]:
        ax.set_xlabel("Stringency")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Motif counts vs stringency by sampled class and motif class", y=1.07)
    fig.tight_layout()
    save_figure(fig, "imbalance_motifs_vs_stringency_aggregated")


def plot_motif_contrast_heatmaps(motifs_vs_stringency: pd.DataFrame) -> None:
    summary = (
        motifs_vs_stringency.groupby(
            ["sampled_label", "bootstrap_fraction", "stringency", "motif_label"],
            as_index=False,
            observed=True,
        )
        .agg(mean_passing=("discriminative_motifs_passing", "mean"))
    )

    wide = summary.pivot_table(
        index=["sampled_label", "bootstrap_fraction", "stringency"],
        columns="motif_label",
        values="mean_passing",
        aggfunc="first",
    ).reset_index()

    wide["np_minus_nn"] = wide["NP"] - wide["NN"]
    wide["signed_log_np_minus_nn"] = (
        np.sign(wide["np_minus_nn"]) * np.log10(np.abs(wide["np_minus_nn"]) + 1)
    )
    wide["np_over_nn"] = (wide["NP"] + 1) / (wide["NN"] + 1)
    wide["log10_np_over_nn"] = np.log10(wide["np_over_nn"])
    finite_values = wide["signed_log_np_minus_nn"].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    max_abs = np.nanmax(np.abs(finite_values)) if finite_values.size else 1
    if max_abs == 0:
        max_abs = 1

    for sampled_label in ["NN", "NP"]:
        subset = wide[wide["sampled_label"] == sampled_label]
        matrix = subset.pivot(
            index="bootstrap_fraction",
            columns="stringency",
            values="signed_log_np_minus_nn",
        ).sort_index(ascending=True)
        values = matrix.to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
        )
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels([f"{value:.1f}" for value in matrix.columns])
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels([f"{value:.1f}" for value in matrix.index])
        ax.set_xlabel("Stringency")
        ax.set_ylabel("Sampling rate")
        ax.set_title(
            f"NP-associated minus NN-associated discriminative motifs\n"
            f"Sampling {sampled_label}; signed log scale"
        )
        cbar = fig.colorbar(image, ax=ax)
        raw_difference_ticks = np.array(
            [-10000, -1000, -100, -10, 0, 10, 100, 1000, 10000]
        )
        transformed_ticks = (
            np.sign(raw_difference_ticks) * np.log10(np.abs(raw_difference_ticks) + 1)
        )
        cbar.set_ticks(transformed_ticks)
        cbar.set_ticklabels([str(value) for value in raw_difference_ticks])
        cbar.set_label("Mean discriminative motif difference (NP - NN)")
        fig.tight_layout()
        save_figure(fig, f"imbalance_NP_minus_NN_heatmap_sampled_{sampled_label}_signed_log")

    ratio_values = wide["log10_np_over_nn"].to_numpy(dtype=float)
    ratio_values = ratio_values[np.isfinite(ratio_values)]
    max_abs_ratio = np.nanmax(np.abs(ratio_values)) if ratio_values.size else 1
    if max_abs_ratio == 0:
        max_abs_ratio = 1

    for sampled_label in ["NN", "NP"]:
        subset = wide[wide["sampled_label"] == sampled_label]
        matrix = subset.pivot(
            index="bootstrap_fraction",
            columns="stringency",
            values="log10_np_over_nn",
        ).sort_index(ascending=True)

        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-max_abs_ratio, vcenter=0, vmax=max_abs_ratio),
        )
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels([f"{value:.1f}" for value in matrix.columns])
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels([f"{value:.1f}" for value in matrix.index])
        ax.set_xlabel("Stringency")
        ax.set_ylabel("Sampling rate")
        ax.set_title(
            f"Log ratio of NP- to NN-associated discriminative motifs\n"
            f"Sampling {sampled_label}"
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("log10((NP + 1) / (NN + 1))")
        fig.tight_layout()
        save_figure(fig, f"imbalance_NP_over_NN_heatmap_sampled_{sampled_label}_log10")

    ratio_values = wide["np_over_nn"].to_numpy(dtype=float)
    ratio_values = ratio_values[np.isfinite(ratio_values)]
    max_ratio = np.nanmax(ratio_values) if ratio_values.size else 1.0001
    max_ratio = max(max_ratio, 1.0001)

    for sampled_label in ["NN", "NP"]:
        subset = wide[wide["sampled_label"] == sampled_label]
        matrix = subset.pivot(
            index="bootstrap_fraction",
            columns="stringency",
            values="np_over_nn",
        ).sort_index(ascending=True)

        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=max_ratio),
        )
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels([f"{value:.1f}" for value in matrix.columns])
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels([f"{value:.1f}" for value in matrix.index])
        ax.set_xlabel("Stringency")
        ax.set_ylabel("Sampling rate")
        ax.set_title(
            f"Ratio of NP- to NN-associated discriminative motifs\n"
            f"Sampling {sampled_label}; normal scale"
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("(NP + 1) / (NN + 1)")
        fig.tight_layout()
        save_figure(fig, f"imbalance_NP_over_NN_heatmap_sampled_{sampled_label}_normal")


def plot_summary_panel(ci: pd.DataFrame, results: pd.DataFrame, usage: pd.DataFrame) -> None:
    motif_summary = (
        results.groupby(["sampled_label", "bootstrap_fraction"], as_index=False, observed=True)
        .agg(
            raw_total=("n_discriminative_motifs", "mean"),
            passing_total=("optimal_full_bootstrap_discriminative_motifs", "mean"),
        )
    )
    usage_summary = (
        usage.groupby(["sampled_label", "bootstrap_fraction"], as_index=False, observed=True)
        .agg(reused_all=("fraction_all_original_motifs_used", "mean"))
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.0), sharex=True)
    line_with_ci(
        axes[0, 0],
        ci,
        x="bootstrap_fraction",
        y="mean_auc",
        low="auc_sd_low",
        high="auc_sd_high",
        hue="sampled_label",
        ylabel="RF LOOCV AUC",
        title="AUC (mean +/- SD)",
        ylim=(0, 1.02),
    )
    line_with_ci(
        axes[0, 1],
        ci,
        x="bootstrap_fraction",
        y="mean_optimal_stringency",
        low=None,
        high=None,
        hue="sampled_label",
        ylabel="Mean optimized stringency",
        title="Stringency",
        ylim=(0, 1.02),
    )
    for sampled_label, group in motif_summary.groupby("sampled_label", observed=True):
        group = group.sort_values("bootstrap_fraction")
        axes[1, 0].plot(group["bootstrap_fraction"], group["raw_total"], marker="o", label=f"raw, sampled {sampled_label}", color=PALETTE.get(sampled_label), linestyle="-")
        axes[1, 0].plot(group["bootstrap_fraction"], group["passing_total"], marker="s", label=f"passing, sampled {sampled_label}", color=PALETTE.get(sampled_label), linestyle="--")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Sampled-class fraction")
    axes[1, 0].set_ylabel("Mean motif count")
    axes[1, 0].set_title("Motif pool size")
    axes[1, 0].legend(frameon=True)

    for sampled_label, group in usage_summary.groupby("sampled_label", observed=True):
        group = group.sort_values("bootstrap_fraction")
        axes[1, 1].plot(group["bootstrap_fraction"], group["reused_all"], marker="o", label=f"sampled {sampled_label}", color=PALETTE.get(sampled_label))
    axes[1, 1].set_xlabel("Sampled-class fraction")
    axes[1, 1].set_ylabel("Fraction reused")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set_title("Original motif reuse")
    axes[1, 1].legend(frameon=True)

    for ax in axes.ravel():
        ax.set_xticks(sorted(ci["bootstrap_fraction"].unique()))
    fig.tight_layout()
    save_figure(fig, "imbalance_summary_four_panel")


def main() -> None:
    configure_style()
    ci = pd.read_csv(OUTPUT_DIR / "imbalanced_bootstrap_auc_ci.csv")
    results = pd.read_csv(OUTPUT_DIR / "imbalanced_bootstrap_results.csv")
    usage = pd.read_csv(OUTPUT_DIR / "bootstrap_original_fold_union_motif_usage.csv")
    motifs_vs_stringency = pd.read_csv(OUTPUT_DIR / "discriminative_motifs_vs_stringency_by_class.csv")

    auc_std = (
        results.groupby(["sampled_label", "bootstrap_fraction"], as_index=False, observed=True)
        .agg(auc_std=("optimal_auc", "std"))
    )
    ci = ci.merge(auc_std, on=["sampled_label", "bootstrap_fraction"], how="left")
    ci["auc_sd_low"] = (ci["mean_auc"] - ci["auc_std"]).clip(lower=0, upper=1)
    ci["auc_sd_high"] = (ci["mean_auc"] + ci["auc_std"]).clip(lower=0, upper=1)

    plot_auc_and_stringency(ci)
    plot_motif_pool(results)
    plot_original_reuse(usage)
    plot_motifs_vs_stringency(motifs_vs_stringency)
    plot_aggregated_motifs_vs_stringency(motifs_vs_stringency)
    plot_motif_contrast_heatmaps(motifs_vs_stringency)
    plot_summary_panel(ci, results, usage)


if __name__ == "__main__":
    main()
