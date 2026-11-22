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
BASE = (
    ROOT / "melanoma_tutorials/melanoma_tutorials/runtime"
    / "absent_3node_patient_permutation_control/aggregate"
)
BLUE = "#7d9fce"
GREEN = "#8e9f78"
RED = "#d84b48"


def load_trials(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    trials = pd.read_csv(path)
    required = {"shuffle_labels", "seed", "optimal_auc", "optimal_stringency", "fixed_stringency"}
    if missing := required.difference(trials.columns):
        raise ValueError(f"Missing trial columns: {sorted(missing)}")
    original = trials.loc[~trials.shuffle_labels]
    shuffled = trials.loc[trials.shuffle_labels].copy()
    if len(original) != 1 or len(shuffled) != 100 or shuffled.seed.nunique() != 100:
        raise ValueError("Expected one original result and 100 unique shuffled seeds")
    if not np.isclose(original.iloc[0].optimal_stringency, original.iloc[0].fixed_stringency):
        raise ValueError("Original optimal stringency differs from the fixed comparator")
    for column in ("optimal_auc", "optimal_stringency"):
        if not np.isfinite(trials[column]).all():
            raise ValueError(f"Non-finite values in {column}")
    return original.iloc[0], shuffled


def style_axes(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#51575b")
    ax.tick_params(labelsize=9, color="#51575b", length=4)
    if grid:
        ax.grid(color="#e9ecee", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)


def draw_auc_hist(ax: plt.Axes, original: pd.Series, shuffled: pd.DataFrame) -> None:
    bins = np.arange(0.4, 0.951, 0.025)
    ax.hist(shuffled.optimal_auc, bins=bins, color=BLUE, edgecolor="white", linewidth=0.7)
    ax.axvline(original.optimal_auc, color=RED, linewidth=1.8,
               label=f"Original = {original.optimal_auc:.3f}")
    ax.set(xlim=(0.4, 0.95), xlabel="AUC", ylabel="Number of permutations")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.42, 1.0))
    style_axes(ax)


def draw_stringency_hist(ax: plt.Axes, original: pd.Series, shuffled: pd.DataFrame) -> None:
    bins = np.arange(0, 0.551, 0.025)
    ax.hist(shuffled.optimal_stringency, bins=bins, color=GREEN,
            edgecolor="white", linewidth=0.7)
    ax.axvline(original.optimal_stringency, color=RED, linewidth=1.8,
               label=f"Original = {original.optimal_stringency:.3f}")
    ax.set(xlim=(0, 0.55), xlabel="Optimal stringency", ylabel="Number of permutations")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axes(ax)


def draw_scatter(ax: plt.Axes, original: pd.Series, shuffled: pd.DataFrame) -> None:
    ax.scatter(shuffled.optimal_stringency, shuffled.optimal_auc, s=30,
               color=BLUE, alpha=0.87, edgecolors="white", linewidths=0.4,
               label="Shuffled permutations", zorder=2)
    ax.scatter([original.optimal_stringency], [original.optimal_auc], s=78,
               marker="D", color=RED, edgecolors="white", linewidths=0.7,
               label="Original", zorder=4)
    ax.set(xlim=(-0.015, 0.55), ylim=(0.35, 0.95),
           xlabel="Optimal stringency", ylabel="Optimal AUC")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_axes(ax, grid=True)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=240, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 3-node melanoma patient-label permutation control.")
    parser.add_argument("--trials", type=Path, default=BASE / "all_trial_summaries.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE)
    args = parser.parse_args()
    original, shuffled = load_trials(args.trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        (draw_auc_hist, "auc_permutation_histogram_blue"),
        (draw_stringency_hist, "optimal_stringency_permutation_histogram_green"),
        (draw_scatter, "optimal_stringency_vs_auc_permutation_scatter"),
    ]
    for draw, name in panels:
        fig, ax = plt.subplots(figsize=(5.3, 4.0))
        draw(ax, original, shuffled)
        save(fig, args.output_dir / f"{name}.png")

    fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.0))
    for ax, (draw, _name) in zip(axes, panels):
        draw(ax, original, shuffled)
    save(fig, args.output_dir / "three_node_permutation_control_three_panels.png")
    print(args.output_dir / "three_node_permutation_control_three_panels.png")


if __name__ == "__main__":
    main()
