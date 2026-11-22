from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
BAR_COLOR = "#216785"
TEXT_COLOR = "#151D2D"
GRID_COLOR = "#D8E1EB"

METRICS = {
    "Accuracy": 0.7450980392156863,
    "Balanced accuracy": 0.7292022792022793,
    "ROC-AUC": 0.8146921084951645,
}
BASELINES = {
    "Accuracy": 20 / 51,
    "Balanced accuracy": 1 / 3,
    "ROC-AUC": 0.5,
}
CONFUSION = np.array(
    [
        [17, 3, 0],
        [4, 13, 1],
        [2, 3, 8],
    ]
)
LABELS = ["NN", "NP", "PN"]


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8FA1B5")
    ax.spines["bottom"].set_color("#8FA1B5")
    ax.tick_params(colors=TEXT_COLOR, length=0, pad=9)


def draw_performance(ax: plt.Axes) -> None:
    names = list(METRICS)
    values = list(METRICS.values())
    x = np.arange(len(names))
    bars = ax.bar(x, values, width=0.59, color=BAR_COLOR, zorder=3)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    style_axes(ax)

    for position, name in zip(x, names):
        ax.hlines(
            BASELINES[name],
            position - 0.43,
            position + 0.43,
            color="#3F4752",
            linewidth=1.5,
            linestyle=(0, (5, 4)),
            zorder=5,
        )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontweight="bold",
        )

    ax.legend(
        handles=[
            Patch(facecolor=BAR_COLOR, edgecolor="none"),
            Line2D([0], [0], color="#3F4752", linewidth=1.5, linestyle=(0, (5, 4))),
        ],
        labels=[
            "One-vs-one + third-class scoring · 3 confidence scores",
            "Baseline",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.035),
        frameon=False,
        handlelength=0.65,
        handletextpad=0.45,
        ncol=2,
        columnspacing=1.6,
    )


def draw_confusion(ax: plt.Axes, fig: plt.Figure) -> None:
    image = ax.imshow(CONFUSION, cmap="Blues", vmin=0, vmax=17)
    ax.set_xticks(np.arange(len(LABELS)), labels=LABELS)
    ax.set_yticks(np.arange(len(LABELS)), labels=LABELS)
    ax.set_xlabel("Predicted", color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("True", color=TEXT_COLOR, labelpad=8)
    ax.tick_params(colors=TEXT_COLOR, length=0, pad=7)

    for row in range(CONFUSION.shape[0]):
        for column in range(CONFUSION.shape[1]):
            value = CONFUSION[row, column]
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > 8 else TEXT_COLOR,
                fontweight="bold" if row == column else "normal",
            )

    for spine in ax.spines.values():
        spine.set_color("#556170")
        spine.set_linewidth(1.0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.045)
    colorbar.set_ticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    colorbar.ax.tick_params(colors=TEXT_COLOR, length=3)
    colorbar.outline.set_edgecolor("#556170")


def save_panel_b() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    draw_performance(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.80)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"panel_b_one_vs_one_third_class_scoring.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_panel_c() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor="white")
    draw_confusion(ax, fig)
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.15, top=0.96)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"panel_c_one_vs_one_third_class_scoring.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined() -> None:
    fig = plt.figure(figsize=(12.4, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1), wspace=0.23)
    ax_b = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    draw_performance(ax_b)
    draw_confusion(ax_c, fig)
    fig.text(0.012, 0.975, "(B)", fontsize=17, fontweight="bold", color=TEXT_COLOR, va="top")
    fig.text(0.542, 0.975, "(C)", fontsize=17, fontweight="bold", color=TEXT_COLOR, va="top")
    fig.subplots_adjust(left=0.045, right=0.96, bottom=0.14, top=0.83)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"panels_bc_one_vs_one_third_class_scoring.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "savefig.facecolor": "white",
        }
    )
    save_panel_b()
    save_panel_c()
    save_combined()


if __name__ == "__main__":
    main()
