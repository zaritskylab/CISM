#!/usr/bin/env python3
"""Render CRC pairwise heatmaps in the layout used for the TNBC figure."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = _REPO_ROOT
SOURCE = ROOT / "analysis/outputs/crc_pairwise"
CELL_TYPES = [
    "Bcell", "CD3T", "CD4T", "CD8T", "DC", "Endothelial", "Fibroblast",
    "Lymphatic", "Macrophage", "Neuron", "Neutrophil", "Plasma", "SMV",
    "Stroma", "Treg", "Tumor", "Unidentified",
]
DISPLAY = {
    "Bcell": "B cells", "CD3T": "CD3 T cells", "CD4T": "CD4 T cells",
    "CD8T": "CD8 T cells",
}
ROWS = ("motif_induced", "all_pairwise")
GROUPS = ("CLR", "DII")


def load_matrix(kind: str, group: str) -> np.ndarray:
    path = SOURCE / f"{kind}_{group}.csv"
    frame = pd.read_csv(path, index_col=0).reindex(index=CELL_TYPES, columns=CELL_TYPES)
    matrix = frame.to_numpy(float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"Invalid probability matrix: {path}")
    return matrix


def draw_heatmap(ax, matrix: np.ndarray, group: str, vmax: float,
                 show_numbers: bool):
    lower = np.tril(np.ones_like(matrix, dtype=bool), -1)
    masked = np.ma.masked_where((matrix == 0) | lower, matrix)
    cmap = plt.get_cmap("Blues" if group == "CLR" else "Reds").copy()
    cmap.set_bad("white")
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax,
                      interpolation="none", aspect="equal")

    labels = [DISPLAY.get(name, name) for name in CELL_TYPES]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_numbers:
        for row, col in zip(*np.where((matrix > 0) & ~lower)):
            value = matrix[row, col]
            text = f"{value:.3f}" if value >= 0.0005 else "<.001"
            ax.text(col, row, text, ha="center", va="center", fontsize=4.5,
                    color="white" if value / vmax > 0.57 else "#444444")

    ax.text(0.03, 0.08, group, transform=ax.transAxes, fontsize=13,
            color="#2166AC" if group == "CLR" else "#B2182B",
            ha="left", va="bottom")
    return image


def render(show_numbers: bool) -> None:
    matrices = {(kind, group): load_matrix(kind, group)
                for kind in ROWS for group in GROUPS}
    suffix = "with_numbers" if show_numbers else "without_numbers"
    stem = SOURCE / f"crc_pairwise_heatmaps_tnbc_style_{suffix}"

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(12, 9.4))
        grid = fig.add_gridspec(
            2, 4, width_ratios=(1, 0.045, 1, 0.045),
            left=0.075, right=0.965, bottom=0.075, top=0.965,
            wspace=0.14, hspace=0.30,
        )

        for row_index, kind in enumerate(ROWS):
            vmax = max(matrices[(kind, group)].max() for group in GROUPS)
            for group_index, group in enumerate(GROUPS):
                heat_col = group_index * 2
                ax = fig.add_subplot(grid[row_index, heat_col])
                cax = fig.add_subplot(grid[row_index, heat_col + 1])
                image = draw_heatmap(ax, matrices[(kind, group)], group, vmax,
                                     show_numbers=show_numbers)
                colorbar = fig.colorbar(image, cax=cax)
                colorbar.ax.tick_params(labelsize=8, length=2)
                # Match the TNBC format: two scales, one written label per row.
                if group == "DII":
                    colorbar.set_label("Probability", fontsize=10, labelpad=9)

            fig.text(0.018, 0.965 if row_index == 0 else 0.485,
                     "A" if row_index == 0 else "B", fontsize=15,
                     ha="left", va="top")

        for extension in ("png", "pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{extension}"), dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {stem}.png/.pdf/.svg")


def main() -> None:
    render(show_numbers=True)
    render(show_numbers=False)


if __name__ == "__main__":
    main()
