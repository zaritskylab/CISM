import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from figures.helpers.style import CLASS_CMAPS

CELL_SIZE = 0.45  # inches per matrix cell, room for a value at VALUE_FONTSIZE
VALUE_FONTSIZE = 7
# inches each row of plot_class_heatmap_rows keeps for its rotated cell type names, and for its title
ROW_LABELS_SPACE = 2.0
ROW_TITLE_SPACE = 0.5


def _draw_class_heatmaps(axes, matrices: dict, annotate: bool, notes: dict = None):
    """One class's upper triangle per axis, each on its own scale; see plot_class_heatmaps."""
    for i, (ax, (label, matrix)) in enumerate(zip(axes, matrices.items())):
        # upper triangle with the diagonal; pairs never connected stay blank
        mask = np.tril(np.ones(matrix.shape, dtype=bool), k=-1) | (matrix.to_numpy() == 0)
        cbar_ax = make_axes_locatable(ax).append_axes('right', size=0.2, pad=0.1)
        sns.heatmap(matrix, mask=mask, cmap=CLASS_CMAPS[label], square=True, ax=ax, cbar_ax=cbar_ax,
                    xticklabels=True, yticklabels=i == 0,
                    annot=annotate, fmt='.3f', annot_kws={'size': VALUE_FONTSIZE})
        if i == len(axes) - 1:
            cbar_ax.set_ylabel('Probability')
        ax.tick_params(length=0)
        ax.tick_params(axis='x', labelrotation=90)
        text = label if notes is None else f'{label}\n{notes[label]}'
        ax.text(0.04, 0.06, text, transform=ax.transAxes, ha='left', va='bottom', fontsize=13)


def plot_class_heatmaps(matrices: dict, annotate: bool, notes: dict = None) -> plt.Figure:
    """Class -> symmetric cell type x cell type matrix, drawn side by side as upper triangles, each on its own scale.

    annotate writes each cell's value in it; notes is extra text per class, written under the class name.
    """
    side = len(next(iter(matrices.values()))) * CELL_SIZE

    fig, axes = plt.subplots(1, len(matrices), figsize=(len(matrices) * side + 3, side))
    _draw_class_heatmaps(axes, matrices, annotate, notes)
    return fig


def plot_class_heatmap_rows(rows: dict, annotate: bool, notes: dict = None) -> plt.Figure:
    """Row title -> class -> matrix: plot_class_heatmaps' panels, one row per title, stacked with the title above.

    notes is row title -> class -> extra text, written under the class name.
    """
    first_row = next(iter(rows.values()))
    side = len(next(iter(first_row.values()))) * CELL_SIZE
    row_height = ROW_TITLE_SPACE + side + ROW_LABELS_SPACE

    fig = plt.figure(figsize=(len(first_row) * side + 3, len(rows) * row_height))
    for subfig, (title, matrices) in zip(np.atleast_1d(fig.subfigures(len(rows), 1)), rows.items()):
        axes = subfig.subplots(1, len(matrices))
        subfig.subplots_adjust(top=1 - ROW_TITLE_SPACE / row_height, bottom=ROW_LABELS_SPACE / row_height)
        _draw_class_heatmaps(axes, matrices, annotate, None if notes is None else notes[title])
        subfig.suptitle(title, fontsize=15, fontweight='bold')
    return fig
