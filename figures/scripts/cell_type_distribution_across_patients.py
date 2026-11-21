"""Cell type frequency in each patient class: each cell type's share of all graph nodes of the class's patients.

For every dataset whose CISM graphs are in data/ (TNBC, Human Glioma, Melanoma). Two classes at a time: the dataset's
default pair (NN vs NP for Melanoma), or the two given with --classes.

Usage, from the repo root:
    python figures/scripts/cell_type_distribution_across_patients.py --dataset glioma
    python figures/scripts/cell_type_distribution_across_patients.py --dataset melanoma --classes NN PN
    python figures/scripts/cell_type_distribution_across_patients.py                    # every dataset with graphs
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.pairwise.pairwise_model import PairwiseAnalysis
from figures.helpers.datasets import DATASETS, class_labels, classes_tag, tissue_state_kwargs
from figures.helpers.graphs import read_graphs
from figures.helpers.style import CLASS_COLORS, GRID_COLOR, apply_style, save_figure

FIGURE_NAME = 'cell_type_distribution_across_patients'
GRAPH_DATASETS = [key for key, config in DATASETS.items() if config.graphs_dir is not None]

BARS_WIDTH = 0.8  # of one cell type's slot, shared by its class bars


def cell_type_frequencies(dataset_key: str, labels: list) -> pd.DataFrame:
    """Cell type x class frequencies, the cell type most frequent on average over the classes first."""
    config = DATASETS[dataset_key]
    full_graph_df = read_graphs(config)
    counts = PairwiseAnalysis(**tissue_state_kwargs(config)).get_cell_type_count_from_classes(
        full_graph_df=full_graph_df, classes=labels, cells_type=config.cells_type, normalize=True)

    frequencies = (counts.pivot(index='Cell Type', columns='patient_class', values='Frequency')
                   .reindex(columns=labels).fillna(0))
    return frequencies.loc[frequencies.mean(axis=1).sort_values(ascending=False).index]


def plot_cell_type_distribution(dataset_key: str, labels: list, ax: plt.Axes = None,
                                frequencies: pd.DataFrame = None) -> plt.Figure:
    """Draws into ax when given (a multi-panel figure), else into a new figure; frequencies skips re-reading graphs."""
    if frequencies is None:
        frequencies = cell_type_frequencies(dataset_key, labels)
    x = np.arange(len(frequencies))
    width = BARS_WIDTH / len(frequencies.columns)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure
    for i, patient_class in enumerate(frequencies.columns):
        offset = (i - (len(frequencies.columns) - 1) / 2) * width
        ax.bar(x + offset, frequencies[patient_class], width=width, color=CLASS_COLORS[patient_class], linewidth=0,
               label=patient_class)
    ax.set_xticks(x, frequencies.index, rotation=90, fontsize=10)
    ax.set_xlim(-0.7, len(x) - 0.3)
    ax.set_xlabel('Cell type')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right', frameon=False, fontsize=12.5)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', choices=GRAPH_DATASETS, default=GRAPH_DATASETS)
    parser.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    args = parser.parse_args()
    if args.classes and len(args.dataset) > 1:
        parser.error('--classes needs a single --dataset')
    try:
        dataset_labels = {key: class_labels(DATASETS[key], args.classes) for key in args.dataset}
    except ValueError as exc:
        parser.error(str(exc))

    apply_style()
    for dataset_key, labels in dataset_labels.items():
        config = DATASETS[dataset_key]
        fig = plot_cell_type_distribution(dataset_key, labels)
        for path in save_figure(fig, dataset_key, FIGURE_NAME + classes_tag(config, labels), formats=args.formats):
            print(f'saved {path}')
        plt.close(fig)


if __name__ == '__main__':
    main()
