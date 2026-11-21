"""Pairwise interactions in each patient class: the probability of two cell types being connected by an edge in the
class's pooled multicellular networks. One upper-triangle heatmap per class, side by side.

For every dataset whose CISM graphs are in data/ (TNBC, Human Glioma, Melanoma). Two classes at a time: the dataset's
default pair (NN vs NP for Melanoma), or the two given with --classes. --annotate writes each cell's probability in it.

Usage, from the repo root:
    python figures/scripts/pairwise_interactions_matrix.py --dataset melanoma
    python figures/scripts/pairwise_interactions_matrix.py --dataset melanoma --classes PN NP --annotate
    python figures/scripts/pairwise_interactions_matrix.py                    # every dataset with graphs
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.pairwise.pairwise_model import PairwiseAnalysis
from figures.helpers.datasets import DATASETS, class_labels, classes_tag, tissue_state_kwargs
from figures.helpers.graphs import read_graphs
from figures.helpers.heatmaps import plot_class_heatmaps
from figures.helpers.style import apply_style, save_figure

FIGURE_NAME = 'pairwise_interactions_matrix'
GRAPH_DATASETS = [key for key, config in DATASETS.items() if config.graphs_dir is not None]


def pairwise_matrices(dataset_key: str, labels: list) -> dict:
    """Class -> cell type x cell type probabilities; cell types connected in neither class are dropped."""
    config = DATASETS[dataset_key]
    full_graph_df = read_graphs(config)
    analysis = PairwiseAnalysis(**tissue_state_kwargs(config))
    matrices = {label: analysis.get_pairwise_counter(full_graph_df=full_graph_df, patient_class=label,
                                                     cells_type=config.cells_type)
                for label in labels}
    cell_types = [cell_type for cell_type in sorted(config.cells_type.values())
                  if any(matrix.loc[cell_type].sum() > 0 for matrix in matrices.values())]
    return {label: matrix.loc[cell_types, cell_types] for label, matrix in matrices.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', choices=GRAPH_DATASETS, default=GRAPH_DATASETS)
    parser.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
    parser.add_argument('--annotate', action='store_true', help="write each cell's probability in it")
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
        fig = plot_class_heatmaps(pairwise_matrices(dataset_key, labels), args.annotate)
        name = FIGURE_NAME + classes_tag(DATASETS[dataset_key], labels) + ('_values' if args.annotate else '')
        for path in save_figure(fig, dataset_key, name, formats=args.formats):
            print(f'saved {path}')
        plt.close(fig)


if __name__ == '__main__':
    main()
