"""Figure S8: robustness of the discriminative motif-induced pairwise interactions to the discrimination stringency
parameter. One row per stringency (0.2, 0.3, 0.4 by default), and in each row, one upper-triangle heatmap per class
(NN, NP). Each heatmap is the probability of two cell types being connected by an edge in the pooled instances of
that class's 4-cell discriminative motifs.

Computed the same way as motif_induced_pairwise_interactions_matrix.py, at each stringency in turn. Motifs come from
the dataset's cached FANMOD results, so FANMOD is not rerun. Each panel shows its class's motif count and writes each
cell's probability in it.

Usage, from the repo root:
    python figures/scripts/motif_induced_pairwise_interaction_by_stringency_parameter_s8_figure.py
    python figures/scripts/motif_induced_pairwise_interaction_by_stringency_parameter_s8_figure.py --stringencies 0.2 0.3
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figures.helpers.datasets import DATASETS, MOTIF_DATASETS, class_labels, classes_tag
from figures.helpers.heatmaps import plot_class_heatmap_rows
from figures.helpers.motifs import load_discriminator, motif_induced_matrices
from figures.helpers.style import apply_style, save_figure

FIGURE_NAME = 'motif_induced_pairwise_interaction_by_stringency_parameter_s8_figure'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=MOTIF_DATASETS, default='melanoma')
    parser.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
    parser.add_argument('--motif-size', type=int, default=4)
    parser.add_argument('--stringencies', type=float, nargs='+', default=[0.2, 0.3, 0.4])
    parser.add_argument('--max-class-features', type=int, help="default: the dataset's")
    # 4 jobs ran out of a 24G session on Melanoma's 4-node motifs; 1 job peaks at ~5G and takes ~4 min
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    args = parser.parse_args()
    config = DATASETS[args.dataset]
    try:
        labels = class_labels(config, args.classes)
    except ValueError as exc:
        parser.error(str(exc))
    max_class_features = args.max_class_features or config.motif_run.max_class_features

    discriminator = load_discriminator(config, args.motif_size, args.n_jobs)
    rows, notes = {}, {}
    for stringency in args.stringencies:
        matrices, motif_counts = motif_induced_matrices(discriminator, config.cells_type, labels, stringency,
                                                        max_class_features, args.n_jobs)
        print(f'stringency {stringency:g}: discriminative motifs per class:', motif_counts)
        title = f'Discrimination stringency {stringency:g}'
        rows[title] = matrices
        notes[title] = {label: f'n = {count} motifs' for label, count in motif_counts.items()}

    apply_style()
    fig = plot_class_heatmap_rows(rows, annotate=True, notes=notes)
    name = (FIGURE_NAME + classes_tag(config, labels) + f'_{args.motif_size}node_stringency'
            + '_'.join(f'{stringency:g}' for stringency in args.stringencies))
    for path in save_figure(fig, args.dataset, name, formats=args.formats):
        print(f'saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
