"""Motif-induced pairwise interactions in each patient class: the probability of two cell types being connected by an
edge in the pooled instances of the class's discriminative motifs. One upper-triangle heatmap per class, side by side.

CISM's own pipeline picks the motifs: TissueStateDiscriminativeMotifs.get_features selects hard-discriminative motifs
inside every leave-one-out fold, GetMotifsResult assigns each to the class most of its patients belong to, and
cism.helpers.create_weight_matrix_from_motifs weighs each motif's edges by its mean frequency. Motifs come from the
dataset's cached FANMOD results, so FANMOD is not rerun.

For every dataset with cached FANMOD results (TNBC, Human Glioma, Melanoma), at its motif size and stringency unless
told otherwise. Two classes at a time: the dataset's default pair (NN vs NP for Melanoma), or the two given with
--classes. --annotate writes each cell's probability in it.

Usage, from the repo root:
    python figures/scripts/motif_induced_pairwise_interactions_matrix.py --dataset melanoma
    python figures/scripts/motif_induced_pairwise_interactions_matrix.py --dataset melanoma --motif-size 4 --annotate
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figures.helpers.datasets import DATASETS, MOTIF_DATASETS, class_labels, classes_tag
from figures.helpers.heatmaps import plot_class_heatmaps
from figures.helpers.motifs import load_discriminator, motif_induced_matrices
from figures.helpers.style import apply_style, save_figure

FIGURE_NAME = 'motif_induced_pairwise_interactions_matrix'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=MOTIF_DATASETS, required=True)
    parser.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
    parser.add_argument('--motif-size', type=int, help="default: the dataset's")
    parser.add_argument('--stringency', type=float, help="default: the dataset's")
    parser.add_argument('--max-class-features', type=int, help="default: the dataset's")
    parser.add_argument('--annotate', action='store_true', help="write each cell's probability in it")
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    args = parser.parse_args()
    config = DATASETS[args.dataset]
    try:
        labels = class_labels(config, args.classes)
    except ValueError as exc:
        parser.error(str(exc))
    motif_size = args.motif_size or config.motif_run.motif_size
    stringency = args.stringency if args.stringency is not None else config.motif_run.stringency
    max_class_features = args.max_class_features or config.motif_run.max_class_features

    discriminator = load_discriminator(config, motif_size, args.n_jobs)
    matrices, motif_counts = motif_induced_matrices(discriminator, config.cells_type, labels, stringency,
                                                    max_class_features, args.n_jobs)
    print('discriminative motifs per class:', motif_counts)
    apply_style()
    fig = plot_class_heatmaps(matrices, args.annotate,
                              notes={label: f'n = {count} motifs' for label, count in motif_counts.items()})
    name = (FIGURE_NAME + classes_tag(config, labels) + f'_{motif_size}node_stringency{stringency:g}'
            + ('_values' if args.annotate else ''))
    for path in save_figure(fig, args.dataset, name, formats=args.formats):
        print(f'saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
