"""Number of patients alive over survival time, split into short- and long-term survival at the CISM cutoff.

Only for datasets whose classes are short vs long survival (TNBC, Human Glioma).

Usage, from the repo root:
    python figures/scripts/short_vs_long_patients.py --dataset glioma
    python figures/scripts/short_vs_long_patients.py                    # every survival dataset
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figures.helpers.datasets import DATASETS, LONG_TERM, SHORT_TERM, load_survival_classes
from figures.helpers.style import ANNOTATION_COLOR, CLASS_COLORS, CUTOFF_COLOR, apply_style, save_figure

FIGURE_NAME = 'short_vs_long_patients'
SURVIVAL_DATASETS = [key for key, config in DATASETS.items() if config.survival is not None]

ARROW_Y = 1.09  # axes fraction, above the plot
LABEL_Y = 1.13
CUTOFF_GAP = 0.02  # fraction of the x range left empty on each side of the cutoff between the two arrows


def plot_short_vs_long_patients(dataset_key: str) -> plt.Figure:
    config = DATASETS[dataset_key]
    cutoff = config.survival.cutoff_days
    patients = load_survival_classes(config)
    group_sizes = patients['patient_class'].value_counts()

    # number of patients still alive at each survival time, starting from the whole cohort at day 0
    survival_days = np.sort(patients['patient_class_id'].to_numpy())
    n_patients = len(survival_days)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(np.r_[0, survival_days], n_patients - np.arange(n_patients + 1), where='post', color='black', linewidth=2)
    ax.set_xlim(left=0)
    ax.set_ylim(0, n_patients * 1.06)
    ax.set_xlabel(config.survival.axis_label)
    ax.set_ylabel('Count')

    ax.axvline(cutoff, color=CUTOFF_COLOR, linestyle=':', linewidth=2)
    ax.annotate(f'{cutoff / 365.25:.1f} years\n({cutoff} days)', xy=(cutoff, 0), xytext=(5, 5),
                textcoords='offset points', ha='left', va='bottom', color=ANNOTATION_COLOR, fontsize=11)

    x_min, x_max = ax.get_xlim()
    gap = CUTOFF_GAP * (x_max - x_min)
    data_x_axes_y = blended_transform_factory(ax.transData, ax.transAxes)
    for patient_class, name, start, end in [(SHORT_TERM, 'Short-term', cutoff - gap, x_min),
                                            (LONG_TERM, 'Long-term', cutoff + gap, x_max)]:
        color = CLASS_COLORS[patient_class]
        ax.annotate('', xy=(end, ARROW_Y), xytext=(start, ARROW_Y), xycoords=data_x_axes_y,
                    arrowprops=dict(arrowstyle='-|>', color=color, linewidth=2.5, mutation_scale=18),
                    annotation_clip=False)
        ax.text((start + end) / 2, LABEL_Y, f'{name} ({group_sizes.get(patient_class, 0)})', transform=data_x_axes_y,
                ha='center', va='bottom', color=color, fontsize=13)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', choices=SURVIVAL_DATASETS, default=SURVIVAL_DATASETS)
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    args = parser.parse_args()

    apply_style()
    for dataset_key in args.dataset:
        fig = plot_short_vs_long_patients(dataset_key)
        for path in save_figure(fig, dataset_key, FIGURE_NAME, formats=args.formats):
            print(f'saved {path}')
        plt.close(fig)


if __name__ == '__main__':
    main()
