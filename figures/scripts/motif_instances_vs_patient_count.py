"""Discriminative motifs' prevalence against their abundance: the number of patients each motif occurs in, and its mean
number of instances per such patient. One dot per motif, each in its own colour as in the CRC figure (highlighted
motifs in the colours of their per-patient bars), or with --color-by class in the colour of the patient class CISM
assigns it to.

CISM's own pipeline picks the motifs, as in motif_induced_pairwise_interactions_matrix.py: get_features selects
hard-discriminative motifs inside every leave-one-out fold and GetMotifsResult assigns each to the class most of its
patients belong to. A motif occurs in a FOV when FANMOD found it significant there (p < 0.05, the only rows CISM keeps);
a patient's instances of it are its FANMOD counts summed over the patient's FOVs. Motifs come from the dataset's cached
FANMOD results, so FANMOD is not rerun.

For every dataset with cached FANMOD results (TNBC, Human Glioma, Melanoma), at its motif size and stringency unless
told otherwise. Two classes at a time: the dataset's default pair (NN vs NP for Melanoma), or the two given with
--classes. --highlight boxes motifs by FANMOD ID and numbers them I, II, III... in the given order (default: the
dataset's highlighted motifs, Figure 8's motifs I-III for TNBC). --highlight alone, or --n-highlight N, boxes the three
(N) motifs furthest top-right: those no other motif beats on both patient count and mean count, most patients first,
then the next such layer once that is used up (Figure 8's motifs I-III again for TNBC). The per-motif table, with each
motif's edges, is printed and saved next to the figure, to pick motifs to highlight from.

The highlighted motifs also get a follow-up figure (<name>_per_patient, with its counts as csv): one bar chart per
motif, in the motif's colour, of its instances in each patient (log scale), only patients with at least one instance,
grouped by class under a bracket in the class's colour. Each motif's numeral in the scatter plot takes the same colour.

Usage, from the repo root:
    python figures/scripts/motif_instances_vs_patient_count.py --dataset tnbc
    python figures/scripts/motif_instances_vs_patient_count.py --dataset glioma --highlight
    python figures/scripts/motif_instances_vs_patient_count.py --dataset glioma --n-highlight 2
    python figures/scripts/motif_instances_vs_patient_count.py --dataset tnbc --stringency 0.5
"""
import argparse
import sys
from pathlib import Path

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cism.helpers import string_base64_pickle
from figures.helpers.datasets import DATASETS, MOTIF_DATASETS, class_labels, classes_tag
from figures.helpers.motifs import discriminative_motifs, load_discriminator
from figures.helpers.style import ANNOTATION_COLOR, CLASS_COLORS, GRID_COLOR, RESULTS_DIR, apply_style, save_figure

FIGURE_NAME = 'motif_instances_vs_patient_count'
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
# highlighted motif I, II, ... colours, as the CRC figure's motifs I (yellow) and II (green)
MOTIF_COLORS = ['#E2C000', '#2CA02C', '#E07B39', '#7B5BB6', '#17A3B8', '#C2185B', '#8C6D31', '#5C6BC0', '#7CB342',
                '#6D4C41']
# --color-by motif: the other motifs' dots cycle through these (tab10 without its green and olive, which would read
# as motifs II and I), as the CRC figure gives each motif its own colour
OTHER_MOTIF_COLORS = ['#1F77B4', '#FF7F0E', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#17BECF']
# the motifs' numerals as Roman numerals, as the CRC figure's: a bold serif face (so I and II get their serifs), in the
# motif's colour with a dark outline that keeps the yellow readable on white
NUMERAL_STYLE = dict(family='DejaVu Serif', fontweight='bold',
                     path_effects=[patheffects.withStroke(linewidth=2, foreground='#2B2730')])

DOT_SIZE = 60
BOX_PAD = 0.03  # half the side of a highlight box, as a fraction of each axis range
TOP_RIGHT_COUNT = 3  # motifs --highlight alone boxes


def describe_motif(motif: str, cells_type: dict) -> str:
    """'<cell type> - <cell type>; ...', one entry per edge of the motif, e.g. 'CD4 T cell - Tumor; ...'."""
    graph = nx.Graph(string_base64_pickle(motif))
    names = {node: cells_type[int(cell_type)] for node, cell_type in graph.nodes(data='type')}
    return '; '.join(sorted(' - '.join(sorted([names[left], names[right]])) for left, right in graph.edges()))


def motif_table(dataset_key: str, labels: list, motif_size: int, stringency: float, max_class_features: int,
                n_jobs: int) -> tuple:
    """Per discriminative motif (ID index): its class, patient count, mean instances per patient and edges; and the
    instances per (ID, Patient_uId), and each patient's class (Patient_uId index)."""
    config = DATASETS[dataset_key]
    discriminator = load_discriminator(config, motif_size, n_jobs)
    motifs = discriminative_motifs(discriminator, labels, stringency, max_class_features, n_jobs)
    motif_class = {motif_id: label for label, motif_ids in motifs.get_motifs_by_class().items() for motif_id in motif_ids}

    motifs_dataset = discriminator.cism.motifs_dataset
    patient_class = discriminator.get_patients_class(labels).patient_class
    rows = motifs_dataset[motifs_dataset.ID.isin(motifs.all_motif_features)
                          & motifs_dataset.Patient_uId.isin(patient_class.index)]
    per_patient = rows.groupby(['ID', 'Patient_uId'], observed=True).Count.sum()
    table = pd.DataFrame({'patient_class': pd.Series(motif_class),
                          'patient_count': per_patient.groupby('ID').size(),
                          'mean_count': per_patient.groupby('ID').mean(),
                          'edges': rows.groupby('ID').motif.first().map(lambda m: describe_motif(m, config.cells_type))})
    table.index.name = 'ID'
    return table.sort_values(['patient_class', 'patient_count', 'mean_count'], ascending=False), per_patient, patient_class


def top_right_motifs(table: pd.DataFrame, count: int) -> list:
    """The IDs of the `count` motifs furthest top-right: the motifs no other motif beats on both patient count and mean
    count, most patients first, then the next such layer of the remaining motifs once that one is used up."""
    remaining = table.sort_values(['patient_count', 'mean_count'], ascending=False)
    picked = []
    while len(picked) < count and not remaining.empty:
        # every motif before one in this order is in at least as many patients, so it is beaten unless its mean count
        # is above all of theirs
        layer = remaining.mean_count > remaining.mean_count.cummax().shift(fill_value=float('-inf'))
        picked += list(remaining.index[layer])
        remaining = remaining[~layer]
    return picked[:count]


def plot_motif_instances_vs_patient_count(table: pd.DataFrame, labels: list, highlight: list,
                                          color_by: str = 'motif') -> plt.Figure:
    """color_by 'motif': a colour per motif, the highlighted ones in their MOTIF_COLORS, as the CRC figure; 'class':
    each dot in its patient class's colour, with a legend of the classes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if color_by == 'class':
        for label in labels:
            motifs = table[table.patient_class == label]
            ax.scatter(motifs.patient_count, motifs.mean_count, s=DOT_SIZE, color=CLASS_COLORS[label],
                       edgecolor='white', linewidth=1, label=f'{label} (n = {len(motifs)})', zorder=3)
        # above the plot, clear of the dots wherever they fall
        ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1), fontsize=10)
    else:
        others = table.drop(index=highlight)
        colors = [OTHER_MOTIF_COLORS[i % len(OTHER_MOTIF_COLORS)] for i in range(len(others))]
        ax.scatter(others.patient_count, others.mean_count, s=DOT_SIZE, color=colors, edgecolor='white', linewidth=1,
                   zorder=3)
        # the highlighted motifs on top, in the colours of their bars in the per-patient figure
        for color, motif_id in zip(MOTIF_COLORS, highlight):
            ax.scatter(*table.loc[motif_id, ['patient_count', 'mean_count']], s=DOT_SIZE, color=color,
                       edgecolor='white', linewidth=1, zorder=4)
    ax.set_xlabel('Patient count')
    ax.set_ylabel('Average count')
    # room on the right for the numeral of a highlighted motif found in the most patients
    ax.set_xlim(table.patient_count.min() - 1, table.patient_count.max() + 2)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(color=GRID_COLOR, linewidth=0.8, zorder=0)

    (x_min, x_max), (y_min, y_max) = ax.get_xlim(), ax.get_ylim()
    half_width, half_height = BOX_PAD * (x_max - x_min), BOX_PAD * (y_max - y_min)
    for numeral, color, motif_id in zip(ROMAN, MOTIF_COLORS, highlight):
        x, y = table.loc[motif_id, ['patient_count', 'mean_count']]
        ax.add_patch(plt.Rectangle((x - half_width, y - half_height), 2 * half_width, 2 * half_height, fill=False,
                                   edgecolor=ANNOTATION_COLOR, linestyle='--', linewidth=1.2, zorder=4))
        # in the motif's colour, as its bars in the per-patient figure
        ax.annotate(numeral, xy=(x + half_width, y - half_height), xytext=(3, 0), textcoords='offset points',
                    ha='left', va='bottom', fontsize=15, color=color, **NUMERAL_STYLE,
                    annotation_clip=False)  # a box at the x axis has its bottom below it; draw the numeral anyway
    return fig


def per_patient_counts(per_patient: pd.Series, patient_class: pd.Series, labels: list, highlight: list,
                       uid_prefix: str) -> pd.DataFrame:
    """One row per (highlighted motif, patient with at least one instance of it): numeral, ID, patient, class, count."""
    rows = []
    for numeral, motif_id in zip(ROMAN, highlight):
        for uid, count in per_patient.loc[motif_id].items():
            if count > 0:
                patient = uid[len(uid_prefix):] if uid.startswith(uid_prefix) else uid
                rows.append(dict(motif=numeral, ID=motif_id, patient=patient, patient_class=patient_class[uid],
                                 count=int(count)))
    counts = pd.DataFrame(rows)
    # classes in the figure's order, patients numerically within each
    counts['class_order'] = counts.patient_class.map({label: i for i, label in enumerate(labels)})
    counts['patient_order'] = pd.to_numeric(counts.patient, errors='coerce')
    return counts.sort_values(['class_order', 'patient_order', 'patient']).drop(columns=['class_order', 'patient_order'])


def plot_per_patient_counts(counts: pd.DataFrame, labels: list) -> plt.Figure:
    """One bar chart per highlighted motif, in its colour: instances per patient on a log scale, the patients of each
    class together under a bracket with the class name. A one-bar gap separates the classes."""
    motifs = [numeral for numeral in ROMAN if numeral in set(counts.motif)]  # I on top
    # bars plus a one-bar gap between the classes the motif occurs in
    widest = max(len(rows) + rows.patient_class.nunique() - 1 for _, rows in counts.groupby('motif'))
    # each bar's slot wide enough for its patient ID written across it at 10 pt
    slot = max(0.42, 0.085 * counts.patient.str.len().max() + 0.12)
    fig_height = 2.6 * len(motifs)
    fig, axes = plt.subplots(len(motifs), 1, figsize=(max(7.0, slot * widest + 1.6), fig_height),
                             squeeze=False, gridspec_kw=dict(hspace=0.75))
    for ax, numeral in zip(axes[:, 0], motifs):
        rows = counts[counts.motif == numeral]
        color = MOTIF_COLORS[ROMAN.index(numeral)]
        x, ticks, names = 0, [], []
        for label in labels:
            patients = rows[rows.patient_class == label]
            if patients.empty:
                continue
            positions = list(range(x, x + len(patients)))
            ax.bar(positions, patients['count'], width=0.75, color=color, zorder=3)
            ticks += positions
            names += list(patients.patient)
            # bracket under the class's patients, below the tick labels: ends turned up, a notch down in the middle
            bracket = ax.get_xaxis_transform()
            left, right, middle, y = positions[0] - 0.4, positions[-1] + 0.4, (positions[0] + positions[-1]) / 2, -0.3
            class_color = CLASS_COLORS[label]
            ax.plot([left, left, right, right], [y + 0.06, y, y, y + 0.06], color=class_color, linewidth=1.8,
                    transform=bracket, clip_on=False, solid_capstyle='round')
            ax.plot([middle, middle], [y, y - 0.06], color=class_color, linewidth=1.8, transform=bracket,
                    clip_on=False)
            ax.text(middle, y - 0.1, label, transform=bracket, ha='center', va='top', color=class_color,
                    fontsize=12, fontweight='bold')
            x = positions[-1] + 2
        ax.set_yscale('log')
        ax.set_ylim(0.7, rows['count'].max() * 2)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        ax.set_xticks(ticks, names, fontsize=10)
        ax.set_xlim(-0.7, max(widest, x - 1) - 0.3)
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.text(0.005, 1.04, numeral, transform=ax.transAxes, ha='left', va='bottom', fontsize=16, color=color,
                **NUMERAL_STYLE)
    fig.supylabel('Count', fontsize=13)
    # under the bottom chart's class names, which hang ~0.85 in below its axis
    bottom = axes[-1, 0].get_position().y0 - 1.05 / fig_height
    fig.text(0.5, bottom, 'Patient', ha='center', va='top', fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=MOTIF_DATASETS, required=True)
    parser.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
    parser.add_argument('--motif-size', type=int, help="default: the dataset's")
    parser.add_argument('--stringency', type=float, help="default: the dataset's")
    parser.add_argument('--max-class-features', type=int, help="default: the dataset's")
    parser.add_argument('--highlight', nargs='*', type=int,
                        help="FANMOD IDs of the motifs to box, numbered I, II, III... (default: the dataset's; "
                             f"no IDs: the {TOP_RIGHT_COUNT} furthest top-right)")
    parser.add_argument('--n-highlight', type=int, metavar='N',
                        help='box the N motifs furthest top-right (instead of IDs given to --highlight)')
    parser.add_argument('--color-by', choices=['motif', 'class'], default='motif',
                        help="'motif' (default, as the CRC figure): a colour per motif; 'class': the two classes' "
                             "colours, with their legend")
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
    if args.highlight is not None and len(args.highlight) > len(ROMAN):
        parser.error(f'at most {len(ROMAN)} motifs can be highlighted')
    if args.n_highlight is not None:
        if args.highlight:
            parser.error('give --highlight IDs or --n-highlight, not both')
        if not 1 <= args.n_highlight <= len(ROMAN):
            parser.error(f'--n-highlight must be between 1 and {len(ROMAN)}')

    table, per_patient, patient_class = motif_table(args.dataset, labels, motif_size, stringency, max_class_features,
                                                    args.n_jobs)
    if args.n_highlight is not None:
        highlight = top_right_motifs(table, args.n_highlight)
    elif args.highlight is None:
        highlight = list(config.motif_run.highlighted_motifs)
    elif not args.highlight:
        highlight = top_right_motifs(table, TOP_RIGHT_COUNT)
    else:
        highlight = args.highlight
    missing = sorted(set(highlight) - set(table.index))
    if missing:
        sys.exit(f'not discriminative motifs of this run, cannot highlight: {missing}')
    with pd.option_context('display.width', 250, 'display.max_colwidth', 120, 'display.max_rows', None,
                           'display.max_columns', None):
        print(table)
    if highlight:
        print('highlighted: ' + ', '.join(f'{numeral} = {motif_id}' for numeral, motif_id in zip(ROMAN, highlight)))

    apply_style()
    fig = plot_motif_instances_vs_patient_count(table, labels, highlight, args.color_by)
    name = FIGURE_NAME + classes_tag(config, labels) + f'_{motif_size}node_stringency{stringency:g}'
    for path in save_figure(fig, args.dataset, name, formats=args.formats):
        print(f'saved {path}')
    plt.close(fig)
    table_path = RESULTS_DIR / args.dataset / f'{name}.csv'
    table.to_csv(table_path)
    print(f'saved {table_path}')

    if highlight:
        counts = per_patient_counts(per_patient, patient_class, labels, highlight, config.patient_uid_prefix)
        fig = plot_per_patient_counts(counts, labels)
        for path in save_figure(fig, args.dataset, f'{name}_per_patient', formats=args.formats):
            print(f'saved {path}')
        plt.close(fig)
        counts_path = RESULTS_DIR / args.dataset / f'{name}_per_patient.csv'
        counts.to_csv(counts_path, index=False)
        print(f'saved {counts_path}')


if __name__ == '__main__':
    main()
