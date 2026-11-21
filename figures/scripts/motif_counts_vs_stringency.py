"""Number of discriminative motifs against the discrimination stringency parameter, one line per motif size and patient
class: the motifs found (FANMOD p < 0.05) in one class's patients only and carried by more than the stringency fraction
of them, counted over the whole cohort with no per-class cap, as TissueStateDiscriminativeMotifs.discover finds them.

It also tabulates the motifs CISM's classifier actually selects, at 0 and the stringencies of the AUC sweep (steps of
0.05, every 0.02 around the dataset's own stringency; figures.helpers.motifs.sweep_stringencies): get_features keeps,
in every leave-one-out fold, the max_class_features most frequent motifs per class above the stringency, and
GetMotifsResult takes their union over the folds. The table is saved as csv and drawn as a light and a dark png.

Motifs come from the dataset's cached FANMOD results. FANMOD runs only for FOVs without one, which takes hours to days
for 5-node motifs, and a 5-node motif table takes tens of GB of memory (~60 GB for Human Glioma): run this on a compute
node.

For every dataset with cached FANMOD results (TNBC, Human Glioma, Melanoma). Two classes at a time: the dataset's
default pair (NN vs NP for Melanoma), or the two given with --classes.

Usage, from the repo root:
    python figures/scripts/motif_counts_vs_stringency.py --dataset glioma
    python figures/scripts/motif_counts_vs_stringency.py --dataset melanoma --classes NN PN --motif-sizes 3 4
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cism.cism import DiscriminativeFeatureKey
from figures.helpers.datasets import DATASETS, MOTIF_DATASETS, class_labels, classes_tag
from figures.helpers.motifs import MOTIF_SIZES, discriminative_motifs, load_discriminator, sweep_stringencies
from figures.helpers.style import MOTIF_SIZE_COLORS, apply_style, results_dir, save_figure

FIGURE_NAME = 'motif_counts_vs_stringency'
CURVE_STRINGENCIES = tuple(round(0.01 * step, 2) for step in range(71))  # 0, 0.01, ..., 0.70
CLASS_LINESTYLES = ('--', ':')  # the first and the second class
TABLE_TITLE = 'Discriminative motifs per stringency'
# the table's palette per theme; motif sizes keep their figure colors, lightened on the dark background
TABLE_THEMES = {
    'light': dict(surface='#FCFCFB', primary='#0B0B0B', secondary='#52514E', muted='#A8A69F', rule='#E2E1DC',
                  band='#F1EFFA', size=MOTIF_SIZE_COLORS),
    'dark': dict(surface='#1A1A19', primary='#FFFFFF', secondary='#C3C2B7', muted='#6B6A63', rule='#33332F',
                 band='#2A2638', size={3: '#58B558', 4: '#A77EE6', 5: '#E89A4A'}),
}


def class_exclusive_counts(discriminator, labels: list, stringencies: list) -> pd.DataFrame:
    """Per stringency and class: the class's exclusive motifs carried by more than the stringency fraction of its
    patients."""
    discovered = discriminator.discover(extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
                                        classes=labels).get_discriminative_motifs()
    # each class-exclusive motif belongs to one class and has one prevalence in it
    motifs = discovered.drop_duplicates('ID')
    return pd.DataFrame([dict(stringency=stringency, patient_class=label,
                              n_motifs=int(((motifs.patient_percentage > stringency)
                                            & (motifs.patient_class == label)).sum()))
                         for stringency in stringencies for label in labels])


def discriminative_counts(discriminator, labels: list, stringencies: list, max_class_features: int) -> pd.DataFrame:
    """Per stringency (ascending) and class: the motifs get_features selects in any leave-one-out fold, in the class
    GetMotifsResult assigns them to."""
    rows, by_class = [], None
    for stringency in stringencies:
        # past a stringency at which no fold keeps a motif, none can
        if by_class is None or any(by_class.values()):
            # one fold at a time: worker processes would each take a copy of the motif table
            by_class = discriminative_motifs(discriminator, labels, stringency, max_class_features,
                                             n_jobs=1).get_motifs_by_class()
        rows += [dict(stringency=stringency, patient_class=label, n_motifs=len(by_class[label])) for label in labels]
        print(f'  stringency {stringency:g}: ' + ', '.join(f'{len(by_class[label])} {label}' for label in labels),
              flush=True)
    return pd.DataFrame(rows)


def stack_sizes(counts: dict) -> pd.DataFrame:
    """Motif size -> counts, stacked into one table led by a motif_size column."""
    return pd.concat(counts, names=['motif_size']).reset_index(level='motif_size').reset_index(drop=True)


def pivot_with_totals(counts: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Stringency x (motif size, class or total) motif counts."""
    table = counts.pivot_table(index='stringency', columns=['motif_size', 'patient_class'], values='n_motifs',
                               aggfunc='sum')
    for size in table.columns.get_level_values(0).unique():
        table[(size, 'total')] = table[(size, labels[0])] + table[(size, labels[1])]
    return table.sort_index(axis=1)


def table_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The table's rows to draw: up to the last one with a motif."""
    nonzero = table.index[(table > 0).any(axis=1)]
    return table.loc[:nonzero.max()] if len(nonzero) else table


def plot_motif_counts(counts: pd.DataFrame, labels: list, ax: plt.Axes = None) -> plt.Figure:
    """Draws into ax when given (a multi-panel figure), else into a new figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
    else:
        fig = ax.figure
    handles = []
    for size, by_size in counts.groupby('motif_size'):
        for label, linestyle in zip(labels, CLASS_LINESTYLES):
            # zero has no place on a log axis
            curve = by_size[(by_size.patient_class == label) & (by_size.n_motifs > 0)]
            handles += ax.plot(curve.stringency, curve.n_motifs, color=MOTIF_SIZE_COLORS.get(size, 'black'),
                               linestyle=linestyle, linewidth=2, label=f'{size}-node, {label}')
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.8)
    ax.set_xlim(0, counts.stringency[counts.n_motifs > 0].max() + 0.02)
    ax.set_xlabel('Fraction of patients that share the motif\n(Discrimination stringency parameter)')
    ax.set_ylabel('#Discriminative motifs')
    # a column per motif size: legends fill column by column
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=counts.motif_size.nunique(),
              frameon=False, fontsize=10.5, handlelength=2.6)
    return fig


def plot_discriminative_table(table: pd.DataFrame, labels: list, selected: float, subtitle: str, footnote: str,
                              colors: dict) -> plt.Figure:
    """The discriminative motif counts as a presentation table: a row per stringency, the selected one highlighted,
    and per motif size a column per class and their total."""
    sizes = sorted({size for size, _ in table.columns})
    columns = list(labels) + ['total']
    col_w = max(1.15, 0.09 * max(map(len, columns)) + 0.4)  # inches, room for the longest class name
    label_w, row_h, x0 = 1.9, 0.42, 0.15
    group_y, class_y, body_top = 1.35, 1.83, 2.2
    group_w = col_w * len(columns)
    width = label_w + group_w * len(sizes) + 0.3
    height = body_top + row_h * len(table) + 0.75

    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(colors['surface'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    ax.text(x0, 0.38, TABLE_TITLE, fontsize=15, fontweight='600', color=colors['primary'], va='center')
    ax.text(x0, 0.72, subtitle, fontsize=9.5, color=colors['secondary'], va='center')
    ax.text(x0 + 0.25, class_y, 'Stringency', fontsize=10.5, fontweight='600', color=colors['secondary'], va='center')
    for i, size in enumerate(sizes):
        left = x0 + label_w + i * group_w
        size_color = colors['size'].get(size, colors['primary'])
        ax.text(left + group_w / 2, group_y, f'{size}-node motifs', fontsize=11.5, fontweight='700', color=size_color,
                ha='center', va='center')
        ax.plot([left + 0.12, left + group_w - 0.12], [group_y + 0.24] * 2, color=size_color, linewidth=2,
                solid_capstyle='round')
        for j, column in enumerate(columns):
            ax.text(left + (j + 1) * col_w - 0.2, class_y, column, fontsize=10.5, color=colors['secondary'], ha='right',
                    va='center', fontweight='600' if column == 'total' else 'normal')
        if i:
            ax.plot([left, left], [class_y - 0.2, body_top + len(table) * row_h], color=colors['rule'], linewidth=0.8)
    ax.plot([x0, width - 0.15], [body_top - 0.08] * 2, color=colors['secondary'], linewidth=0.9)

    for r, (stringency, row) in enumerate(table.iterrows()):
        y = body_top + (r + 0.5) * row_h
        is_selected = np.isclose(stringency, selected)
        if is_selected:
            ax.add_patch(plt.Rectangle((x0, y - row_h / 2), width - 0.3, row_h, color=colors['band'], linewidth=0,
                                       zorder=0))
        ax.text(x0 + 0.25, y, f'{stringency:g}  (selected)' if is_selected else f'{stringency:.2f}', fontsize=10.5,
                color=colors['primary'], va='center', fontweight='700' if is_selected else 'normal')
        for i, size in enumerate(sizes):
            for j, column in enumerate(columns):
                value = int(row[(size, column)])
                ax.text(x0 + label_w + i * group_w + (j + 1) * col_w - 0.2, y, f'{value:,}', fontsize=10.5,
                        ha='right', va='center', color=colors['muted'] if value == 0 else colors['primary'],
                        fontweight='700' if is_selected or column == 'total' else 'normal')
        if r < len(table) - 1:
            ax.plot([x0, width - 0.15], [y + row_h / 2] * 2, color=colors['rule'], linewidth=0.6)

    ax.text(x0, body_top + len(table) * row_h + 0.35, footnote, fontsize=8.8, color=colors['secondary'], va='center')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=MOTIF_DATASETS, required=True)
    parser.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
    parser.add_argument('--motif-sizes', nargs='+', type=int, default=list(MOTIF_SIZES))
    parser.add_argument('--max-class-features', type=int, help="default: the dataset's")
    parser.add_argument('--n-jobs', type=int, default=4, help='FOVs to run FANMOD on and parse at a time')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    args = parser.parse_args()
    config = DATASETS[args.dataset]
    try:
        labels = class_labels(config, args.classes)
    except ValueError as exc:
        parser.error(str(exc))
    max_class_features = args.max_class_features or config.motif_run.max_class_features
    selected = round(config.motif_run.stringency, 4)
    table_stringencies = sorted({0.0} | set(sweep_stringencies(selected)))
    curve_stringencies = sorted(set(CURVE_STRINGENCIES) | set(table_stringencies))

    exclusive, discriminative = {}, {}
    for size in args.motif_sizes:
        print(f'\n=== {size}-node motifs ===', flush=True)
        discriminator = load_discriminator(config, size, args.n_jobs)
        class_sizes = discriminator.get_patients_class(labels).patient_class.value_counts()
        exclusive[size] = class_exclusive_counts(discriminator, labels, curve_stringencies)
        print(f'Discriminative {size}-node motifs:', flush=True)
        discriminative[size] = discriminative_counts(discriminator, labels, table_stringencies, max_class_features)
        del discriminator  # a 5-node motif table takes tens of GB

    name = FIGURE_NAME + classes_tag(config, labels)
    out_dir = results_dir(args.dataset)
    exclusive = stack_sizes(exclusive)
    exclusive.to_csv(out_dir / f'{name}_long_format.csv', index=False)
    table = pivot_with_totals(exclusive[exclusive.stringency.isin(table_stringencies)], labels)
    table.to_csv(out_dir / f'{name}_table.csv')
    print('\nClass-exclusive motifs above the stringency (whole cohort, no per-class cap):')
    print(table.to_string())
    discriminative = pivot_with_totals(stack_sizes(discriminative), labels)
    discriminative.to_csv(out_dir / f'{name}_discriminative_table.csv')
    print('\nDiscriminative motifs (union over the leave-one-out folds of each fold\'s selection):')
    print(discriminative.to_string())
    print(f'saved the csv tables to {out_dir}')

    apply_style()
    fig = plot_motif_counts(exclusive, labels)
    for path in save_figure(fig, args.dataset, name, formats=args.formats):
        print(f'saved {path}')
    plt.close(fig)

    subtitle = (f'Union over the {class_sizes.sum()} leave-one-out folds of the top {max_class_features} motifs per '
                'class above the stringency  ·  '
                + ' / '.join(f'{class_sizes.get(label, 0)} {label}' for label in labels))
    footnote = (f"Selected by CISM's get_features and GetMotifsResult; each fold keeps at most {max_class_features} "
                'motifs per class.')
    shown = table_rows(discriminative)
    for theme, colors in TABLE_THEMES.items():
        fig = plot_discriminative_table(shown, labels, selected, subtitle, footnote, colors)
        for path in save_figure(fig, args.dataset, f'{name}_discriminative_table_{theme}', formats=['png']):
            print(f'saved {path}')
        plt.close(fig)


if __name__ == '__main__':
    main()
