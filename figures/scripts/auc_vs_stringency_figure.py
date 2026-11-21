"""CISM's leave-one-patient-out ROC AUC against the discrimination stringency parameter, one line per motif size, with
dashed baselines scored on the same patients: pairwise cell type contacts (baselines/pairwise), cell type distribution
(baselines/cell_type_distribution), chance (AUC = 0.5), and the GNN (baselines/GNN, run separately) where the dataset
lists its AUCs.

Every point reruns TissueStateDiscriminativeMotifs.analyze_motifs: the hard-discriminative motifs are re-selected inside
every leave-one-out fold and a random forest is fit on the patients' motif frequencies, its forests drawing from
RandomState(seed). Lines are the mean over the seeds (--seed-band shades +/- 1 SD around them); the dashed square marks the dataset's own motif
size and stringency. Every motif size is scored at the same stringencies: steps of 0.05, resampled every 0.02 around
the dataset's own stringency, which takes the place of the step nearest it (figures.helpers.motifs.sweep_stringencies;
0.4411 gives ..., 0.40, 0.42, 0.4411, 0.46, 0.48, 0.50, ...). Motifs come from the dataset's cached FANMOD results.
FANMOD runs only for FOVs without one, which takes hours to days for 5-node motifs, and a 5-node motif table takes
tens of GB of memory: run this on a compute node.

Each result is appended to <figure>_results.csv next to the figure as soon as it is computed, so a rerun picks up where
an interrupted one stopped; --plot-only redraws the figure from it.

For every dataset with cached FANMOD results (TNBC, Human Glioma, Melanoma). Two classes at a time: the dataset's
default pair (NN vs NP for Melanoma), or the two given with --classes.

Usage, from the repo root:
    python figures/scripts/auc_vs_stringency_figure.py --dataset glioma
    python figures/scripts/auc_vs_stringency_figure.py --dataset tnbc --motif-sizes 3 4 --seeds 3
    python figures/scripts/auc_vs_stringency_figure.py --dataset glioma --plot-only
    python figures/scripts/auc_vs_stringency_figure.py --dataset glioma --plot-only --seed-band
"""
import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.cell_type_distribution import CellTypeDistributionAnalysis
from baselines.pairwise import PairwiseAnalysis
from figures.helpers.datasets import (DATASETS, MOTIF_DATASETS, class_labels, classes_tag, load_patient_classes,
                                      tissue_state_kwargs)
from figures.helpers.graphs import read_graphs
from figures.helpers.motifs import (MOTIF_SIZES, STRINGENCY_GRID, hard_discriminative_fc, load_discriminator,
                                    sweep_stringencies)
from figures.helpers.style import MOTIF_SIZE_COLORS, apply_style, results_dir, save_figure

FIGURE_NAME = 'auc_vs_stringency'
N_SEEDS = 5
RESULT_COLUMNS = ['key', 'method', 'motif_size', 'stringency', 'max_class_features', 'seed', 'auc', 'mean_features',
                  'n_patients', 'status']
# method -> line name, color, and its analysis: built from the patient class arguments and run like PairwiseAnalysis
BASELINES = {
    'pairwise': ('Pairwise', '#E0559A', PairwiseAnalysis),
    'cell_type': ('Cell Type Distribution', '#2B3FD6', CellTypeDistributionAnalysis),
}
RANDOM_COLOR = '#7F7F7F'
GNN_COLOR = '#17B3C9'
BASELINE_OVERLAP = 0.005  # baselines closer than this in AUC would hide each other ...
BASELINE_SEPARATION = 3.0  # ... so they are drawn this many points apart (a line is 1.6 pt thick)


def motif_key(motif_size: int, stringency: float, max_class_features: int, seed: int) -> str:
    return f'motifs|{motif_size}|{stringency:.4f}|{max_class_features}|{seed}'


def baseline_key(method: str, seed: int) -> str:
    return f'{method}|{seed}'


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=RESULT_COLUMNS)


def append_result(path: Path, **row):
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(path, mode='a', header=not path.exists(), index=False)


def run_baselines(config, labels: list, n_seeds: int, results_path: Path):
    """Score every baseline missing one of the seeds 0..n_seeds-1, on the dataset's graphs."""
    done = set(load_results(results_path)['key'])
    methods = [method for method in BASELINES if any(baseline_key(method, seed) not in done for seed in range(n_seeds))]
    if not methods:
        return
    graphs = read_graphs(config)
    patient_classes = load_patient_classes(config).patient_class
    labelled = patient_classes.index[patient_classes.isin(labels)]
    n_patients = graphs.Patient_uId[graphs.Patient_uId.isin(labelled)].nunique()
    for method in methods:
        name, _, analysis = BASELINES[method]
        print(f'\n=== {name} baseline ===', flush=True)
        aucs = analysis(**tissue_state_kwargs(config)).analyze(full_graph_df=graphs, cells_type=config.cells_type,
                                                              labels=labels, trials=n_seeds)
        for seed, auc in enumerate(aucs):
            if baseline_key(method, seed) not in done:
                append_result(results_path, key=baseline_key(method, seed), method=method, seed=seed, auc=auc,
                              n_patients=n_patients, status='ok')
        print(f'{name} AUC per seed: ' + ', '.join(f'{auc:.4f}' for auc in aucs), flush=True)


def motifs_auc(discriminator, labels: list, stringency: float, max_class_features: int, seed: int) -> tuple:
    """Leave-one-patient-out ROC AUC, mean number of motifs per fold, and 'ok' or 'no_features' if a fold selected
    none."""
    feature_conf = hard_discriminative_fc(labels, stringency, max_class_features)
    try:
        # one fold at a time: all folds' forests draw from the one RandomState, which parallel folds would draw from in
        # a nondeterministic order
        result = discriminator.analyze_motifs(feature_conf=feature_conf, exclude_patients=[], n_jobs=1,
                                              random_state=np.random.RandomState(seed))
    except ValueError as exc:
        if 'No discriminative features' not in str(exc):
            raise
        return np.nan, 0.0, 'no_features'
    return result.get_roc_auc_score(), float(result.results.cFeatures.mean()), 'ok'


_SWEEP = {}  # the discriminator and settings the forked sweep workers read (shared copy-on-write, not pickled)


def _sweep_point(point: tuple) -> tuple:
    stringency, seed = point
    started = time.time()
    auc, mean_features, status = motifs_auc(_SWEEP['discriminator'], _SWEEP['labels'], stringency,
                                            _SWEEP['max_class_features'], seed)
    return stringency, seed, auc, mean_features, status, time.time() - started


def run_motif_sweep(config, labels: list, motif_size: int, stringencies: list, max_class_features: int, n_seeds: int,
                    results_path: Path, n_jobs: int, workers: int = 1):
    """Score the motif size at every stringency and seed missing from the results, `workers` points at a time. Each
    point draws from its own RandomState(seed), so the AUCs do not depend on how many run at once; only this process
    writes the results csv."""
    done = set(load_results(results_path)['key'])
    todo = [(stringency, seed) for stringency in stringencies for seed in range(n_seeds)
            if motif_key(motif_size, stringency, max_class_features, seed) not in done]
    if not todo:
        return
    print(f'\n=== {motif_size}-node motifs: {len(todo)} leave-one-out runs to do, {workers} at a time ===', flush=True)
    discriminator = load_discriminator(config, motif_size, n_jobs)
    n_patients = len(discriminator.get_patients_class(labels))
    _SWEEP.update(discriminator=discriminator, labels=labels, max_class_features=max_class_features)
    if workers > 1:
        # fork: the workers share the loaded motif table instead of each loading or receiving a copy
        pool = multiprocessing.get_context('fork').Pool(workers)
        points = pool.imap_unordered(_sweep_point, todo)
    else:
        pool, points = None, map(_sweep_point, todo)
    try:
        for stringency, seed, auc, mean_features, status, seconds in points:
            append_result(results_path, key=motif_key(motif_size, stringency, max_class_features, seed),
                          method='motifs', motif_size=motif_size, stringency=stringency,
                          max_class_features=max_class_features, seed=seed, auc=auc, mean_features=mean_features,
                          n_patients=n_patients, status=status)
            shown = f'AUC {auc:.4f}, {mean_features:.1f} motifs per fold' if status == 'ok' else 'a fold selected no motif'
            print(f'[{time.strftime("%H:%M:%S")}] {motif_size}-node, stringency {stringency:g}, seed {seed}: {shown} '
                  f'({seconds:.0f} s)', flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _SWEEP.clear()


def baseline_lines(results: pd.DataFrame, gnn_aucs: tuple) -> list:
    """(name, color, mean AUC) of each dashed line: the scored baselines, chance, and the GNN when it has AUCs."""
    lines = [(name, color, results.auc[results.method == method].mean())
             for method, (name, color, _) in BASELINES.items() if (results.method == method).any()]
    lines.append(('Random', RANDOM_COLOR, 0.5))
    if gnn_aucs:
        lines.append(('GNN', GNN_COLOR, float(np.mean(gnn_aucs))))
    return lines


def plot_auc_vs_stringency(motifs: pd.DataFrame, labels: list, lines: list, selected: tuple,
                           ax: plt.Axes = None, seed_band: bool = False) -> plt.Figure:
    """motifs: the motif results; lines: (name, color, AUC) of each dashed baseline; selected: (motif size, stringency)
    of the point to box; seed_band: shade +/- 1 SD over the seeds around each motif size's line. Draws into ax when
    given (a multi-panel figure), else into a new figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
    else:
        fig = ax.figure
    # each baseline labelled just past the right edge, at the end of its line: the motif lines span the whole x range,
    # so a label inside the axes could land on one of them. Lines at (nearly) the same AUC are drawn
    # BASELINE_SEPARATION points apart, stacked symmetrically around their mean, so none hides another
    ordered = sorted(lines, key=lambda line: line[2], reverse=True)
    groups = []
    for line in ordered:
        if groups and abs(line[2] - groups[-1][0][2]) < BASELINE_OVERLAP:
            groups[-1].append(line)
        else:
            groups.append([line])
    annotations = []
    for group in groups:
        for j, (name, color, level) in enumerate(group):
            shift = ((len(group) - 1) / 2 - j) * BASELINE_SEPARATION  # points; 0 for a line on its own
            transform = offset_copy(ax.get_yaxis_transform(), fig=fig, y=shift, units='points')
            ax.plot([0, 1], [level, level], transform=transform, color=color, linewidth=1.6,
                    linestyle=(0, (2.5, 2.5)), zorder=1)
            annotations.append(ax.annotate(name, xy=(1, level), xycoords=('axes fraction', 'data'), xytext=(6, 0),
                                           textcoords='offset points', ha='left', va='center', color=color,
                                           fontsize=11, annotation_clip=False))

    curves = {}
    for size, rows in motifs.groupby('motif_size'):
        curve = curves[size] = rows.groupby('stringency').auc.agg(['mean', 'std', 'count'])
        color = MOTIF_SIZE_COLORS.get(size, 'black')
        if seed_band and (curve['count'] > 1).any():
            spread = curve['std'].fillna(0)
            ax.fill_between(curve.index, curve['mean'] - spread, curve['mean'] + spread, color=color, alpha=0.14,
                            linewidth=0, zorder=2)
        ax.plot(curve.index, curve['mean'], color=color, linewidth=2, marker='o', markersize=6, zorder=3,
                label=f'{size}-node motifs')
    # one row above the axes, clear of the data: the lines cross, so labels on them could land on another line
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=len(curves), frameon=False, fontsize=11,
              handlelength=2.2)

    size, stringency = selected
    if size in curves:
        hit = curves[size].index[np.isclose(curves[size].index, stringency)][:1]
        # small enough to stay clear of the points FINE_STEP away on either side
        ax.scatter(hit, curves[size].loc[hit, 'mean'], s=200, marker='s', facecolors='none', edgecolors='black',
                   linewidths=1.3, linestyles='--', zorder=4)

    levels = [level for _, _, level in lines] + [value for curve in curves.values() for value in curve['mean']]
    ax.set_ylim(min(levels) - 0.05, max(levels) + 0.06)
    ax.set_xlim(motifs.stringency.min() - 0.02, motifs.stringency.max() + 0.02)
    ax.set_xlabel('Discrimination stringency parameter')
    ax.set_ylabel(f'{labels[0]} vs {labels[1]} AUC')
    separate_labels(ax, annotations)
    return fig


def separate_labels(ax: plt.Axes, annotations: list, gap: float = 3):
    """Space the baseline labels (ordered top to bottom) a font height plus `gap` points apart where they would
    overlap: each group of crowded labels is stacked evenly around the mean height of its lines, so labels of lines at
    the same AUC sit one just above and one just below it. Labels with room stay at their line. Call once the y range
    is final."""
    y0, y1 = ax.get_ylim()
    points_per_unit = ax.get_window_extent().height * 72 / ax.figure.dpi / (y1 - y0)
    heights = [annotation.xy[1] * points_per_unit for annotation in annotations]
    needed = [annotation.get_fontsize() + gap for annotation in annotations]

    def stack(group):
        step = max(needed[i] for i in group)
        top = np.mean([heights[i] for i in group]) + step * (len(group) - 1) / 2
        return [top - k * step for k in range(len(group))]

    groups = [[i] for i in range(len(annotations))]
    merged = True
    while merged:  # join neighbouring groups until no two stacks overlap
        merged = False
        for k in range(len(groups) - 1):
            upper, lower = stack(groups[k]), stack(groups[k + 1])
            if upper[-1] - lower[0] < max(needed[groups[k][-1]], needed[groups[k + 1][0]]):
                groups[k] += groups.pop(k + 1)
                merged = True
                break
    for group in groups:
        for i, y in zip(group, stack(group)):
            annotations[i].xyann = (annotations[i].xyann[0], y - heights[i])


def summarize(results: pd.DataFrame, gnn_aucs: tuple):
    """Print each baseline's AUC and each motif size's best, mean +/- SD over the seeds."""
    print('\n=== ROC AUC, mean +/- SD over the seeds ===')
    for method, (name, _, _) in BASELINES.items():
        rows = results[results.method == method]
        if len(rows):
            print(f'{name:<24} {rows.auc.mean():.4f} +/- {rows.auc.std():.4f}  '
                  f'({len(rows)} seeds, {int(rows.n_patients.iloc[0])} patients)')
    if gnn_aucs:
        print(f'{"GNN":<24} {np.mean(gnn_aucs):.4f} +/- {np.std(gnn_aucs, ddof=1):.4f}  '
              f'({len(gnn_aucs)} seeds, from baselines/GNN)')
    for size, rows in results[results.method == 'motifs'].groupby('motif_size'):
        curve = rows.groupby('stringency').auc.agg(['mean', 'std'])
        best = curve['mean'].idxmax()
        print(f'{int(size)}-node motifs: best {curve.loc[best, "mean"]:.4f} +/- {curve.loc[best, "std"]:.4f} at '
              f'stringency {best:g}  ({int(rows.n_patients.iloc[0])} patients)')
    if results.n_patients.nunique() > 1:
        print(f'WARNING: the methods were scored on different numbers of patients '
              f'({sorted(results.n_patients.astype(int).unique().tolist())}), so their AUCs are not directly '
              'comparable')


def warn_if_incomplete(results: pd.DataFrame, motif_sizes: list, stringencies: list, max_class_features: int,
                       n_seeds: int):
    """A line that ends early looks just like one that ran out of discriminative motifs, so say which it is."""
    done = set(results['key'])
    for size in motif_sizes:
        missing = [f'{stringency:g}' for stringency in stringencies
                   if any(motif_key(size, stringency, max_class_features, seed) not in done for seed in range(n_seeds))]
        if missing:
            print(f'WARNING: {size}-node motifs miss seeds at stringency {", ".join(missing)}; rerun without '
                  '--plot-only to finish', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=MOTIF_DATASETS, required=True)
    parser.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
    parser.add_argument('--motif-sizes', nargs='+', type=int, default=list(MOTIF_SIZES))
    parser.add_argument('--stringencies', nargs='+', type=float, default=list(STRINGENCY_GRID),
                        help="the coarse grid, refined around the dataset's own stringency (default: 0.05, ..., 0.6)")
    parser.add_argument('--seeds', type=int, default=N_SEEDS, help=f'seeds 0..N-1 per point (default: {N_SEEDS})')
    parser.add_argument('--max-class-features', type=int, help="default: the dataset's")
    parser.add_argument('--n-jobs', type=int, default=4, help='FOVs to run FANMOD on and parse at a time')
    parser.add_argument('--workers', type=int, default=1,
                        help='stringency x seed points to score at once (each takes one CPU; same AUCs as 1)')
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from the saved results')
    parser.add_argument('--seed-band', action='store_true', help='shade +/- 1 SD over the seeds around each line')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    args = parser.parse_args()
    config = DATASETS[args.dataset]
    try:
        labels = class_labels(config, args.classes)
    except ValueError as exc:
        parser.error(str(exc))
    max_class_features = args.max_class_features or config.motif_run.max_class_features
    stringencies = sweep_stringencies(config.motif_run.stringency, args.stringencies)
    name = FIGURE_NAME + classes_tag(config, labels)
    results_path = results_dir(args.dataset) / f'{name}_results.csv'
    print(f'{config.display_name}, {labels[0]} vs {labels[1]}: results in {results_path}', flush=True)
    print('stringencies: ' + ', '.join(f'{stringency:g}' for stringency in stringencies), flush=True)

    if not args.plot_only:
        run_baselines(config, labels, args.seeds, results_path)
        for size in args.motif_sizes:
            run_motif_sweep(config, labels, size, stringencies, max_class_features, args.seeds, results_path,
                            args.n_jobs, args.workers)

    results = load_results(results_path)
    wanted = ({baseline_key(method, seed) for method in BASELINES for seed in range(args.seeds)}
              | {motif_key(size, stringency, max_class_features, seed) for size in args.motif_sizes
                 for stringency in stringencies for seed in range(args.seeds)})
    scored = results[results['key'].isin(wanted) & (results.status == 'ok')]
    # the GNN was run on the default classes only
    gnn_aucs = config.gnn_aucs if set(labels) == set(class_labels(config)) else ()
    summarize(scored, gnn_aucs)
    warn_if_incomplete(results, args.motif_sizes, stringencies, max_class_features, args.seeds)

    motifs = scored[scored.method == 'motifs'].astype({'motif_size': int})
    if motifs.empty:
        print('No motif results to plot yet')
        return
    apply_style()
    fig = plot_auc_vs_stringency(motifs, labels, baseline_lines(scored, gnn_aucs),
                                 (config.motif_run.motif_size, config.motif_run.stringency), seed_band=args.seed_band)
    for path in save_figure(fig, args.dataset, name, formats=args.formats):
        print(f'saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
