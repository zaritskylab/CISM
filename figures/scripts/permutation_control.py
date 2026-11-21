"""Patient-label permutation control: CISM's LOOCV AUC on the real labels against shuffled labels.

Each labeling (the real one with --original, or one shuffle per seed) reruns the hard-discriminative motif pipeline of
TissueStateDiscriminativeMotifs.analyze_motifs over the stringencies the CISM run searched: motifs are selected inside
every leave-one-out fold and a random forest is fit on the patients' motif frequencies. Motifs come from a serialized
CISM run (tutorial 02's <dataset>_cism_ready.pkl), so FANMOD is not rerun. `aggregate` compares the real optimized AUC
with the shuffles and draws two figures: histograms of the shuffles' best AUC (A) and of the stringency that reached it
(B), each marking the original's (permutation_control), and their joint distribution, every shuffle's optimal stringency and AUC against the original's
(permutation_control_joint), with the number of shuffles that match or exceed the original on both.

Two classes at a time: the dataset's default pair (NN vs NP for Melanoma), or the two given with --classes, which
`aggregate` then needs too.

Usage, from the repo root:
    python figures/scripts/permutation_control.py run --dataset tnbc --artifact TNBC_cism_ready.pkl --original
    python figures/scripts/permutation_control.py run --dataset tnbc --artifact TNBC_cism_ready.pkl --seed-start 0 --count 10
    python figures/scripts/permutation_control.py aggregate --dataset tnbc
    python figures/scripts/permutation_control.py run --dataset melanoma --classes NN PN --artifact <run>_cism_ready.pkl --original
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figures.helpers.artifacts import load_artifact
from figures.helpers.datasets import (DATASETS, class_labels, classes_tag, load_patient_classes,
                                      tissue_state_kwargs)
from figures.helpers.style import GRID_COLOR, RESULTS_DIR, save_figure

FIGURE_NAME = 'permutation_control'
LABELED_DATASETS = [key for key, config in DATASETS.items()
                    if config.survival is not None or config.patient_class_csv is not None]
MOTIF_COLUMNS = ['Patient', 'FOV', 'ID', 'Freq', 'Count', 'p_value', 'nunique_colors']
# where each forest's random state starts:
#   historical     - one RandomState(seed) advancing over every stringency and fold (crc/tnbc_label_permutation.py)
#   per-stringency - a fresh RandomState(seed) per stringency, advancing over the folds (tune_stringency with n_jobs=1)
#   per-fit        - every forest seeded with seed (tune_stringency with worker processes)
RNG_MODES = ['historical', 'per-stringency', 'per-fit']

AUC_COLOR = '#7698CA'
STRINGENCY_COLOR = '#8A9E74'
ORIGINAL_COLOR = '#DF5948'
FIXED_COLOR = '#555555'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_run(artifact_path, patient_uid_prefix):
    """Motif rows and saved settings of a serialized CISM run.

    Patient_uId is re-keyed the way the dataset's patient class csv keys patients. The tuned feature configuration,
    the tuning trials and the run's own patient classes are None when the artifact does not hold them.
    """
    artifact = load_artifact(artifact_path)
    saved = artifact if isinstance(artifact, dict) else {'cism': artifact}
    motifs_dataset = saved['cism'].motifs_dataset
    patient_uids = patient_uid_prefix + motifs_dataset['Patient'].astype(str)
    uid_map = dict(zip(motifs_dataset['Patient_uId'].astype(str), patient_uids))
    motifs = motifs_dataset[MOTIF_COLUMNS].copy()
    motifs['Patient_uId'] = patient_uids
    discriminator = saved.get('discriminator')
    saved_classes = None
    if discriminator is not None:
        saved_classes = discriminator.patient_class_df['patient_class'].rename(index=lambda uid: uid_map.get(uid, uid))
    return {'motifs': motifs, 'motif_size': int(saved['cism'].motif_size), 'tuned': saved.get('best_feature_conf'),
            'trials': saved.get('tuning_trials_df'), 'saved_classes': saved_classes}


def check_saved_classes(labels, saved_classes):
    """The CISM run must have split these patients into the same groups (its class names may differ)."""
    saved = saved_classes.reindex(labels.index)
    pairs = pd.crosstab(labels, saved) > 0
    if saved.isna().any() or (pairs.sum(axis=0) != 1).any() or (pairs.sum(axis=1) != 1).any():
        raise ValueError(f'Patient classes differ from the ones saved in the artifact:\n{pd.crosstab(labels, saved)}')


def prepare_data(data, patients):
    significant = data.loc[data.p_value < .05].copy()
    missing = set(patients) - set(significant.Patient_uId)
    if missing:
        raise ValueError(f'Every labeled patient must have significant motif rows; missing: {sorted(missing)}')
    if (significant.Freq <= 0).any():
        raise ValueError('Significant motif frequencies must be positive')
    significant = significant[significant.Patient_uId.isin(patients)].copy()
    significant['Total'] = significant.Count / significant.Freq
    numerator = significant.groupby(['Patient_uId', 'ID'], observed=True).Count.sum()
    denominator = significant.drop_duplicates(['Patient_uId', 'ID', 'FOV']).groupby(['Patient_uId', 'ID'], observed=True).Total.sum()
    matrix = (numerator / denominator).unstack(fill_value=0).reindex(patients)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError('Invalid motif features')
    return significant, matrix


def training_candidates(significant, labels, classes, held_out):
    train = significant[significant.Patient_uId != held_out].copy()
    train['patient_class'] = pd.Categorical(train.Patient_uId.map(labels), categories=classes)
    denominator = train.groupby('patient_class', observed=True).Patient_uId.nunique()
    nclasses = train.groupby('ID', observed=True).patient_class.nunique()
    train = train[train.ID.isin(nclasses[nclasses == 1].index) & (train.nunique_colors >= 1)].copy()
    train['patient_count'] = train.groupby('ID', observed=True).Patient_uId.transform('nunique')
    train['prevalence'] = train.patient_count / train.patient_class.astype(str).map(denominator)
    return train


def select_ids(candidates, threshold, cap):
    ranked = candidates[candidates.prevalence > threshold].sort_values('Freq', ascending=False)
    # Exact historical CISM behavior: cap rows, then deduplicate motif IDs.
    return ranked.groupby('patient_class', observed=True).head(cap).ID.unique().tolist()


def evaluate(significant, matrix, labels, thresholds, config, n_jobs):
    classes, positive = config['classes'], config['positive_class']

    def prepare_fold(patient):
        candidates = training_candidates(significant, labels, classes, patient)
        # CISM fits patients class by class, in the labels' class order, preserving source motif row order.
        order = [p for group in pd.unique(labels) for p in significant.Patient_uId.unique()
                 if p != patient and labels.loc[p] == group]
        features = {t: select_ids(candidates, t, config['max_class_features']) for t in thresholds}
        return patient, order, features

    prepared = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(prepare_fold)(patient) for patient in labels.index)

    # Reserve each forest's RNG state serially, stringency by stringency in search order, then patient by patient,
    # and fit independently in parallel. sklearn's forest consumes one int32 seed per tree, so this preserves the
    # serial sequence regardless of worker scheduling.
    rng = np.random.RandomState(config['forest_seed'])
    tasks = []
    for threshold in thresholds:
        if config['rng_mode'] == 'per-stringency':
            rng = np.random.RandomState(config['forest_seed'])
        for patient, order, selections in prepared:
            if config['rng_mode'] == 'per-fit':
                rng = np.random.RandomState(config['forest_seed'])
            features = selections[threshold]
            state = rng.get_state() if features else None
            if features:
                rng.randint(np.iinfo(np.int32).max, size=config['trees'])
            tasks.append((patient, order, features, threshold, state))

    def fit(patient, order, features, threshold, state):
        probability = .5  # Retain every held-out patient; never drop folds.
        if features:
            local_rng = np.random.RandomState()
            local_rng.set_state(state)
            model = RandomForestClassifier(n_estimators=config['trees'], random_state=local_rng, n_jobs=1)
            model.fit(matrix.loc[order, features].to_numpy(), labels.loc[order].to_numpy())
            probability = float(model.predict_proba(matrix.loc[[patient], features].to_numpy())[0,
                                      np.flatnonzero(model.classes_ == positive)[0]])
        return {'patient': patient, 'true_class': labels.loc[patient], 'stringency': threshold,
                'p_positive': probability, 'n_features': len(features), 'no_features': not features}

    predictions = pd.DataFrame(Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(fit)(*task) for task in tasks))
    rows = []
    for threshold, table in predictions.groupby('stringency', sort=True):
        assert len(table) == len(labels) and table.patient.nunique() == len(labels)
        rows.append({'stringency': threshold,
                     'auc': float(roc_auc_score(table.true_class == positive, table.p_positive)),
                     'no_feature_folds': int(table.no_features.sum()),
                     'mean_features': float(table.n_features.mean())})
    return pd.DataFrame(rows), predictions


def run(args):
    if args.n_jobs < 1 or args.trees < 1 or (args.max_class_features is not None and args.max_class_features < 1):
        raise ValueError('Workers, trees, and feature cap must be positive')
    dataset = DATASETS[args.dataset]
    classes = args.classes

    saved = load_run(args.artifact, dataset.patient_uid_prefix)
    fixed_stringency = args.fixed_stringency if args.fixed_stringency is not None else getattr(saved['tuned'], 'shared_percentage', None)
    if fixed_stringency is None:
        raise ValueError('The artifact has no tuned stringency; pass --fixed-stringency')
    max_class_features = args.max_class_features or getattr(saved['tuned'], 'max_class_features', 30)
    if args.grid:
        grid = args.grid
    elif saved['trials'] is not None:
        # the stringencies the CISM run's tuning searched, in trial order, so every shuffle repeats the same search
        grid = saved['trials'].sort_values('number').params_shared_percentage.tolist()
    else:
        raise ValueError('The artifact has no tuning trials; pass the stringencies to search with --grid')
    grid = [float(t) for t in dict.fromkeys(grid)]
    if not all(0 <= t <= 1 for t in grid + [fixed_stringency]):
        raise ValueError('Stringencies must be in [0,1]')

    patient_classes = load_patient_classes(dataset)['patient_class']
    original = patient_classes[patient_classes.isin(classes)]
    missing = sorted(set(original.index) - set(saved['motifs'].Patient_uId))
    if missing:
        raise ValueError(f'Labeled patients missing from the artifact: {missing}')
    if set(original) != set(classes):
        raise ValueError(f'Expected patients of both classes {classes}, found {original.value_counts().to_dict()}')
    if saved['saved_classes'] is not None:
        check_saved_classes(original, saved['saved_classes'])

    labels_path = tissue_state_kwargs(dataset)['tissue_state_csv_path']
    config = {'dataset': args.dataset, 'artifact_sha256': digest(args.artifact), 'labels_sha256': digest(labels_path),
              'code_sha256': digest(__file__), 'motif_size': saved['motif_size'], 'classes': list(classes),
              # the alphabetically first class, sklearn's classes_[0] (CLR for CRC, Long-term survival for TNBC)
              'positive_class': sorted(classes)[0], 'grid': grid, 'fixed_stringency': float(fixed_stringency),
              'forest_seed': args.forest_seed, 'trees': args.trees, 'rng_mode': args.rng_mode,
              'max_class_features': int(max_class_features), 'sklearn': sklearn.__version__,
              'numpy': np.__version__, 'pandas': pd.__version__,
              'zero_features': 'Constant p_positive=0.5, all patients retained',
              'tie_break': 'Lowest stringency among equal maximum AUCs'}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    significant, matrix = prepare_data(saved['motifs'], original.index)
    # search order matters for the advancing RNG modes
    thresholds = grid + ([config['fixed_stringency']] if config['fixed_stringency'] not in grid else [])
    failed = []
    for seed in ([None] if args.original else range(args.seed_start, args.seed_start + args.count)):
        directory = args.output_root / ('original' if seed is None else f'shuffled/seed_{seed}')
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / 'summary.json'
        if summary_path.exists():
            old = json.loads(summary_path.read_text())
            if old['fingerprint'] != fingerprint or old['seed'] != seed:
                raise ValueError(f'Incompatible completed run: {directory}; use a new output root')
            print(f'Already complete: {directory}', flush=True)
            continue
        print(f'Starting {directory}', flush=True)
        labels = original.copy()
        if seed is not None:
            labels.iloc[:] = np.random.RandomState(seed).permutation(original.to_numpy())
        assert labels.value_counts().equals(original.value_counts())
        labels.rename('patient_class').to_csv(directory / 'patient_labels.csv', index_label='patient')
        atomic_json(directory / 'config.json', config)
        try:
            scores, predictions = evaluate(significant, matrix, labels, thresholds, config, args.n_jobs)
            scores.to_csv(directory / 'stringency_sweep.csv', index=False)
            predictions.to_csv(directory / 'loocv_predictions.csv', index=False)
            search = scores[scores.stringency.isin(grid)]
            best = search.sort_values(['auc', 'stringency'], ascending=[False, True]).iloc[0]
            fixed = scores.set_index('stringency').loc[config['fixed_stringency']]
            summary = {'seed': seed, 'fingerprint': fingerprint, 'config': config,
                       'optimal_auc': float(best.auc), 'optimal_stringency': float(best.stringency),
                       'fixed_auc': float(fixed.auc), 'fixed_stringency': config['fixed_stringency'],
                       'optimal_no_feature_folds': int(best.no_feature_folds),
                       'fixed_no_feature_folds': int(fixed.no_feature_folds),
                       'label_hash': hashlib.sha256(labels.to_csv().encode()).hexdigest(),
                       'n_patients': len(labels), 'class_counts': labels.value_counts().to_dict()}
            atomic_json(summary_path, summary)  # Completion marker written last.
            print(json.dumps({k: v for k, v in summary.items() if k != 'config'}), flush=True)
        except Exception as exc:
            atomic_json(directory / 'failure.json', {'error': repr(exc)})
            failed.append(seed)
            print(f'FAILED seed {seed}: {exc}', file=sys.stderr, flush=True)
    if failed:
        raise RuntimeError(f'Failed seeds: {failed}')


def plot_permutation_control(frame, original) -> plt.Figure:
    """Histograms over the shuffles: (A) best AUC, (B) the stringency that reached it, each with the original's.

    Stringencies are counted in 0.05-wide bins centred on multiples of 0.05, so a 0.05-step grid keeps one bar per
    value and a continuous (Optuna) grid does not draw overlapping bars."""
    panels = [('A', 'optimal_auc', 'Best AUC', AUC_COLOR),
              ('B', 'optimal_stringency', 'Corresponding stringency parameter', STRINGENCY_COLOR)]
    with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 10}):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
        for ax, (letter, metric, xlabel, color) in zip(axes, panels):
            if metric == 'optimal_stringency':
                frequencies = (frame[metric] / .05).round().mul(.05).round(2).value_counts().sort_index()
                ax.bar(frequencies.index, frequencies.values, width=.035,
                       color=color, edgecolor='white', linewidth=.6)
            else:
                ax.hist(frame[metric], bins=np.linspace(0, 1, 21), color=color, edgecolor='white', linewidth=.6)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.axvline(original[metric], color=ORIGINAL_COLOR, lw=1.5, label=f'Original = {original[metric]:.3f}')
            ax.set(xlim=(0, 1), xlabel=xlabel, ylabel='Number of shuffles')
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False, fontsize=7)
            ax.text(-.18, 1.04, letter, transform=ax.transAxes, fontsize=14, va='bottom', ha='left')
        fig.tight_layout()

    return fig


def joint_exceedances(frame, original) -> int:
    """Shuffles whose optimized AUC and optimal stringency both match or exceed the original's."""
    return int(((frame.optimal_auc >= original['optimal_auc'])
                & (frame.optimal_stringency >= original['optimal_stringency'])).sum())


def plot_joint_distribution(frame, original) -> plt.Figure:
    """Each shuffle's optimal stringency and optimized AUC, with the original labeling's on top. Shuffles can only land
    on the searched stringencies, so several can share a point (e.g. AUC 0.5 when folds select no motif): such a point
    is labeled with its number of shuffles."""
    with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 10}):
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        ax.scatter(frame.optimal_stringency, frame.optimal_auc, s=22, color=AUC_COLOR, alpha=.85, linewidths=0,
                   label='Shuffled permutations', zorder=2)
        stacks = frame.groupby(['optimal_stringency', 'optimal_auc']).size()
        for (stringency, auc), count in stacks[stacks > 1].items():
            ax.annotate(f'×{count}', (stringency, auc), xytext=(4, 3), textcoords='offset points', fontsize=6.5,
                        color=FIXED_COLOR, zorder=4)
        ax.scatter([original['optimal_stringency']], [original['optimal_auc']], s=60, marker='D',
                   color=ORIGINAL_COLOR, edgecolor='#7A2318', linewidths=.8, label='Original', zorder=3)
        ax.grid(color=GRID_COLOR, linewidth=.6, zorder=0)
        ax.set(xlabel='Optimal stringency', ylabel='Optimal AUC')
        # above the axes, where it cannot cover a point: the original often sits in a corner
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, frameon=False, fontsize=8, handletextpad=.3)
        fig.tight_layout()

    return fig


def aggregate(args):
    original = json.loads((args.output_root / 'original/summary.json').read_text())
    if sorted(original['config']['classes']) != sorted(args.classes):
        raise ValueError(f"{args.output_root} compares {original['config']['classes']}, not {args.classes}")
    records, missing = [], []
    for seed in range(args.seed_start, args.seed_start + args.expected_shuffles):
        path = args.output_root / f'shuffled/seed_{seed}/summary.json'
        if not path.exists():
            missing.append(seed)
            continue
        item = json.loads(path.read_text())
        if item['fingerprint'] != original['fingerprint'] or item['seed'] != seed:
            raise ValueError(f'Incompatible configuration or seed: {path}')
        records.append(item)
    if missing:
        raise ValueError(f'Missing {len(missing)} shuffles; seeds: {missing}')
    if not records:
        raise ValueError('No permutations available')
    frame = pd.DataFrame([{k: v for k, v in item.items() if k != 'config'} for item in records])
    out = args.output_root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'permutation_results.csv', index=False)
    tests = []
    for metric in ['optimal_auc', 'fixed_auc']:
        exceedances = int((frame[metric] >= original[metric]).sum())
        tests.append({'metric': metric, 'original': original[metric],
                      'permutation_mean': float(frame[metric].mean()),
                      'permutation_sd': float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0,
                      'n_shuffles': len(frame), 'n_greater_equal': exceedances,
                      'empirical_p': (exceedances + 1) / (len(frame) + 1)})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / 'empirical_p_values.csv', index=False)
    both = joint_exceedances(frame, original)
    joint = {'metrics': ['optimal_auc', 'optimal_stringency'], 'n_shuffles': len(frame),
             'n_both_greater_equal': both, 'empirical_p': (both + 1) / (len(frame) + 1)}
    atomic_json(out / 'experiment_summary.json', {'original': original, 'tests': tests.to_dict('records'),
                'joint_test': joint, 'n_shuffles': len(frame), 'unique_labelings': int(frame.label_hash.nunique()),
                'note': 'AUC maxima are over the searched stringencies, not a continuous or nested-CV optimum.'})
    print(tests.to_string(index=False))
    print(f"{both} of {len(frame)} shuffles match or exceed the original on both optimized AUC "
          f"({original['optimal_auc']:.3f}) and optimal stringency ({original['optimal_stringency']:.3g}); "
          f"empirical p = {joint['empirical_p']:.4g}")
    print(f'Tables saved to {out}')

    name = FIGURE_NAME + classes_tag(DATASETS[args.dataset], args.classes)
    figures = {name: plot_permutation_control(frame, original),
               f'{name}_joint': plot_joint_distribution(frame, original)}
    for figure_name, fig in figures.items():
        for path in save_figure(fig, args.dataset, figure_name, formats=args.formats):
            print(f'saved {path}')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--artifact', type=Path, required=True,
                            help="serialized CISM run, e.g. tutorial 02's <dataset>_cism_ready.pkl")
    run_parser.add_argument('--original', action='store_true')
    run_parser.add_argument('--seed-start', type=int, default=0)
    run_parser.add_argument('--count', type=int, default=10)
    run_parser.add_argument('--n-jobs', type=int, default=4)
    run_parser.add_argument('--forest-seed', type=int, default=0)
    run_parser.add_argument('--rng-mode', choices=RNG_MODES, default='historical')
    run_parser.add_argument('--trees', type=int, default=100)
    run_parser.add_argument('--max-class-features', type=int, help="default: the artifact's tuned value, else 30")
    run_parser.add_argument('--grid', nargs='+', type=float,
                            help="stringencies to search, in order (default: the artifact's tuning trials)")
    run_parser.add_argument('--fixed-stringency', type=float, help="default: the artifact's tuned stringency")
    agg_parser = sub.add_parser('aggregate')
    agg_parser.add_argument('--expected-shuffles', type=int, default=100)
    agg_parser.add_argument('--seed-start', type=int, default=0)
    agg_parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    for p in [run_parser, agg_parser]:
        p.add_argument('--dataset', choices=LABELED_DATASETS, required=True)
        p.add_argument('--classes', nargs=2, help="the two classes to compare (default: the dataset's default pair)")
        p.add_argument('--output-root', type=Path,
                       help='default: figures/results/<dataset>/<dataset>[_<A>_vs_<B>]_label_permutation')
    args = parser.parse_args()
    if args.seed_start < 0 or (args.command == 'run' and args.count < 1):
        parser.error('Seed start must be nonnegative and count positive')
    dataset = DATASETS[args.dataset]
    try:
        args.classes = class_labels(dataset, args.classes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_root is None:
        args.output_root = (RESULTS_DIR / args.dataset
                            / f'{args.dataset}{classes_tag(dataset, args.classes)}_label_permutation')
    (run if args.command == 'run' else aggregate)(args)


if __name__ == '__main__':
    main()
