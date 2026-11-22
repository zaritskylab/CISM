#!/usr/bin/env python3
"""Patient-label permutations of CRC 4-node CISM with matched grid-search LOOCV.

No FANMOD rerun or SHAP calculation is needed. Selection and feature values
match the notebook's hard-discriminative motif pipeline. See accompanying README.
"""
from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/crc-matplotlib')
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(ANALYSIS_DIR))
from analysis.crc_characterization.plot_crc_motifs_vs_stringency import LegacyReader

ROOT = _REPO_ROOT
GRID = [.2, .25, .3, .35, .4, .45, .5, .55, .6, .7, .75, .8]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_data(data, patients):
    significant = data.loc[data.p_value < .05].copy()
    if set(significant.Patient_uId) != set(patients):
        raise ValueError('Every labeled patient must have significant motif rows')
    if (significant.Freq <= 0).any():
        raise ValueError('Significant motif frequencies must be positive')
    significant['Total'] = significant.Count / significant.Freq
    numerator = significant.groupby(['Patient_uId', 'ID'], observed=True).Count.sum()
    denominator = significant.drop_duplicates(['Patient_uId', 'ID', 'FOV']).groupby(['Patient_uId', 'ID'], observed=True).Total.sum()
    matrix = (numerator / denominator).unstack(fill_value=0).reindex(patients)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError('Invalid motif features')
    return significant, matrix


def training_candidates(significant, labels, held_out):
    train = significant[significant.Patient_uId != held_out].copy()
    train['patient_class'] = pd.Categorical(train.Patient_uId.map(labels), categories=['CLR', 'DII'])
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


def evaluate_fold(significant, matrix, labels, held_out, thresholds, cap, forest_seed, trees):
    candidates = training_candidates(significant, labels, held_out)
    # CISM fits patients in class order, preserving source motif row order.
    train_order = [p for group in ['CLR', 'DII'] for p in
                   significant.Patient_uId.unique() if p != held_out and labels.loc[p] == group]
    cache, rows = {}, []
    for threshold in thresholds:
        features = select_ids(candidates, threshold, cap)
        key = tuple(features)
        if key not in cache:
            if not features:
                probability = .5  # Retain every held-out patient; never drop folds.
            else:
                model = RandomForestClassifier(n_estimators=trees, random_state=forest_seed, n_jobs=1)
                model.fit(matrix.loc[train_order, features].to_numpy(), labels.loc[train_order].to_numpy())
                probability = float(model.predict_proba(matrix.loc[[held_out], features].to_numpy())[0,
                                          np.flatnonzero(model.classes_ == 'CLR')[0]])
            cache[key] = probability
        rows.append({'patient': held_out, 'true_class': labels.loc[held_out], 'stringency': threshold,
                     'p_CLR': cache[key], 'n_features': len(features), 'no_features': not features})
    return rows


def evaluate(significant, matrix, labels, thresholds, config, n_jobs):
    if config.get('rng_mode') == 'historical':
        return evaluate_historical(significant, matrix, labels, thresholds, config, n_jobs)
    folds = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(evaluate_fold)(significant, matrix, labels, patient, thresholds,
                               config['max_class_features'], config['forest_seed'], config['trees'])
        for patient in labels.index)
    predictions = pd.DataFrame([row for fold in folds for row in fold])
    rows = []
    for threshold, table in predictions.groupby('stringency', sort=True):
        assert len(table) == len(labels) and table.patient.nunique() == len(labels)
        rows.append({'stringency': threshold,
                     'auc': float(roc_auc_score(table.true_class == 'CLR', table.p_CLR)),
                     'no_feature_folds': int(table.no_features.sum()),
                     'mean_features': float(table.n_features.mean())})
    return pd.DataFrame(rows), predictions


def evaluate_historical(significant, matrix, labels, thresholds, config, n_jobs):
    """Replay the notebook's shared RNG in threshold-major, patient-major order.

    Reserve each forest's RNG state serially, then fit independently in parallel.
    sklearn's forest consumes one int32 seed per tree. This preserves the exact
    serial notebook sequence regardless of worker scheduling.
    """
    def prepare_fold(patient):
        candidates = training_candidates(significant, labels, patient)
        order = [p for group in pd.unique(labels) for p in significant.Patient_uId.unique()
                 if p != patient and labels.loc[p] == group]
        features = {t: select_ids(candidates, t, config['max_class_features']) for t in thresholds}
        return patient, order, features

    prepared = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(prepare_fold)(patient) for patient in labels.index)
    rng = np.random.RandomState(config['forest_seed'])
    tasks = []
    for threshold in thresholds:
        for patient, order, selections in prepared:
            features = selections[threshold]
            state = rng.get_state() if features else None
            if features:
                rng.randint(np.iinfo(np.int32).max, size=config['trees'])
            tasks.append((patient, order, features, threshold, state))

    def fit(patient, order, features, threshold, state):
        probability = .5
        if features:
            local_rng = np.random.RandomState()
            local_rng.set_state(state)
            model = RandomForestClassifier(n_estimators=config['trees'], random_state=local_rng, n_jobs=1)
            model.fit(matrix.loc[order, features].to_numpy(), labels.loc[order].to_numpy())
            probability = float(model.predict_proba(matrix.loc[[patient], features].to_numpy())[0,
                                      np.flatnonzero(model.classes_ == 'CLR')[0]])
        return {'patient': patient, 'true_class': labels.loc[patient], 'stringency': threshold,
                'p_CLR': probability, 'n_features': len(features), 'no_features': not features}

    predictions = pd.DataFrame(Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(fit)(*task) for task in tasks))
    rows = []
    for threshold, table in predictions.groupby('stringency', sort=True):
        assert len(table) == len(labels) and table.patient.nunique() == len(labels)
        rows.append({'stringency': threshold,
                     'auc': float(roc_auc_score(table.true_class == 'CLR', table.p_CLR)),
                     'no_feature_folds': int(table.no_features.sum()),
                     'mean_features': float(table.n_features.mean())})
    return pd.DataFrame(rows), predictions


def run(args):
    if args.n_jobs < 1 or args.trees < 1 or args.max_class_features < 1:
        raise ValueError('Workers, trees, and feature cap must be positive')
    grid = sorted(set(args.grid))
    if not grid or not all(0 <= t <= 1 for t in grid + [args.fixed_stringency]):
        raise ValueError('Stringencies must be in [0,1]')
    with args.artifact.open('rb') as handle:
        artifact = LegacyReader(handle).load()
    if artifact.motif_size != 4:
        raise ValueError('This experiment expects the CRC 4-node artifact')
    raw = pd.read_csv(args.labels)[['Patient', 'Group']].dropna().copy()
    raw.Patient = raw.Patient.astype(int)
    raw.Group = raw.Group.astype(int)
    if raw.Patient.duplicated().any() or not raw.Group.isin([1, 2]).all():
        raise ValueError('Labels must be unique patients with groups 1=CLR, 2=DII')
    original = raw.set_index(raw.Patient.map(lambda p: f'CRC{p}')).Group.map({1: 'CLR', 2: 'DII'})
    if set(original.index) != set(artifact.motifs_dataset.Patient_uId):
        raise ValueError('Artifact patients and label patients differ')
    if original.value_counts().to_dict() != {'DII': 18, 'CLR': 17}:
        raise ValueError('Expected the 35-patient CRC cohort (17 CLR, 18 DII)')
    config = {'artifact_sha256': digest(args.artifact), 'labels_sha256': digest(args.labels),
              'code_sha256': digest(__file__), 'grid': grid, 'fixed_stringency': args.fixed_stringency,
              'forest_seed': args.forest_seed, 'trees': args.trees, 'rng_mode': args.rng_mode,
              'max_class_features': args.max_class_features, 'sklearn': sklearn.__version__,
              'numpy': np.__version__, 'pandas': pd.__version__,
              'positive_class': 'CLR', 'zero_features': 'Constant p_CLR=0.5, all patients retained',
              'tie_break': 'Lowest stringency among equal maximum AUCs'}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    significant, matrix = prepare_data(artifact.motifs_dataset, original.index)
    thresholds = sorted(set(grid + [args.fixed_stringency]))
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
            fixed = scores.set_index('stringency').loc[args.fixed_stringency]
            summary = {'seed': seed, 'fingerprint': fingerprint, 'config': config,
                       'optimal_auc': float(best.auc), 'optimal_stringency': float(best.stringency),
                       'fixed_auc': float(fixed.auc), 'fixed_stringency': args.fixed_stringency,
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


def aggregate(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    original = json.loads((args.output_root / 'original/summary.json').read_text())
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
    pd.DataFrame(tests).to_csv(out / 'empirical_p_values.csv', index=False)
    panels = [('optimal_auc', 'Optimized AUC', '#7698ca'),
              ('optimal_stringency', 'Optimal stringency', '#8a9e74'),
              ('fixed_auc', f"AUC at fixed stringency {original['fixed_stringency']:g}", '#24558c')]
    figure_reference = original
    reference_path = getattr(args, 'figure_reference', None)
    if reference_path is not None:
        figure_reference = json.loads(Path(reference_path).read_text())
        if figure_reference['fixed_stringency'] != original['fixed_stringency']:
            raise ValueError('Figure reference must use the same fixed stringency')
    figure_tests = []
    for metric in ['optimal_auc', 'fixed_auc']:
        exceedances = int((frame[metric] >= figure_reference[metric]).sum())
        figure_tests.append({'metric': metric, 'reference': figure_reference[metric],
                             'n_greater_equal': exceedances,
                             'empirical_p': (exceedances + 1) / (len(frame) + 1)})
    atomic_json(out / 'figure_provenance.json', {
        'reference_source': str(reference_path) if reference_path else 'original/summary.json',
        'reference_values': {k: figure_reference[k] for k in ['optimal_auc', 'optimal_stringency', 'fixed_auc', 'fixed_stringency']},
        'permutations': 'Existing completed permutations, unchanged',
        'comparisons': figure_tests,
        'note': ('Historical reference uses advancing RNG; completed permutations use per-fit seed 0. '
                 'Exceedance comparisons reuse that existing null distribution; RNG settings are not matched.'
                 if reference_path else 'Original and permutations have matched configurations.')})
    with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 10}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        for ax, (metric, title, color) in zip(axes, panels):
            if metric == 'optimal_stringency':
                frequencies = frame[metric].value_counts().sort_index()
                ax.bar(frequencies.index, frequencies.values, width=.035,
                       color=color, edgecolor='white', linewidth=.6)
            else:
                ax.hist(frame[metric], bins=np.linspace(0, 1, 21), color=color, edgecolor='white', linewidth=.6)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            reference_label = 'Original'
            ax.axvline(figure_reference[metric], color='#df5948', lw=1.5,
                       label=f"{reference_label} = {figure_reference[metric]:.3f}")
            if metric == 'optimal_stringency' and not np.isclose(original[metric], original['fixed_stringency']):
                ax.axvline(original['fixed_stringency'], color='#555555', ls='--', lw=1,
                           label=f"Notebook optimum = {original['fixed_stringency']:g}")
            ax.set(xlim=(0, 1), xlabel=title, ylabel='Number of shuffles')
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False, fontsize=7)
            if metric != 'optimal_stringency':
                test = next(t for t in figure_tests if t['metric'] == metric)
                test_label = 'Exceedance p' if reference_path else 'Permutation p'
                ax.set_title(f"{test_label} = {test['empirical_p']:.4g}", fontsize=10)
        fig.tight_layout()
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'crc_permutation_histograms.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    atomic_json(out / 'experiment_summary.json', {'original': original, 'tests': tests,
                'n_shuffles': len(frame), 'unique_labelings': int(frame.label_hash.nunique()),
                'note': 'AUC maxima are over the prespecified grid, not a continuous or nested-CV optimum.'})
    print(pd.DataFrame(tests).to_string(index=False))
    print(f'Plots and tables saved to {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--artifact', type=Path, default=Path('/home/omriavit/research/cism_CRC_4.pkl'))
    run_parser.add_argument('--labels', type=Path, default=Path('/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv'))
    run_parser.add_argument('--original', action='store_true')
    run_parser.add_argument('--seed-start', type=int, default=0)
    run_parser.add_argument('--count', type=int, default=10)
    run_parser.add_argument('--n-jobs', type=int, default=4)
    run_parser.add_argument('--forest-seed', type=int, default=0)
    run_parser.add_argument('--rng-mode', choices=['historical', 'per-fit'], default='historical')
    run_parser.add_argument('--trees', type=int, default=100)
    run_parser.add_argument('--max-class-features', type=int, default=30)
    run_parser.add_argument('--grid', nargs='+', type=float, default=GRID)
    run_parser.add_argument('--fixed-stringency', type=float, default=.55)
    agg_parser = sub.add_parser('aggregate')
    agg_parser.add_argument('--expected-shuffles', type=int, default=100)
    agg_parser.add_argument('--seed-start', type=int, default=0)
    agg_parser.add_argument('--figure-reference', type=Path,
                            help='Display a historical original summary; keep completed permutation values unchanged')
    for p in [run_parser, agg_parser]:
        p.add_argument('--output-root', type=Path, default=ROOT / 'analysis/outputs/crc_label_permutation_historical')
    args = parser.parse_args()
    if args.seed_start < 0 or (args.command == 'run' and args.count < 1):
        parser.error('Seed start must be nonnegative and count positive')
    (run if args.command == 'run' else aggregate)(args)


if __name__ == '__main__':
    main()
