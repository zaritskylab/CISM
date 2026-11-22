#!/usr/bin/env python3
"""Reassess saved TNBC motifs using the updated CRC permutation evaluator."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from analysis.permutation_controls import crc_label_permutation as engine

ROOT = _REPO_ROOT
ENCODING = {'Long-term survival': 'CLR', 'Short-term survival': 'DII'}


class ArtifactState:
    """Read saved state without importing optional CISM analysis dependencies."""
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args


class StateReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(('cism.', 'CISM.')):
            return ArtifactState
        return super().find_class(module, name)


def inputs(artifact_path, label_path):
    with artifact_path.open('rb') as handle:
        artifact = StateReader(handle).load()
    cism = artifact['cism']
    if cism.motif_size != 3 or cism.iterations != 1000:
        raise ValueError('Expected saved 3-node, 1000-iteration TNBC artifact')
    raw = pd.read_csv(label_path, header=None, names=['patient', 'survival_days']).set_index('patient')
    if raw.index.has_duplicates or raw.survival_days.isna().any():
        raise ValueError('Invalid survival labels')
    cutoff = artifact['survival_rate_cutoff']
    labels = raw.survival_days.map(lambda d: 'Short-term survival' if d < cutoff else 'Long-term survival')
    saved = artifact['discriminator'].patient_class_df.patient_class
    pd.testing.assert_series_equal(labels, saved.reindex(labels.index), check_names=False)
    if set(labels.index) != set(cism.motifs_dataset.Patient_uId):
        raise ValueError('Patient mismatch between survival labels and motif data')
    if labels.value_counts().to_dict() != {'Long-term survival': 30, 'Short-term survival': 7}:
        raise ValueError('Unexpected TNBC cohort')
    # Reevaluate exactly the three candidates searched in the saved tutorial,
    # preserving trial order for the updated advancing-RNG evaluator.
    grid = artifact['tuning_trials_df'].sort_values('number').params_shared_percentage.tolist()
    return artifact, labels, grid


def run(args):
    artifact, original, grid = inputs(args.artifact, args.labels)
    fixed = float(artifact['best_feature_conf'].shared_percentage)
    if fixed not in grid:
        raise ValueError('Saved optimum absent from the saved search candidates')
    config = dict(artifact_sha256=engine.digest(args.artifact), labels_sha256=engine.digest(args.labels),
                  code_sha256=engine.digest(__file__), evaluator_sha256=engine.digest(engine.__file__),
                  reader_sha256=engine.digest(ROOT / 'analysis/crc_characterization/plot_crc_motifs_vs_stringency.py'),
                  grid=grid, fixed_stringency=fixed, forest_seed=0, trees=100, rng_mode='historical',
                  max_class_features=int(artifact['best_feature_conf'].max_class_features),
                  sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
                  positive_class='Long-term survival', internal_encoding=ENCODING,
                  survival_cutoff_days=artifact['survival_rate_cutoff'],
                  zero_features='Probability 0.5; retain all patients',
                  search='Saved three trial thresholds in trial order; matched for every shuffle',
                  rng_policy='Updated evaluator: shared RNG advances across thresholds and folds',
                  historical_auc=float(artifact['tuning_best_score']))
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    significant, matrix = engine.prepare_data(artifact['cism'].motifs_dataset, original.index)
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = [None] if args.original else range(args.seed_start, args.seed_start + args.count)
    for seed in seeds:
        directory = args.output_root / ('original' if seed is None else f'shuffled/seed_{seed}')
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'summary.json').exists():
            previous = json.loads((directory / 'summary.json').read_text())
            if previous['fingerprint'] != fingerprint or previous['seed'] != seed:
                raise ValueError(f'Incompatible completed run: {directory}')
            print(f'Already complete: {directory}', flush=True)
            continue
        labels = original.copy()
        if seed is not None:
            labels.iloc[:] = np.random.RandomState(seed).permutation(original.to_numpy())
        assert labels.value_counts().equals(original.value_counts())
        labels.rename('patient_class').to_csv(directory / 'patient_labels.csv', index_label='patient')
        engine.atomic_json(directory / 'config.json', config)
        start = time.monotonic()
        scores, predictions = engine.evaluate(significant, matrix, labels.map(ENCODING), grid, config, args.n_jobs)
        predictions.true_class = predictions.true_class.map({v: k for k, v in ENCODING.items()})
        predictions = predictions.rename(columns={'p_CLR': 'p_long_term_survival'})
        scores.to_csv(directory / 'stringency_sweep.csv', index=False)
        predictions.to_csv(directory / 'loocv_predictions.csv', index=False)
        best = scores.sort_values(['auc', 'stringency'], ascending=[False, True]).iloc[0]
        fixed_row = scores.set_index('stringency').loc[fixed]
        summary = dict(seed=seed, fingerprint=fingerprint, config=config,
                       optimal_auc=float(best.auc), optimal_stringency=float(best.stringency),
                       fixed_auc=float(fixed_row.auc), fixed_stringency=fixed,
                       optimal_no_feature_folds=int(best.no_feature_folds),
                       fixed_no_feature_folds=int(fixed_row.no_feature_folds),
                       label_hash=hashlib.sha256(labels.to_csv().encode()).hexdigest(),
                       n_patients=len(labels), class_counts=labels.value_counts().to_dict())
        engine.atomic_json(directory / 'summary.json', summary)
        print(json.dumps(dict(seed=seed, optimal_auc=summary['optimal_auc'], fixed_auc=summary['fixed_auc'],
                              seconds=round(time.monotonic()-start, 1))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'aggregate'])
    parser.add_argument('--artifact', type=Path, default=ROOT / 'tutorials/runtime/serialized/TNBC_cism_ready.pkl')
    parser.add_argument('--labels', type=Path, default=ROOT / 'pet_data/TNBC/patient_class.csv')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'analysis/outputs/tnbc_label_permutation_updated')
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--expected-shuffles', type=int, default=100)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()
    if min(args.count, args.expected_shuffles, args.n_jobs) < 1 or args.seed_start < 0:
        parser.error('Counts and workers must be positive; seeds nonnegative')
    if args.command == 'run':
        run(args)
    else:
        args.figure_reference = None
        engine.aggregate(args)
        # The common aggregator uses a CRC filename; publish dataset-specific names.
        for path in (args.output_root / 'aggregate').glob('crc_permutation_histograms.*'):
            path.rename(path.with_name(path.name.replace('crc_', 'tnbc_', 1)))


if __name__ == '__main__':
    main()
