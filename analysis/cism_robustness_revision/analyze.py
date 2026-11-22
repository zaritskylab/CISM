#!/usr/bin/env python3
"""Read-only source audit and reproducible CISM uncertainty analysis.

Run with .venv/bin/python analysis/cism_robustness_revision/analyze.py.
Google documents and historical artifacts are never written.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cism-robustness-mpl')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis.noise_robustness import cism_noise_robustness as core
HIST = ROOT / 'motif_noise_summary_outputs/final_analysis'
EXTERNAL = Path('/groups/assafza_group/assafza/Omri/processed_data_23_10_2025')
THRESHOLDS = [.2, .3, .4, .42, .44, .45, .46, .48, .5, .6, .7, .8]

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def patient_bootstrap(y, scores, draws=10000, seed=20261119):
    """Stratified bootstrap of fixed held-out predictions, with half credit for ties."""
    y, scores = np.asarray(y, bool), np.asarray(scores, float)
    a, b = scores[y], scores[~y]
    if not len(a) or not len(b) or not np.isfinite(scores).all():
        raise ValueError('Both classes and finite predictions are required')
    rng = np.random.default_rng(seed)
    wa = rng.multinomial(len(a), np.ones(len(a))/len(a), size=draws)
    wb = rng.multinomial(len(b), np.ones(len(b))/len(b), size=draws)
    kernel = (a[:,None] > b[None,:]).astype(float) + .5*(a[:,None] == b[None,:])
    samples = np.einsum('bi,ij,bj->b', wa, kernel, wb, optimize=True)/(len(a)*len(b))
    return np.quantile(samples, [.025, .975]).tolist()

def summarize_historical():
    runs = [json.loads(p.read_text()) for p in sorted((HIST/'runs/raw').glob('run_*.json'))]
    assert len(runs) == 100 and {r['run_id'] for r in runs} == set(range(100))
    predictions, metrics, curves, audit = [], [], [], []
    first = runs[0]
    class_map = first['settings']['patient_classes']
    for r in runs:
        assert r['settings'] == first['settings'] and r['patients'] == first['patients']
        assert r['n_fovs'] == 76 and r['export_validation']['matched']
        paths = sorted(p for p in (Path('/groups/assafza_group/assafza/Omri/golan_output/output/parse_cache')/f"run_{r['run_id']}").rglob('*.df_cache') if p.parent.name == '4')
        signature = core.result_fingerprint({'paths': [str(p) for p in paths]}, r['settings'])
        assert len(paths) == 76 and signature == r['fingerprint'], f"Changed source: {r['run_id']}"
        audit.append(dict(run_id=r['run_id'], n_fovs=len(paths), fingerprint_matches=True, export_matches=True))
        y = np.array([class_map[p] == 'NN' for p in r['patients']])
        for curve in r['curves']:
            probabilities = r['probabilities'][str(curve['stringency'])]
            score = roc_auc_score(y, probabilities)
            if curve['rf_admissible']:
                assert np.isclose(score, curve['auc'], atol=1e-12)
            curves.append(dict(run_id=r['run_id'], **curve))
        for mode in ['fixed', 'optimal']:
            selection = r['selections'][mode]
            probs = r['probabilities'][str(selection['stringency'])]
            lo, hi = patient_bootstrap(y, probs)
            metrics.append(dict(run_id=r['run_id'], mode=mode, stringency=selection['stringency'], auc=selection['auc'], ci_low=lo, ci_high=hi, n_motifs=selection['n_motifs']))
            for patient, target, prob in zip(r['patients'], y, probs):
                predictions.append(dict(run_id=r['run_id'], mode=mode, patient=patient, label=class_map[patient], positive_class='NN', y_true=int(target), probability=prob))
    pd.DataFrame(predictions).to_csv(OUT/'historical_patient_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUT/'historical_run_metrics.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT/'historical_stringency_curves.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT/'historical_source_audit.csv', index=False)
    reference = set(first['selections']['fixed']['motif_ids'])
    all_ids = sorted(set.union(*(set(r['selections']['fixed']['motif_ids']) for r in runs)))
    recovery = []
    for mid in all_ids:
        n = sum(mid in r['selections']['fixed']['motif_ids'] for r in runs)
        recovery.append(dict(motif_id=mid, in_reference=mid in reference, selected_runs=n, recovery_percent=n))
    pd.DataFrame(recovery).sort_values('recovery_percent', ascending=False).to_csv(OUT/'motif_recovery.csv', index=False)
    summary = {}
    for mode, rows in pd.DataFrame(metrics).groupby('mode'):
        summary[mode] = dict(n=len(rows), mean_auc=rows.auc.mean(), sd_auc=rows.auc.std(ddof=1), min_auc=rows.auc.min(), max_auc=rows.auc.max(), mean_motifs=rows.n_motifs.mean(), sd_motifs=rows.n_motifs.std(ddof=1), reference_run=rows[rows.run_id==0].iloc[0].to_dict())
    summary['source_versions'] = json.loads((HIST/'software_versions.json').read_text())
    write_json(OUT/'historical_summary.json', summary)
    print('Historical audit and bootstrap complete', summary['fixed'], flush=True)

def prepare(name, paths, classes, positive, negative, motif_size, threshold, repeat_seeds):
    target = OUT/'prepared'/f'{name}.pkl'
    manifest = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    signature = hashlib.sha256(json.dumps([manifest, classes, THRESHOLDS]).encode()).hexdigest()
    if target.exists():
        with target.open('rb') as f: saved = pickle.load(f)
        if saved['signature'] == signature:
            return saved
    print('Reading', name, len(paths), 'FOV tables', flush=True)
    frame = core.load_significant_rows(paths, .05)
    # Internal binary names are required by the already validated core preparer.
    # Metadata retains the actual clinical labels and probability orientation.
    mapping = {p: ('NN' if label == positive else 'NP') for p, label in classes.items()}
    data = core.prepare_data(frame, mapping, THRESHOLDS)
    actual_patients = {p for p in frame.patient if p in classes}
    assert actual_patients == set(classes)
    assert not frame.duplicated(['patient', 'FOV', 'ID']).any()
    saved = dict(name=name, signature=signature, manifest=manifest, positive=positive, negative=negative, classes=classes, motif_size=motif_size, threshold=threshold, repeat_seeds=repeat_seeds, data=data)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as f: pickle.dump(saved, f)
    write_json(OUT/'prepared'/f'{name}_manifest.json', {k:v for k,v in saved.items() if k!='data'})
    print('Prepared', name, 'candidate motifs', len(data['ids']), flush=True)
    return saved

def fit_job(name, seed, thresholds):
    start = time.monotonic()
    with (OUT/'prepared'/f'{name}.pkl').open('rb') as f: spec = pickle.load(f)
    path = OUT/'executions'/f'{name}_seed_{seed:03d}.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['signature'] == spec['signature'] and old['thresholds'] == thresholds:
            return f'{name} seed {seed}: cached'
    records, unions, folds, probabilities, fits = core.evaluate_run(spec['data'], thresholds, 30, 100, seed)
    result = dict(name=name, signature=spec['signature'], seed=seed, thresholds=thresholds, positive=spec['positive'], negative=spec['negative'], motif_size=spec['motif_size'], patients=spec['data']['patients'], labels=[spec['classes'][p] for p in spec['data']['patients']], curves=records, probabilities={str(k):v for k,v in probabilities.items()}, motif_ids={str(k):[int(spec['data']['ids'][ix]) for ix in sorted(ids)] for k,ids in unions.items()}, folds=folds, fits=fits, seconds=time.monotonic()-start)
    write_json(path, result)
    return f'{name} seed {seed}: {fits} fits, {result["seconds"]:.1f}s'

def run_new(workers=4):
    labels = pd.read_csv(EXTERNAL/'patient_class.csv', header=None, names=['patient','label']).set_index('patient').label.to_dict()
    jobs = []
    # A direct replay of the historical reference is separate from the older
    # three-size caches, which can differ in graph preprocessing and annotations.
    r = json.loads((HIST/'runs/raw/run_000.json').read_text())
    paths = sorted(p for p in Path('/groups/assafza_group/assafza/Omri/golan_output/output/parse_cache/run_0').rglob('*.df_cache') if p.parent.name=='4')
    prepare('nn_np_raw4', paths, r['settings']['patient_classes'], 'NN', 'NP', 4, .46, True)
    jobs += [('nn_np_raw4', seed, THRESHOLDS if seed==0 else [.46]) for seed in range(30)]
    for size in [3,4,5]:
        for positive, negative, tag, fixed in [('NN','NP','nn_np',.46),('NP','PN','np_pn',.4)]:
            classes = {p:v for p,v in labels.items() if v in [positive,negative]}
            paths = sorted(p for p in (EXTERNAL/f'parse_cache/melanoma_distance_100_XX/{size}').glob('*.df_cache') if 'Melanoma'+p.name.split('_')[0] in classes)
            name = f'{tag}_size{size}'
            repeat = (tag=='np_pn' and size==5) or (tag=='nn_np' and size==4)
            prepare(name, paths, classes, positive, negative, size, fixed, repeat)
            jobs += [(name, seed, THRESHOLDS if seed==0 else [fixed]) for seed in range(30 if repeat else 1)]
    versions = {k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','pyarrow','matplotlib']}
    write_json(OUT/'execution_environment.json', dict(python=sys.version, versions=versions, seeds=list(range(30)), trees=100, workers=workers, thresholds=THRESHOLDS, selection='strict prevalence > threshold; p < .05; exclusive in training; top 30 FOV rows/class then unique IDs; feature selection within LOOCV; stable frequency ties; lexical patient order'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_job, *job) for job in jobs]
        for f in as_completed(futures): print(f.result(), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['historical','new','all'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage in ['historical','all']: summarize_historical()
    if args.stage in ['new','all']: run_new(args.workers)

if __name__ == '__main__': main()
