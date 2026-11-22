#!/usr/bin/env python3
"""Seed repetitions for TNBC Fig. 8A; saved and literal caption thresholds.

Both settings use matched per-fold forest states from the saved third trial,
so their difference isolates features selected at the two fixed thresholds.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from analyze import ROOT,OUT,patient_bootstrap,write_json
sys.path.insert(0,str(ROOT/'analysis'))
from analysis.permutation_controls import tnbc_label_permutation as tnbc
from analysis.permutation_controls import crc_label_permutation as engine

DEST=OUT/'tnbc'
ARTIFACT=ROOT/'tutorials/runtime/serialized/TNBC_cism_ready.pkl'
LABELS=ROOT/'pet_data/TNBC/patient_class.csv'
BASE=ROOT/'analysis/outputs/tnbc_label_permutation_updated/original'

def prepare():
    DEST.mkdir(exist_ok=True)
    artifact,actual,grid=tnbc.inputs(ARTIFACT,LABELS)
    baseline=json.loads((BASE/'summary.json').read_text())
    assert engine.digest(ARTIFACT)==baseline['config']['artifact_sha256']
    assert engine.digest(LABELS)==baseline['config']['labels_sha256']
    saved=float(artifact['best_feature_conf'].shared_percentage)
    assert saved==grid[-1]
    thresholds={'saved':saved,'figure_8a':.45}
    labels=actual.map(tnbc.ENCODING)
    significant,matrix=engine.prepare_data(artifact['cism'].motifs_dataset,labels.index)
    cap=int(artifact['best_feature_conf'].max_class_features)
    folds=[];unions={name:set() for name in thresholds}
    for patient in labels.index:
        candidates=engine.training_candidates(significant,labels,patient)
        selections={th:engine.select_ids(candidates,th,cap) for th in grid+[.45]}
        order=[p for group in pd.unique(labels) for p in significant.Patient_uId.unique() if p!=patient and labels.loc[p]==group]
        settings={}
        for name,th in thresholds.items():
            ids=selections[th]
            assert ids, (patient,th)
            unions[name].update(ids)
            settings[name]=dict(ids=list(map(int,ids)),x_train=matrix.loc[order,ids].to_numpy(),y_train=labels.loc[order].to_numpy(),x_test=matrix.loc[[patient],ids].to_numpy())
        folds.append(dict(patient=patient,label=actual.loc[patient],has_features={th:bool(selections[th]) for th in grid},settings=settings))
    prepared=dict(folds=folds,grid=grid,saved=saved,thresholds=thresholds,n_motifs={name:len(ids) for name,ids in unions.items()})
    with (DEST/'prepared.pkl').open('wb') as f:pickle.dump(prepared,f)
    write_json(DEST/'source_manifest.json',dict(artifact=str(ARTIFACT),labels=str(LABELS),artifact_sha256=engine.digest(ARTIFACT),labels_sha256=engine.digest(LABELS),script_sha256=engine.digest(__file__),adapter_sha256=engine.digest(tnbc.__file__),evaluator_sha256=engine.digest(engine.__file__),figure='8A (formerly 7A), TNBC',motif_size=3,thresholds=thresholds,n_motifs=prepared['n_motifs'],survival_cutoff_days=1000,positive_class='Long-term survival',n_long=30,n_short=7,trees=100,seeds=list(range(30)),saved_trial_grid=grid,rng_policy='Advancing historical RNG across saved trial grid; use third-trial per-fold states for both thresholds to isolate threshold sensitivity',fanmod_repetitions=False,documents_edited=False))
    print('Prepared TNBC',prepared['n_motifs'],flush=True)

def run(seed):
    with (DEST/'prepared.pkl').open('rb') as f:prepared=pickle.load(f)
    rng=np.random.RandomState(seed);states={}
    for th in prepared['grid']:
        for fold in prepared['folds']:
            if fold['has_features'][th]:
                if th==prepared['saved']:states[fold['patient']]=rng.get_state()
                rng.randint(np.iinfo(np.int32).max,size=100)
        if th==prepared['saved']:break
    outputs={}
    for name,th in prepared['thresholds'].items():
        rows=[]
        for fold in prepared['folds']:
            data=fold['settings'][name]
            local=np.random.RandomState();local.set_state(states[fold['patient']])
            clf=RandomForestClassifier(n_estimators=100,random_state=local,n_jobs=1)
            clf.fit(data['x_train'],data['y_train'])
            prob=float(clf.predict_proba(data['x_test'])[0,np.flatnonzero(clf.classes_=='CLR')[0]])
            rows.append(dict(seed=seed,setting=name,stringency=th,patient=fold['patient'],true_class=fold['label'],p_long_term_survival=prob,n_features=len(data['ids'])))
        y=np.array([r['true_class']=='Long-term survival' for r in rows]);probs=np.array([r['p_long_term_survival'] for r in rows])
        low,high=patient_bootstrap(y,probs)
        outputs[name]=dict(seed=seed,setting=name,stringency=th,auc=float(roc_auc_score(y,probs)),ci_low=low,ci_high=high,n_patients=37,n_motifs=prepared['n_motifs'][name],predictions=rows)
    write_json(DEST/f'seed_{seed:03d}.json',outputs)
    return outputs

def main():
    prepare()
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(30)))
    predictions=pd.DataFrame([row for r in results for setting in r.values() for row in setting['predictions']])
    predictions.to_csv(DEST/'patient_predictions.csv',index=False)
    metrics=pd.DataFrame([{k:r[k] for k in ['seed','setting','stringency','auc','ci_low','ci_high','n_patients','n_motifs']} for output in results for r in output.values()])
    metrics.to_csv(DEST/'seed_metrics.csv',index=False)
    old=pd.read_csv(BASE/'loocv_predictions.csv')
    fixed=results[0]['saved']['stringency']
    old=old[np.isclose(old.stringency,fixed,rtol=0,atol=1e-14)].set_index('patient')
    replay=predictions[(predictions.seed==0)&(predictions.setting=='saved')].set_index('patient').reindex(old.index)
    assert len(old)==37
    maxdiff=float(np.max(np.abs(old.p_long_term_survival-replay.p_long_term_survival)))
    assert maxdiff<1e-14, maxdiff
    assert len(predictions)==2220
    summaries={}
    for name,g in metrics.groupby('setting'):
        assert len(g)==30 and set(g.seed)==set(range(30))
        for seed,p in predictions[predictions.setting==name].groupby('seed'):
            assert p.patient.nunique()==37 and p.true_class.value_counts().to_dict()=={'Long-term survival':30,'Short-term survival':7}
            assert np.isclose(roc_auc_score(p.true_class=='Long-term survival',p.p_long_term_survival),g[g.seed==seed].auc.iloc[0])
        summaries[name]=dict(n_runs=30,stringency=float(g.stringency.iloc[0]),n_motifs=int(g.n_motifs.iloc[0]),mean_auc=g.auc.mean(),sd_auc=g.auc.std(),min_auc=g.auc.min(),max_auc=g.auc.max(),reference=g[g.seed==0].drop(columns='setting').iloc[0].to_dict())
    summaries['audit']=dict(saved_reference_max_probability_difference=maxdiff,n_patients=37,n_long=30,n_short=7,bootstrap_replicates=10000,fanmod_repetitions=False,gnn_runs=0,documents_edited=False)
    write_json(DEST/'summary.json',summaries)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
