#!/usr/bin/env python3
"""CRC extension: preserve historical feature selection and RNG progression.

Reserve RNG draws for earlier thresholds without refitting those forests; this
reproduces the fixed-0.55 portion of the complete historical grid search.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from analyze import ROOT,OUT,patient_bootstrap,write_json
sys.path.insert(0,str(ROOT/'analysis'))
from analysis.permutation_controls import crc_label_permutation as crc

DEST=OUT/'crc'
ARTIFACT='/home/omriavit/research/cism_CRC_4.pkl'
LABELS='/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv'
BASE=ROOT/'analysis/outputs/crc_label_permutation_historical/original'

def prepare():
    DEST.mkdir(exist_ok=True)
    original=json.loads((BASE/'summary.json').read_text())
    assert crc.digest(ARTIFACT)==original['config']['artifact_sha256']
    assert crc.digest(LABELS)==original['config']['labels_sha256']
    with open(ARTIFACT,'rb') as f:artifact=crc.LegacyReader(f).load()
    raw=pd.read_csv(LABELS)[['Patient','Group']].dropna().astype(int)
    labels=raw.set_index(raw.Patient.map(lambda p:f'CRC{p}')).Group.map({1:'CLR',2:'DII'})
    significant,matrix=crc.prepare_data(artifact.motifs_dataset,labels.index)
    assert labels.value_counts().to_dict()=={'DII':18,'CLR':17}
    folds=[];union=set()
    for patient in labels.index:
        candidates=crc.training_candidates(significant,labels,patient)
        selections={th:crc.select_ids(candidates,th,30) for th in crc.GRID}
        order=[p for group in pd.unique(labels) for p in significant.Patient_uId.unique() if p!=patient and labels.loc[p]==group]
        fixed=selections[.55];assert fixed
        union.update(fixed)
        folds.append(dict(patient=patient,label=labels.loc[patient],order=order,features=fixed,x_train=matrix.loc[order,fixed].to_numpy(),y_train=labels.loc[order].to_numpy(),x_test=matrix.loc[[patient],fixed].to_numpy(),has_features={th:bool(ids) for th,ids in selections.items()}))
    prepared=dict(folds=folds,n_motifs=len(union),artifact_sha256=original['config']['artifact_sha256'],labels_sha256=original['config']['labels_sha256'],grid=crc.GRID)
    with (DEST/'prepared.pkl').open('wb') as f:pickle.dump(prepared,f)
    write_json(DEST/'source_manifest.json',dict(artifact=ARTIFACT,labels=LABELS,artifact_sha256=prepared['artifact_sha256'],labels_sha256=prepared['labels_sha256'],n_motifs=len(union),fixed_stringency=.55,grid=crc.GRID,forest_seed_policy='Historical RNG, reserve one int32 draw per tree for every earlier nonempty grid fold; fit only at fixed .55',trees=100,seeds=list(range(30)),positive='CLR',code_sha256=crc.digest(__file__)))
    print('Prepared CRC: 35 patients,',len(union),'fold-union motifs',flush=True)

def run(seed):
    with (DEST/'prepared.pkl').open('rb') as f:prepared=pickle.load(f)
    rng=np.random.RandomState(seed);states={}
    for th in prepared['grid']:
        for fold in prepared['folds']:
            if fold['has_features'][th]:
                if th==.55:states[fold['patient']]=rng.get_state()
                rng.randint(np.iinfo(np.int32).max,size=100)
        if th==.55:break
    rows=[]
    for fold in prepared['folds']:
        local=np.random.RandomState();local.set_state(states[fold['patient']])
        clf=RandomForestClassifier(n_estimators=100,random_state=local,n_jobs=1)
        clf.fit(fold['x_train'],fold['y_train'])
        prob=float(clf.predict_proba(fold['x_test'])[0,np.flatnonzero(clf.classes_=='CLR')[0]])
        rows.append(dict(seed=seed,patient=fold['patient'],true_class=fold['label'],p_CLR=prob,n_features=len(fold['features'])))
    y=np.array([r['true_class']=='CLR' for r in rows]);probs=np.array([r['p_CLR'] for r in rows])
    lo,hi=patient_bootstrap(y,probs)
    result=dict(seed=seed,auc=float(roc_auc_score(y,probs)),ci_low=lo,ci_high=hi,n_patients=35,n_motifs=prepared['n_motifs'],predictions=rows)
    write_json(DEST/f'seed_{seed:03d}.json',result)
    return result

def main():
    prepare()
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(30)))
    predictions=pd.DataFrame([row for r in results for row in r['predictions']])
    predictions.to_csv(DEST/'patient_predictions.csv',index=False)
    metrics=pd.DataFrame([{k:r[k] for k in ['seed','auc','ci_low','ci_high','n_patients','n_motifs']} for r in results])
    metrics.to_csv(DEST/'seed_metrics.csv',index=False)
    original=pd.read_csv(BASE/'loocv_predictions.csv').query('stringency == 0.55').set_index('patient')
    replay=predictions[predictions.seed==0].set_index('patient').reindex(original.index)
    maxdiff=float(np.max(np.abs(original.p_CLR-replay.p_CLR)))
    assert maxdiff==0, maxdiff
    assert len(predictions)==1050 and predictions.groupby('seed').patient.nunique().eq(35).all()
    for r in results:assert np.isclose(roc_auc_score(np.array([p['true_class'] for p in r['predictions']])=='CLR',[p['p_CLR'] for p in r['predictions']]),r['auc'])
    summary=dict(n_runs=30,fixed_stringency=.55,motif_size=4,n_patients=35,n_CLR=17,n_DII=18,mean_auc=metrics.auc.mean(),sd_auc=metrics.auc.std(),min_auc=metrics.auc.min(),max_auc=metrics.auc.max(),reference=metrics[metrics.seed==0].iloc[0].to_dict(),historical_reference_max_prediction_difference=maxdiff,fanmod_repetitions=False,gnn_runs=0,documents_edited=False)
    write_json(DEST/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
