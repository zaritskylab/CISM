#!/usr/bin/env python3
"""Baselines, final multiclass stack, and paper anti-motif uncertainty.

Only task-owned local outputs are written. Connected documents are read-only.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
from types import SimpleNamespace
import typing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score
from analyze import ROOT,OUT,patient_bootstrap,write_json
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from analysis.permutation_controls.tnbc_label_permutation import StateReader
from analysis.permutation_controls import crc_label_permutation as engine

DEST=OUT/'additional'
MEL=Path('/groups/assafza_group/assafza/Omri/processed_data_23_10_2025')
STACK=ROOT/'multi_class/runtime/without_pp_remove_tumor/one_vs_one_third_class_confidence_stack'

def save_prepared(name,data):
    (DEST/'prepared').mkdir(parents=True,exist_ok=True)
    with (DEST/'prepared'/f'{name}.pkl').open('wb') as f:pickle.dump(data,f)
    metadata={k:v for k,v in data.items() if k not in ['x','y','folds','matrix','labels_array']}
    write_json(DEST/'prepared'/f'{name}_manifest.json',metadata)
    print('Prepared',name,flush=True)

def graph_counts(directory,labels,ntypes):
    """Pool unique retained nodes and deduplicated undirected edges per patient."""
    prefix='CRC' if next(iter(labels)).startswith('CRC') else 'TNBC' if next(iter(labels)).startswith('TNBC') else 'Melanoma'
    counts={p:np.zeros(ntypes,dtype=np.int64) for p in labels}
    edges={p:np.zeros((ntypes,ntypes),dtype=np.int64) for p in labels}
    manifest=[]
    for path in sorted(directory.glob('Patient_*_FOV*.txt')):
        pid=prefix+re.match(r'Patient_(\d+)_',path.name)[1]
        if pid not in labels:continue
        a=np.loadtxt(path,dtype=np.int64,ndmin=2)
        assert a.shape[1]==4
        allids=np.r_[a[:,0],a[:,1]];alltypes=np.r_[a[:,2],a[:,3]]
        sort=np.argsort(allids,kind='stable');allids=allids[sort];alltypes=alltypes[sort]
        ids,starts=np.unique(allids,return_index=True)
        low=np.minimum.reduceat(alltypes,starts);high=np.maximum.reduceat(alltypes,starts)
        assert np.array_equal(low,high) and (low>=0).all() and (low<ntypes).all()
        e=np.unique(np.sort(a[:,:2],axis=1),axis=0)
        assert (e[:,0]!=e[:,1]).all()
        types=low[np.searchsorted(ids,e)]
        c=np.bincount(types[:,0]*ntypes+types[:,1],minlength=ntypes**2).reshape(ntypes,ntypes)
        # Historical GraphReader increments both orientations, even on diagonal.
        edges[pid]+=c+c.T
        counts[pid]+=np.bincount(low,minlength=ntypes)
        manifest.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),patient=pid,nodes=len(ids),edges=len(e)))
    assert {m['patient'] for m in manifest}==set(labels)
    return counts,edges,manifest

def verify_graph_feature_alignment(patients,x,counts,edges,representation):
    """Audit each feature row against independently patient-keyed graph counts."""
    n=len(counts[patients[0]])
    pairs=np.triu_indices(n,1)
    expected=np.stack([counts[p]/counts[p].sum() if representation=='cell_type'
                       else edges[p][pairs]/edges[p].sum() for p in patients])
    np.testing.assert_allclose(x,expected,rtol=0,atol=1e-15,
                               err_msg='Feature rows do not match patient IDs')

def prepare_baselines():
    mel=pd.read_csv(MEL/'patient_class.csv',header=None,names=['patient','label']).set_index('patient').label.to_dict()
    mel={p:l for p,l in mel.items() if l in ['NN','NP','PN']}
    raw=pd.read_csv(ROOT/'pet_data/TNBC/patient_class.csv',header=None,names=['patient','days'])
    tnbc=dict(zip(raw.patient,np.where(raw.days<1000,'Short-term survival','Long-term survival')))
    raw=pd.read_csv('/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv')[['Patient','Group']].dropna().astype(int)
    crc=dict(zip('CRC'+raw.Patient.astype(int).astype(str),raw.Group.map({1:'CLR',2:'DII'})))
    all_graphs={}
    for source,directory,labels,n in [('melanoma',MEL/'data/melanoma_distance_100_XX',mel,16),('tnbc',ROOT/'pet_data/TNBC',tnbc,16),('crc',Path('/home/omriavit/research/CRC_graphs'),crc,17)]:
        cache=DEST/'prepared'/f'{source}_graph_counts.pkl'
        if cache.exists():
            with cache.open('rb') as f:all_graphs[source]=pickle.load(f)
        else:
            value=graph_counts(directory,labels,n);cache.parent.mkdir(parents=True,exist_ok=True)
            with cache.open('wb') as f:pickle.dump(value,f)
            all_graphs[source]=value
    for tag,source,labels,positive in [('melanoma_nn_np','melanoma',{p:l for p,l in mel.items() if l in ['NN','NP']},'NN'),('melanoma_np_pn','melanoma',{p:l for p,l in mel.items() if l in ['NP','PN']},'NP'),('tnbc','tnbc',tnbc,'Long-term survival'),('crc','crc',crc,'CLR')]:
        counts,edges,manifest=all_graphs[source]
        patients=sorted(labels)
        n=len(next(iter(counts.values())))
        xcells=np.stack([counts[p]/counts[p].sum() for p in patients])
        pairs=list(zip(*np.triu_indices(n,1)))
        xpairs=np.stack([[edges[p][a,b]/edges[p].sum() for a,b in pairs] for p in patients])
        subset=[m for m in manifest if m['patient'] in labels]
        for representation,x,rng_mode in [('cell_type',xcells,'per_fold'),('pairwise',xpairs,'advancing')]:
            # CRC raw-cell composition has its own numeric patient order.
            # Do not let that order leak into the following pairwise matrix,
            # whose rows were constructed in lexical patient order above.
            patients=sorted(labels)
            cols=[str(i) for i in range(n)] if representation=='cell_type' else [f'{a}:{b}' for a,b in pairs]
            sources=subset
            if tag=='crc' and representation=='cell_type':
                path=ROOT/'analysis/outputs/crc_singleton_classifier/patient_cell_type_frequencies.csv'
                freq=pd.read_csv(path,index_col=0);freq.index='CRC'+freq.index.astype(str)
                patients=list(freq.index);x=freq.to_numpy();cols=list(freq.columns)
                sources=[dict(path=str(path),sha256=engine.digest(path))]
            else:
                verify_graph_feature_alignment(patients,x,counts,edges,representation)
            assert np.isfinite(x).all()
            name=tag+'_'+representation
            data=dict(name=name,kind='baseline',dataset=tag,representation=representation,patients=patients,actual_labels=[labels[p] for p in patients],positive=positive,n_patients=len(patients),n_features=x.shape[1],feature_columns=cols,rng_mode=rng_mode,sources=sources,x=x,y=np.array([labels[p] for p in patients]),definition='Pooled retained graph-node counts per patient / total retained nodes' if representation=='cell_type' else 'Historical distinct cell-type pairs only (i<j), symmetric edge count divided by all symmetric edge counts; same-type pairs omitted from features but retained in denominator')
            if tag=='crc' and representation=='cell_type':data['definition']='Saved CRC raw-cell type frequencies, matching run_crc_singleton_classifier.py'
            save_prepared(name,data)
            pd.DataFrame(x,index=patients,columns=cols).to_csv(DEST/'prepared'/f'{name}_features.csv')

def native_methods():
    path=ROOT/'cism/cism.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TissueStateDiscriminativeMotifs')
    functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_extract_discriminative','_extract_features']]
    for fn in functions:fn.decorator_list=[]
    ns=dict(pd=pd,np=np,typing=typing,FeatureConfiguration=object,SoftDiscriminativeFC=type('Soft',(),{}),DiscriminativeMotifs=SimpleNamespace(PATIENT_CLASS='patient_class',PATIENT_COUNT_KEY='patient_count',PATIENT_PERCENTAGE_KEY='patient_percentage'))
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
    return ns

def prepare_antimotifs():
    source=ROOT/'melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np/Melanoma_NN_NP_absent_cism_ready.pkl'
    with source.open('rb') as f:artifact=StateReader(f).load()
    frame=artifact['discriminator'].cism.motifs_dataset
    classes=artifact['discriminator'].patient_class_df.patient_class
    classes=classes[classes.isin(['NN','NP'])]
    significant,matrix=engine.prepare_data(frame,classes.index)
    assert (pd.to_numeric(significant.z_score)<0).all() and (significant.p_value<.05).all()
    assert set(classes.value_counts())=={18,20}
    ns=native_methods()
    conf=SimpleNamespace(get_shared_percentage=lambda:.35,max_class_features=30,use_cells_type_composition=False,use_motifs=True,cells_type_composition_vec=[],switch_motif_with_cell_type_composition=[],switch_cell_type_composition_with_motif=[],switch_cell_type_composition_with_motif_map={},cells_type_composition_hash_remove=[],motifs_ids_remove=[])
    folds=[];union=set()
    for patient in classes.index:
        train=significant[significant.Patient_uId!=patient].copy()
        train['patient_class']=pd.Categorical(train.Patient_uId.map(classes))
        selected,_=ns['_extract_discriminative'](None,train,'patient_class','ID',[],1,1,.05)
        _,ids,_=ns['_extract_features'](selected,conf)
        assert ids
        union.update(ids)
        order=[p for group in pd.unique(classes) for p in significant.Patient_uId.unique() if p!=patient and classes.loc[p]==group]
        folds.append(dict(patient=patient,label=classes.loc[patient],x_train=matrix.loc[order,ids].to_numpy(),y_train=classes.loc[order].to_numpy(),x_test=matrix.loc[[patient],ids].to_numpy(),ids=list(map(int,ids))))
    save_prepared('underrepresented_4node_035',dict(name='underrepresented_4node_035',kind='antimotif',positive='NN',patients=list(classes.index),actual_labels=classes.tolist(),n_patients=len(classes),n_motifs=len(union),stringency=.35,significance='Two-tailed normal Z p<0.05 with Z<0; positive depletion features; observed frequency >0',sources=[dict(path=str(source),sha256=engine.digest(source))],folds=folds))

def prepare_multiclass():
    from cism import build_patient_motif_matrix
    from multi_class.confidence_stacking import make_confidence_scoring_folds,_select_with_training_only_fallback
    from multi_class.motif_panels import make_comparisons,_classes_for_comparison
    source=ROOT/'multi_class/runtime/without_pp_remove_tumor/artifact/without_pp_remove_tumor_cism_ready.pkl'
    with source.open('rb') as f:artifact=StateReader(f).load()
    frame=artifact['cism'].motifs_dataset
    classes=pd.read_csv(ROOT/'melanoma_tutorials/pet_data/Melanoma/patient_class.csv',header=None,names=['patient','patient_class']).set_index('patient')
    classes=classes[classes.patient_class.isin(['NN','NP','PN'])]
    classes.index=classes.index.astype(str)
    tuning_path=STACK.parent/'one_vs_one_union/comparison_tuning.csv'
    tuning=pd.read_csv(tuning_path).set_index('comparison')
    matrix=build_patient_motif_matrix(frame,patient_ids=list(classes.index))
    folds=[];columns=[]
    for comparison in make_comparisons(('NN','NP','PN'),'one_vs_one_union'):
        local=_classes_for_comparison(classes,comparison)
        column=f'confidence_{comparison.name}_{comparison.positive_label}';columns.append(column)
        for fold in make_confidence_scoring_folds(classes,comparison):
            train=local.loc[list(fold.train_patients)]
            selected,used=_select_with_training_only_fallback(frame[frame.Patient_uId.isin(fold.train_patients)],train,comparison,float(tuning.loc[comparison.name,'stringency']),max_class_features=30,p_value_threshold=.05)
            ids=selected.ID.drop_duplicates().tolist();assert ids
            # Building once globally is identical because the features aggregate
            # within each patient independently, without using class labels.
            folds.append(dict(column=column,positive=comparison.positive_label,train_patients=list(fold.train_patients),score_patients=list(fold.score_patients),x_train=matrix.loc[list(fold.train_patients),ids].to_numpy(),y_train=train.patient_class.to_numpy(),x_test=matrix.loc[list(fold.score_patients),ids].to_numpy(),ids=list(map(int,ids)),stringency=used,fold_type=fold.fold_type))
    assert len(classes)==51 and len(folds)==105
    save_prepared('multiclass_confidence_stack',dict(name='multiclass_confidence_stack',kind='multiclass',patients=list(classes.index),actual_labels=classes.patient_class.tolist(),classes=['NN','NP','PN'],n_patients=51,columns=columns,folds=folds,tuning=tuning.stringency.to_dict(),fully_nested=False,sources=[dict(path=str(source),sha256=engine.digest(source)),dict(path=str(tuning_path),sha256=engine.digest(tuning_path))]))

def multiclass_bootstrap(labels,probabilities,predicted,draws=10000,seed=20261120):
    labels=np.asarray(labels);probabilities=np.asarray(probabilities);predicted=np.asarray(predicted)
    classes=['NN','NP','PN'];rng=np.random.default_rng(seed)
    weights=np.zeros((draws,len(labels)))
    for label in classes:
        mask=labels==label;n=mask.sum();weights[:,mask]=rng.multinomial(n,np.ones(n)/n,size=draws)
    correct=(labels==predicted).astype(float)
    accuracy=weights@correct/len(labels)
    balanced=sum(weights[:,labels==c]@correct[labels==c]/(labels==c).sum() for c in classes)/3
    scores=[]
    for j,c in enumerate(classes):
        mask=labels==c;a=probabilities[mask,j];b=probabilities[~mask,j]
        kernel=(a[:,None]>b[None,:]).astype(float)+.5*(a[:,None]==b[None,:])
        scores.append(np.einsum('bi,ij,bj->b',weights[:,mask],kernel,weights[:,~mask],optimize=True)/(len(a)*len(b)))
    results={'accuracy':accuracy,'balanced_accuracy':balanced,'roc_auc_ovr_macro':np.mean(scores,axis=0)}
    results.update({f'roc_auc_ovr_{c}':s for c,s in zip(classes,scores)})
    return {k:np.quantile(v,[.025,.975]).tolist() for k,v in results.items()}

def binary_job(data,seed):
    probabilities=[];rng=np.random.RandomState(seed)
    if data['kind']=='baseline':
        for i in range(data['n_patients']):
            train=np.arange(data['n_patients'])!=i
            forest=RandomForestClassifier(n_estimators=100,random_state=rng if data['rng_mode']=='advancing' else seed,n_jobs=1)
            forest.fit(data['x'][train],data['y'][train])
            probabilities.append(float(forest.predict_proba(data['x'][[i]])[0,list(forest.classes_).index(data['positive'])]))
    else:
        for fold in data['folds']:
            forest=RandomForestClassifier(n_estimators=100,random_state=rng,n_jobs=1)
            forest.fit(fold['x_train'],fold['y_train'])
            probabilities.append(float(forest.predict_proba(fold['x_test'])[0,list(forest.classes_).index(data['positive'])]))
    y=np.array(data['actual_labels'])==data['positive']
    auc=float(roc_auc_score(y,probabilities));low,high=patient_bootstrap(y,probabilities)
    return dict(name=data['name'],seed=seed,kind=data['kind'],auc=auc,ci_low=low,ci_high=high,patients=data['patients'],actual_labels=data['actual_labels'],positive=data['positive'],probabilities=probabilities)

def multiclass_job(data,seed):
    x=pd.DataFrame(index=data['patients'],columns=data['columns'],dtype=float)
    for fold in data['folds']:
        forest=RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1)
        forest.fit(fold['x_train'],fold['y_train'])
        probs=forest.predict_proba(fold['x_test'])[:,list(forest.classes_).index(fold['positive'])]
        x.loc[fold['score_patients'],fold['column']]=probs
    assert not x.isna().any().any()
    y=np.array(data['actual_labels']);probabilities=[];predicted=[]
    for i in range(len(y)):
        train=np.arange(len(y))!=i
        model=LogisticRegression(penalty='l2',C=1.,solver='lbfgs',max_iter=2000,random_state=seed)
        model.fit(x.iloc[train],y[train])
        p=model.predict_proba(x.iloc[[i]])[0];probabilities.append(p.tolist());predicted.append(str(model.classes_[p.argmax()]))
        assert list(model.classes_)==data['classes']
    metrics=dict(accuracy=float(accuracy_score(y,predicted)),balanced_accuracy=float(balanced_accuracy_score(y,predicted)),roc_auc_ovr_macro=float(roc_auc_score(y,probabilities,labels=data['classes'],multi_class='ovr',average='macro')))
    for j,c in enumerate(data['classes']):metrics[f'roc_auc_ovr_{c}']=float(roc_auc_score(y==c,np.asarray(probabilities)[:,j]))
    cis=multiclass_bootstrap(y,probabilities,predicted)
    replay={}
    if seed==0:
        reference=pd.read_csv(STACK/'confidence_features.csv',index_col=0).reindex(data['patients'])
        replay['max_confidence_difference']=float(np.max(np.abs(reference.to_numpy()-x.to_numpy())))
        reference=pd.read_csv(STACK/'loocv_predictions.csv').set_index('patient_id').reindex(data['patients'])
        replay['max_probability_difference']=float(np.max(np.abs(reference[[f'prob_{c}' for c in data['classes']]].to_numpy()-np.asarray(probabilities))))
        assert replay['max_confidence_difference']<1e-12,replay
        assert replay['max_probability_difference']<1e-10,replay
    return dict(name=data['name'],seed=seed,kind='multiclass',metrics=metrics,confidence_intervals=cis,patients=data['patients'],actual_labels=data['actual_labels'],classes=data['classes'],probabilities=probabilities,predicted=predicted,confidence_features=x.to_numpy().tolist(),reference_replay=replay)

def job(name,seed):
    path=DEST/'runs'/name/f'seed_{seed:03d}.json'
    if path.exists():return name,seed,'cached'
    with (DEST/'prepared'/f'{name}.pkl').open('rb') as f:data=pickle.load(f)
    result=multiclass_job(data,seed) if data['kind']=='multiclass' else binary_job(data,seed)
    write_json(path,result)
    return name,seed,round(result.get('auc',result.get('metrics',{}).get('roc_auc_ovr_macro',0)),5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['baseline','advanced','run','all'],default='all');parser.add_argument('--names',nargs='+');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--seed-stop',type=int,default=30,help='Exclusive upper seed bound')
    args=parser.parse_args()
    assert 0<=args.seed_start<args.seed_stop
    DEST.mkdir(exist_ok=True)
    if args.stage in ['baseline','all']:prepare_baselines()
    if args.stage in ['advanced','all']:
        prepare_antimotifs();prepare_multiclass()
    if args.stage in ['run','all']:
        names=[p.stem for p in (DEST/'prepared').glob('*_manifest.json')]
        names=[n.removesuffix('_manifest') for n in names]
        if args.names:names=[name for name in names if name in args.names]
        assert names
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(job,name,seed) for name in names for seed in range(args.seed_start,args.seed_stop)]
            for future in as_completed(futures):print(future.result(),flush=True)

if __name__=='__main__':main()
