#!/usr/bin/env python3
"""Verify and summarize all eight baselines, multiclass, and anti-motifs."""
from __future__ import annotations
import json
import pickle
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cism-robustness-mpl')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score
from analyze import ROOT,OUT,write_json
from additional_uncertainty import DEST,STACK,verify_graph_feature_alignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=[f'{dataset}_{kind}' for dataset in ['melanoma_nn_np','melanoma_np_pn','tnbc','crc'] for kind in ['cell_type','pairwise']]+['underrepresented_4node_035','multiclass_confidence_stack']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--multiclass-runs',type=int,choices=[30,60],default=60)
    args=parser.parse_args()
    binary_summary=[];multi_summary=[];multi_batches=[];predictions=[];metrics=[];replay={}
    for name in NAMES:
        paths=sorted((DEST/'runs'/name).glob('seed_*.json'))
        results=[json.loads(p.read_text()) for p in paths]
        n_runs=args.multiclass_runs if name=='multiclass_confidence_stack' else 30
        assert len(results)==n_runs and {r['seed'] for r in results}==set(range(n_runs)),(name,len(results))
        reference=next(r for r in results if r['seed']==0)
        if name=='multiclass_confidence_stack':
            replay[name]=reference['reference_replay']
            for result in results:
                y=np.array(result['actual_labels']);probs=np.array(result['probabilities']);predicted=np.array(result['predicted'])
                assert result['patients']==reference['patients'] and result['actual_labels']==reference['actual_labels']
                assert len(y)==51 and len(set(result['patients']))==51
                assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
                assert pd.Series(y).value_counts().to_dict()=={'NN':20,'NP':18,'PN':13}
                np.testing.assert_allclose(probs.sum(axis=1),1)
                assert np.isclose(result['metrics']['roc_auc_ovr_macro'],roc_auc_score(y,probs,labels=result['classes'],multi_class='ovr',average='macro'))
                assert np.isclose(result['metrics']['accuracy'],accuracy_score(y,predicted))
                assert np.isclose(result['metrics']['balanced_accuracy'],balanced_accuracy_score(y,predicted))
                for key,value in result['metrics'].items():metrics.append(dict(name=name,seed=result['seed'],metric=key,value=value,ci_low=result['confidence_intervals'][key][0],ci_high=result['confidence_intervals'][key][1]))
                for p,label,pred,prob in zip(result['patients'],y,predicted,probs):predictions.append(dict(name=name,seed=result['seed'],patient=p,label=label,predicted=pred,prob_NN=prob[0],prob_NP=prob[1],prob_PN=prob[2]))
            for key in reference['metrics']:
                values=[r['metrics'][key] for r in results]
                multi_summary.append(dict(name=name,metric=key,n_runs=n_runs,mean=np.mean(values),sd=np.std(values,ddof=1),min=min(values),max=max(values),reference=reference['metrics'][key],ci_low=reference['confidence_intervals'][key][0],ci_high=reference['confidence_intervals'][key][1]))
                batches=[('first_30',0,30),('all_trials',0,n_runs)]
                if n_runs==60:batches.insert(1,('additional_30',30,60))
                for batch,start,stop in batches:
                    subset=[r for r in results if start<=r['seed']<stop]
                    scores=np.array([r['metrics'][key] for r in subset])
                    multi_batches.append(dict(batch=batch,metric=key,n_runs=len(subset),seed_start=start,seed_stop_exclusive=stop,mean=scores.mean(),sd=scores.std(ddof=1),min=scores.min(),max=scores.max(),n_above_reference=int((scores>reference['metrics'][key]+1e-12).sum()),n_equal_reference=int(np.isclose(scores,reference['metrics'][key],rtol=0,atol=1e-12).sum())))
        else:
            manifest=json.loads((DEST/'prepared'/f'{name}_manifest.json').read_text())
            expected=manifest['n_patients']
            if manifest['kind']=='baseline':
                with (DEST/'prepared'/f'{name}.pkl').open('rb') as handle:prepared=pickle.load(handle)
                if name=='crc_cell_type':
                    frequencies=pd.read_csv(ROOT/'analysis/outputs/crc_singleton_classifier/patient_cell_type_frequencies.csv',index_col=0)
                    frequencies.index='CRC'+frequencies.index.astype(str)
                    np.testing.assert_allclose(prepared['x'],frequencies.loc[prepared['patients']].to_numpy(),rtol=0,atol=1e-15)
                else:
                    graph_source='melanoma' if name.startswith('melanoma') else manifest['dataset']
                    with (DEST/'prepared'/f'{graph_source}_graph_counts.pkl').open('rb') as handle:counts,edges,_=pickle.load(handle)
                    verify_graph_feature_alignment(prepared['patients'],prepared['x'],counts,edges,prepared['representation'])
            for result in results:
                assert result['patients']==reference['patients'] and result['actual_labels']==reference['actual_labels']
                assert result['patients']==manifest['patients'] and result['actual_labels']==manifest['actual_labels']
                assert len(result['patients'])==len(set(result['patients']))==expected
                y=np.array(result['actual_labels'])==result['positive'];probs=np.array(result['probabilities'])
                assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
                assert np.isclose(roc_auc_score(y,probs),result['auc'])
                metrics.append(dict(name=name,seed=result['seed'],metric='auc',value=result['auc'],ci_low=result['ci_low'],ci_high=result['ci_high']))
                for p,label,prob in zip(result['patients'],result['actual_labels'],probs):predictions.append(dict(name=name,seed=result['seed'],patient=p,label=label,positive=result['positive'],probability=prob))
            values=[r['auc'] for r in results]
            binary_summary.append(dict(name=name,n_patients=expected,n_runs=30,mean_auc=np.mean(values),sd_auc=np.std(values,ddof=1),min_auc=min(values),max_auc=max(values),reference_auc=reference['auc'],ci_low=reference['ci_low'],ci_high=reference['ci_high'],positive=reference['positive']))
            if name=='crc_cell_type':
                old=pd.read_csv(ROOT/'analysis/outputs/crc_singleton_classifier/loocv_predictions.csv')
                old=old[old.seed==0].copy();old['patient']='CRC'+old.Patient.astype(str)
                old=old.set_index('patient').reindex(reference['patients'])
                difference=float(np.max(np.abs(old.p_CLR.to_numpy()-reference['probabilities'])))
                assert difference<1e-12,difference
                replay[name]=dict(max_probability_difference=difference)
            if name=='underrepresented_4node_035':
                old=pd.read_csv(ROOT/'melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np/figures/auc_fixed_stringencies_two_vs_lower_tail.csv')
                historical=float(old.loc[(old.test=='two_tailed') & np.isclose(old.stringency,.35),'roc_auc'].iloc[0])
                assert np.isclose(reference['auc'],historical,rtol=0,atol=1e-12)
                replay[name]=dict(historical_auc=historical,reference_auc=reference['auc'],auc_replayed=True,probabilities_replayed=False,reason='Historical threshold CSV stores AUC only; no probability-vector equality claimed.')
    binary=pd.DataFrame(binary_summary);multi=pd.DataFrame(multi_summary)
    binary.to_csv(DEST/'binary_summary.csv',index=False);multi.to_csv(DEST/'multiclass_summary.csv',index=False)
    pd.DataFrame(multi_batches).to_csv(DEST/'multiclass_batch_summary.csv',index=False)
    pd.DataFrame(metrics).to_csv(DEST/'seed_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(DEST/'patient_predictions.csv',index=False)
    write_json(DEST/'reference_replays.json',replay)
    completed_runs=270+args.multiclass_runs
    write_json(DEST/'completion_audit.json',dict(status='passed',n_analyses=10,seeds_per_analysis={name:(args.multiclass_runs if name=='multiclass_confidence_stack' else 30) for name in NAMES},n_completed_runs=completed_runs,n_baseline_analyses=8,multiclass_patients=51,underrepresented_patients=38,underrepresented_stringency=.35,fully_nested_multiclass=False,bootstrap_replicates_per_interval=10000,crc_cell_reference_replay=replay['crc_cell_type'],multiclass_reference_replay=replay['multiclass_confidence_stack'],documents_edited=False,gnn_runs=0,baseline_patient_feature_alignment='passed',crc_pairwise_patient_order_corrected=True))
    labels=['Melanoma NN–NP','Melanoma NP–PN','TNBC','CRC']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for j,(kind,color) in enumerate([('cell_type','#5b80b4'),('pairwise','#c18c51')]):
        subset=binary.set_index('name').loc[[f'{tag}_{kind}' for tag in ['melanoma_nn_np','melanoma_np_pn','tnbc','crc']]]
        x=np.arange(4)+(j-.5)*.16
        axes[0].errorbar(x,subset.mean_auc,yerr=subset.sd_auc,fmt='o',capsize=5,color=color,label=kind.replace('_',' '))
    axes[0].axhline(.5,c='gray',ls=':');axes[0].set(xticks=range(4),xticklabels=labels,ylim=(.2,1),ylabel='AUC (mean ± classifier-seed SD)',title='Cell-type and historical pairwise baselines')
    axes[0].tick_params(axis='x',rotation=15);axes[0].legend(frameon=False)
    anti=binary[binary.name=='underrepresented_4node_035'].iloc[0]
    auc=multi[multi.metric=='roc_auc_ovr_macro'].iloc[0]
    axes[1].errorbar([0,1],[anti.mean_auc,auc['mean']],yerr=[anti.sd_auc,auc.sd],fmt='ko',capsize=6)
    axes[1].axhline(.5,c='gray',ls=':');axes[1].set(xticks=[0,1],xticklabels=['4-cell anti-motifs\nstringency 0.35 (30 seeds)',f'Three-class confidence stack\n({args.multiclass_runs} seeds)'],ylabel='AUC (mean ± classifier-seed SD)',ylim=(.2,1),title='Additional CISM analyses')
    fig.tight_layout();fig.savefig(DEST/'uncertainty_summary.png',dpi=180);fig.savefig(DEST/'uncertainty_summary.pdf');plt.close(fig)
    print(binary.to_string(index=False));print(multi.to_string(index=False));print(pd.DataFrame(multi_batches).to_string(index=False));print(f'All {completed_runs} requested runs verified.')

if __name__=='__main__':main()
