#!/usr/bin/env python3
"""Produce source tables and static scientific figures from verified executions."""
from __future__ import annotations
import base64
import itertools
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cism-robustness-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import OUT, core, patient_bootstrap, write_json

ORIGINAL_TYPES = ['b cell','cd4 apc','cd4 t cell','treg','cd8 t cell','DC','gc','hevs','mac','memory%cd4 t cell','nk cell','neutrophil','stroma','tumor','unidentified','blood vessels']

def main():
    executions = [json.loads(p.read_text()) for p in sorted((OUT/'executions').glob('*.json'))]
    expected = {'nn_np_raw4':30,'nn_np_size3':1,'nn_np_size4':30,'nn_np_size5':1,'np_pn_size3':1,'np_pn_size4':1,'np_pn_size5':30}
    counts = pd.Series([r['name'] for r in executions]).value_counts().to_dict()
    assert counts == expected, (counts, expected)
    rows, predrows, mainrows, distrows = [], [], [], []
    matrices = {}
    for r in executions:
        y = np.array(r['labels']) == r['positive']
        assert len(set(r['patients'])) == len(y)
        for curve in r['curves']:
            probs = r['probabilities'][str(curve['stringency'])]
            assert len(probs) == len(y)
            info = dict(name=r['name'], seed=r['seed'], positive=r['positive'], negative=r['negative'], motif_size=r['motif_size'])
            rows.append({**info, **curve})
            for p,label,prob in zip(r['patients'], r['labels'], probs):
                predrows.append(dict(**info, stringency=curve['stringency'], patient=p, label=label, probability=prob))
            fixed = .4 if r['positive']=='NP' else .46
            if curve['stringency'] == fixed and r['name'] in ['nn_np_raw4','nn_np_size4','np_pn_size5']:
                lo,hi = patient_bootstrap(y, probs) if curve['rf_admissible'] else (None,None)
                mainrows.append(dict(**info, **curve, ci_low=lo, ci_high=hi))
        if r['seed'] != 0: continue
        with (OUT/'prepared'/f"{r['name']}.pkl").open('rb') as f: prepared = pickle.load(f)
        data = prepared['data']
        indices = {int(mid):ix for ix,mid in enumerate(data['ids'])}
        all_ids = set(itertools.chain.from_iterable(r['motif_ids'].values()))
        names = core.GRAPH_TYPE_NAMES if r['name']=='nn_np_raw4' else ORIGINAL_TYPES
        cache = {mid:core.topology_vectors(data['graphs'][mid], type_names=names) for mid in all_ids}
        for th, ids in r['motif_ids'].items():
            d = core.make_distributions(data, {indices[mid] for mid in ids}, cache)
            for internal,actual in [('NN',r['positive']),('NP',r['negative'])]:
                v = d[internal]['pair']
                matrices[r['name'],float(th),actual] = v
                if v is not None:
                    for pair,prob in zip(core.PAIRS, v):
                        distrows.append(dict(name=r['name'], stringency=float(th), label=actual, cell_a=pair[0], cell_b=pair[1], probability=prob))
    allrows = pd.DataFrame(rows)
    allrows.to_csv(OUT/'new_stringency_results.csv', index=False)
    pd.DataFrame(predrows).to_csv(OUT/'new_patient_predictions.csv', index=False)
    mainrows = pd.DataFrame(mainrows)
    mainrows.to_csv(OUT/'classifier_seed_metrics.csv', index=False)
    pd.DataFrame(distrows).to_csv(OUT/'motif_induced_pair_distributions.csv', index=False)
    summary = []
    for name,g in mainrows.groupby('name'):
        valid = g[g.rf_admissible]
        summary.append(dict(name=name, n_requested=len(g), n_valid=len(valid), mean_auc=valid.auc.mean(), sd_auc=valid.auc.std(ddof=1), min_auc=valid.auc.min(), max_auc=valid.auc.max(), reference_auc=g[g.seed==0].auc.iloc[0], reference_ci_low=g[g.seed==0].ci_low.iloc[0], reference_ci_high=g[g.seed==0].ci_high.iloc[0], n_motifs=g.n_motifs.iloc[0]))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT/'classifier_seed_summary.csv', index=False)
    native_metrics=pd.read_csv(OUT/'native_np_pn_seed_metrics.csv')
    native_summary=json.loads((OUT/'native_np_pn_summary.json').read_text())
    primary=summary[summary.name!='np_pn_size5'].copy()
    primary=pd.concat([primary,pd.DataFrame([dict(name='np_pn_size5_original_selector',n_requested=30,n_valid=30,mean_auc=native_summary['mean_auc'],sd_auc=native_summary['sd_auc'],min_auc=native_summary['min_auc'],max_auc=native_summary['max_auc'],reference_auc=native_summary['reference']['auc'],reference_ci_low=native_summary['reference']['ci_low'],reference_ci_high=native_summary['reference']['ci_high'],n_motifs=210)])],ignore_index=True)
    if (OUT/'crc/summary.json').exists():
        crc_summary=json.loads((OUT/'crc/summary.json').read_text())
        primary=pd.concat([primary,pd.DataFrame([dict(name='crc_size4_historical_rng',n_requested=30,n_valid=30,mean_auc=crc_summary['mean_auc'],sd_auc=crc_summary['sd_auc'],min_auc=crc_summary['min_auc'],max_auc=crc_summary['max_auc'],reference_auc=crc_summary['reference']['auc'],reference_ci_low=crc_summary['reference']['ci_low'],reference_ci_high=crc_summary['reference']['ci_high'],n_motifs=119)])],ignore_index=True)
    if (OUT/'tnbc/summary.json').exists():
        tnbc_summary=json.loads((OUT/'tnbc/summary.json').read_text())['figure_8a']
        primary=pd.concat([primary,pd.DataFrame([dict(name='tnbc_size3_fig8a_stringency_045',n_requested=30,n_valid=30,mean_auc=tnbc_summary['mean_auc'],sd_auc=tnbc_summary['sd_auc'],min_auc=tnbc_summary['min_auc'],max_auc=tnbc_summary['max_auc'],reference_auc=tnbc_summary['reference']['auc'],reference_ci_low=tnbc_summary['reference']['ci_low'],reference_ci_high=tnbc_summary['reference']['ci_high'],n_motifs=tnbc_summary['n_motifs'])])],ignore_index=True)
    primary.to_csv(OUT/'primary_classifier_summary.csv',index=False)
    correlations = []
    for comparison, positive,negative in [('nn_np','NN','NP'),('np_pn','NP','PN')]:
        for (left,right) in [(.2,.3),(.3,.4),(.4,.46),(.46,.48)]:
            for size in [3,4,5]:
                name = f'{comparison}_size{size}'
                for label in [positive,negative]:
                    a,b = matrices.get((name,left,label)),matrices.get((name,right,label))
                    if a is not None and b is not None:
                        rho,p = spearmanr(a,b)
                        cross = spearmanr(matrices[name,right,positive], matrices[name,right,negative]).statistic if matrices.get((name,right,positive)) is not None and matrices.get((name,right,negative)) is not None else None
                        correlations.append(dict(comparison=comparison, kind='stringency', left=left,right=right,motif_size=size,label=label,spearman_rho=rho, descriptive_p=p, cross_class_rho_at_right=cross))
        for left,right in [(3,4),(4,5)]:
            for label in [positive,negative]:
                a,b = matrices.get((f'{comparison}_size{left}',.3,label)), matrices.get((f'{comparison}_size{right}',.3,label))
                if a is not None and b is not None:
                    rho,p = spearmanr(a,b)
                    correlations.append(dict(comparison=comparison,kind='motif_size',left=left,right=right,motif_size=None,label=label,spearman_rho=rho,descriptive_p=p,cross_class_rho_at_right=None))
    pd.DataFrame(correlations).to_csv(OUT/'pair_distribution_sensitivity.csv', index=False)
    historical = pd.read_csv(OUT/'historical_stringency_curves.csv')
    hs = historical.groupby('stringency').agg(n_requested=('run_id','size'),n_valid=('auc','count'),mean_auc=('auc','mean'),sd_auc=('auc','std'),mean_motifs=('n_motifs','mean'),mean_empty_folds=('empty_folds','mean')).reset_index()
    hs.to_csv(OUT/'historical_parameter_summary.csv', index=False)
    recovery = pd.read_csv(OUT/'motif_recovery.csv')
    fig, axes = plt.subplots(2,2,figsize=(12,9))
    ax = axes[0,0]
    raw = pd.read_csv(OUT/'historical_run_metrics.csv').query('mode == "fixed"')
    ax.hist(raw.auc,bins=12,color='#487aa1',edgecolor='white')
    ax.axvline(raw.auc.mean(),color='black',ls='--',label=f'mean {raw.auc.mean():.3f}; SD {raw.auc.std():.3f}')
    ax.set(xlabel='Patient LOOCV AUC',ylabel='Executions',title='A. FANMOD execution variability (n=100)');ax.legend(fontsize=9)
    ax=axes[0,1]
    valid=hs[hs.n_valid>0]
    ax.plot(valid.stringency, valid.mean_auc, 'o-',color='#487aa1')
    ax.fill_between(valid.stringency,valid.mean_auc-valid.sd_auc,valid.mean_auc+valid.sd_auc,alpha=.2)
    ax.axhline(.5,c='gray',ls=':');ax.axvline(.46,c='gray',ls='--')
    ax.set(xlabel='Stringency',ylabel='AUC (mean ± execution SD)',ylim=(.4,1),title='B. NN–NP: four-cell motifs')
    ax=axes[1,0]
    recovery=recovery.sort_values(['in_reference','recovery_percent'],ascending=False).reset_index(drop=True)
    ax.bar(np.arange(len(recovery)),recovery.recovery_percent,color=np.where(recovery.in_reference,'#487aa1','#df9448'))
    ax.axhline(90,c='gray',ls=':');ax.set(xlabel='Motif (ordered by recovery within reference group)',ylabel='Executions selected (%)',title='C. Stability relative to saved run 0')
    ax=axes[1,1]
    groups=[('NN–NP raw\n4-cell',mainrows[(mainrows.name=='nn_np_raw4')&mainrows.rf_admissible]),('NN–NP original\n4-cell',mainrows[(mainrows.name=='nn_np_size4')&mainrows.rf_admissible]),('NP–PN original\n5-cell',native_metrics)]
    if (OUT/'crc/seed_metrics.csv').exists():groups.append(('CRC\n4-cell',pd.read_csv(OUT/'crc/seed_metrics.csv')))
    if (OUT/'tnbc/seed_metrics.csv').exists():groups.append(('TNBC\n3-cell, 0.45',pd.read_csv(OUT/'tnbc/seed_metrics.csv').query('setting == "figure_8a"')))
    for i,(label,g) in enumerate(groups):
        ax.scatter(i+np.random.default_rng(0).uniform(-.1,.1,len(g)),g.auc,s=18,alpha=.6)
        ax.errorbar(i,g.auc.mean(),yerr=g.auc.std(),fmt='ko',capsize=5)
    ax.set(xticks=list(range(len(groups))),xticklabels=[label for label,g in groups],ylabel='AUC',title='D. Classifier seed variability (n=30 each)',ylim=(.4,1))
    ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(OUT/'uncertainty_summary.png',dpi=180);fig.savefig(OUT/'uncertainty_summary.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,tag,title in zip(axes,['nn_np','np_pn'],['NN–NP (original caches)','NP–PN (original caches)']):
        for size,color in [(3,'#54a24b'),(4,'#9467bd'),(5,'#e69445')]:
            g=allrows[(allrows.name==f'{tag}_size{size}')&(allrows.seed==0)]
            ax.plot(g.stringency,g.auc,'o-',label=f'{size}-cell',color=color)
            for _,row in g[~g.rf_admissible].iterrows():ax.plot(row.stringency,.42,'x',color=color)
        ax.axhline(.5,c='gray',ls=':');ax.set(xlabel='Stringency',ylabel='AUC',title=title,ylim=(.4,1));ax.legend()
    fig.text(.5,.01,'Seed 0; crosses at the plot floor mark settings with at least one empty-feature training fold.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'parameter_sensitivity.png',dpi=180);fig.savefig(OUT/'parameter_sensitivity.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex='col')
    for col,(name,title) in enumerate([('nn_np_size4','NN–NP: four-cell motifs'),('np_pn_size5','NP–PN: five-cell motifs')]):
        g=pd.read_csv(OUT/f'{name}_native_sweep.csv')
        axes[0,col].plot(g.stringency,g.auc,'o-',color='#487aa1')
        bad=g[~g.rf_admissible]
        axes[0,col].scatter(bad.stringency,[.42]*len(bad),marker='x',color='#df9448')
        axes[0,col].axhline(.5,color='gray',ls=':')
        axes[0,col].set(title=title,ylabel='LOOCV AUC (RF seed 0)',ylim=(.4,1))
        axes[1,col].plot(g.stringency,g.n_motifs,'o-',color='#9467bd')
        axes[1,col].set(xlabel='Discrimination stringency',ylabel='Fold-union motif count')
    fig.text(.5,.01,'Original CISM selector; crosses indicate undefined AUC due to empty-feature folds.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'primary_parameter_sensitivity.png',dpi=180);fig.savefig(OUT/'primary_parameter_sensitivity.pdf');plt.close(fig)
    ref=json.loads((OUT/'executions/nn_np_raw4_seed_000.json').read_text())
    old=json.loads((OUT.parents[1]/'motif_noise_summary_outputs/final_analysis/runs/raw/run_000.json').read_text())
    replay=dict(historical_auc=old['selections']['fixed']['auc'],new_auc=next(x['auc'] for x in ref['curves'] if x['stringency']==.46),selected_motif_sets_identical=set(ref['motif_ids']['0.46'])==set(old['selections']['fixed']['motif_ids']),max_prediction_difference=float(np.max(np.abs(np.array(ref['probabilities']['0.46'])-np.array(old['probabilities']['0.46'])))))
    write_json(OUT/'reference_replay_audit.json',replay)
    print(primary.to_string(index=False));print('Replay audit:',replay)
    print('All 94 first-pass executions, 30 original-selector NP–PN executions, and native primary sweeps complete.')

if __name__=='__main__': main()
