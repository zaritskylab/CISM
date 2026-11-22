"""Patient-level undirected interaction analysis; no manuscript files are changed.

Run: python analysis/patient_level_validation/interactions.py
Original graph code mapping is taken from Melanoma_100_distance_NN_NP.ipynb.
All eligible edge types are tested because exact manuscript S2 targets were unavailable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cism_patient_validation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numba import njit
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
GRAPH_DIR = Path('/groups/assafza_group/assafza/Omri/processed_data_23_10_2025/data/melanoma_distance_100_XX')
LABELS = ROOT.parents[1] / 'melanoma_tutorials/pet_data/Melanoma/patient_class.csv'
NAMES = ['B cell', 'CD4 APC', 'CD4 T cell', 'CD4 Treg', 'CD8 T cell', 'DCs',
         'Germinal Center B cell', 'Hevs', 'MAC', 'Memory CD4 T Cell', 'NK cell',
         'Neutrophil', 'Stroma', 'Tumor', 'Unidentified', 'Vessels']
ELIGIBLE = [i for i in range(16) if i != 13]
PAIRS = [(a,b) for a in ELIGIBLE for b in ELIGIBLE if a <= b]
LOOKUP = np.full((16,16), -1, dtype=np.int64)
for j,(a,b) in enumerate(PAIRS):
    LOOKUP[a,b] = LOOKUP[b,a] = j
TARGETS = [(0,0),(9,9),(0,9)]  # Handoff examples, NOT verified manuscript targets.

def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p),0,-1)))
    return out

def read_graph(path):
    a = np.loadtxt(path, dtype=np.int64, ndmin=2)
    assert a.shape[1] == 4
    ids, inv = np.unique(a[:,:2].ravel(), return_inverse=True)
    edges = np.unique(np.sort(inv.reshape(-1,2),axis=1),axis=0)
    assert np.all(edges[:,0] != edges[:,1]), f'Self loop: {path}'
    types = np.full(len(ids),-1,np.int64)
    for col in (0,1):
        idx = np.searchsorted(ids,a[:,col])
        for v,t in zip(idx,a[:,col+2]):
            assert types[v] in (-1,t), f'Inconsistent cell type: {path}'
            types[v]=t
    assert np.all(types>=0)
    assert not np.any(types==13), f'Tumor unexpectedly present in NN/NP graph: {path}'
    return ids,types,edges,len(a)

def expectation(types,edges,lookup=LOOKUP,strata=None):
    """Exact expected counts under exchangeable labels, including strata if given."""
    if strata is None:
        n=len(types); counts=np.bincount(types,minlength=16)
        return np.array([len(edges)*counts[a]*(counts[b]-(a==b))*(1 if a==b else 2)/(n*(n-1))
                         for a,b in PAIRS],float)
    result=np.zeros(len(PAIRS))
    levels=np.unique(strata)
    ns={s:int(np.sum(strata==s)) for s in levels}
    cs={s:np.bincount(types[strata==s],minlength=16) for s in levels}
    edge_strata=np.sort(strata[edges],axis=1)
    uv,number=np.unique(edge_strata,axis=0,return_counts=True)
    for (s,t),m in zip(uv,number):
        for j,(a,b) in enumerate(PAIRS):
            if s==t:
                q=cs[s][a]*(cs[s][b]-(a==b))*(1 if a==b else 2)/(ns[s]*(ns[s]-1))
            else:
                q=(cs[s][a]*cs[t][b]+(0 if a==b else cs[s][b]*cs[t][a]))/(ns[s]*ns[t])
            result[j]+=m*q
    return result

@njit(nogil=True,cache=True)
def permute_counts(types,edges,lookup,iterations,seed,strata):
    """Shuffle labels within each FOV (or degree stratum); never resample edges."""
    np.random.seed(seed)
    n_pairs=int(lookup.max())+1
    out=np.zeros((iterations,n_pairs),np.int32)
    groups=[np.where(strata==s)[0] for s in np.unique(strata)]
    for r in range(iterations):
        shuffled=types.copy()
        for indices in groups:
            vals=shuffled[indices].copy()
            np.random.shuffle(vals)
            shuffled[indices]=vals
        for e in range(len(edges)):
            j=lookup[shuffled[edges[e,0]],shuffled[edges[e,1]]]
            out[r,j]+=1
    return out

def work(item,iterations,seed,degree_iterations):
    path,pid,group,index=item
    ids,types,edges,rows=read_graph(path)
    observed=np.bincount(LOOKUP[types[edges[:,0]],types[edges[:,1]]],minlength=len(PAIRS))
    expected=expectation(types,edges)
    null=permute_counts(types,edges,LOOKUP,iterations,seed+index,np.zeros(len(types),np.int64))
    degree=np.bincount(edges.ravel(),minlength=len(types))
    # Degree bins fixed a priori; spatial edge/degree boundaries retained.
    strata=np.digitize(degree,[4,6,8]).astype(np.int64)
    d_expected=expectation(types,edges,strata=strata)
    d_null=permute_counts(types,edges,LOOKUP,degree_iterations,seed+100000+index,strata)
    metadata=dict(patient_id=pid,group=group,fov=re.search(r'FOV(\d+)',path.name)[1],
                  graph_file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  graph_rows=rows,nodes=len(types),edges=len(edges),duplicate_rows_removed=rows-len(edges),
                  degree_mean=float(degree.mean()),degree_max=int(degree.max()),
                  degree_le_3_fraction=float(np.mean(degree<=3)))
    return metadata,observed,expected,null,d_expected,d_null

def compare_groups(frame,value,scope):
    results=[]
    for a,b in PAIRS:
        z=frame[(frame.code_a==a)&(frame.code_b==b)]
        x=z.loc[z.group=='NN',value].to_numpy(); y=z.loc[z.group=='NP',value].to_numpy()
        assert len(x)==20 and len(y)==18
        res=mannwhitneyu(x,y,alternative='two-sided',method='asymptotic',use_continuity=True)
        row=dict(code_a=a,code_b=b,cell_type_a=NAMES[a],cell_type_b=NAMES[b],
                 interaction=f'{NAMES[a]} — {NAMES[b]}',n_NN=len(x),n_NP=len(y),
                 median_NN=np.median(x),q25_NN=np.quantile(x,.25),q75_NN=np.quantile(x,.75),
                 median_NP=np.median(y),q25_NP=np.quantile(y,.25),q75_NP=np.quantile(y,.75),
                 median_difference_NN_minus_NP=np.median(x)-np.median(y),
                 rank_biserial_NN_minus_NP=2*res.statistic/(len(x)*len(y))-1,
                 U_NN=res.statistic,p_raw=res.pvalue,value=value,family=scope)
        results.append(row)
    out=pd.DataFrame(results);out['p_holm']=holm(out.p_raw)
    return out

def spatial_results(patient_ids,groups,observed,expected,null,pseudo,seed,scope):
    rng=np.random.default_rng(seed)
    rows=[]
    for group in ['NN','NP']:
        idx=np.where(groups==group)[0]
        ratios=(observed[idx]+pseudo)/(expected[idx]+pseudo)
        null_ratios=(null[idx]+pseudo)/(expected[idx,None,:]+pseudo)
        stat=ratios.mean(axis=0); null_stat=null_ratios.mean(axis=0)
        boot=ratios[rng.integers(0,len(idx),size=(10000,len(idx)))].mean(axis=1)
        low,high=np.quantile(boot,[.025,.975],axis=0)
        count=np.sum(null_stat>=stat,axis=0)
        count2=np.sum(np.abs(null_stat-1)>=np.abs(stat-1),axis=0)
        for j,(a,b) in enumerate(PAIRS):
            rows.append(dict(group=group,n_patients=len(idx),code_a=a,code_b=b,
                             cell_type_a=NAMES[a],cell_type_b=NAMES[b],interaction=f'{NAMES[a]} — {NAMES[b]}',
                             mean_enrichment=stat[j],median_enrichment=np.median(ratios[:,j]),
                             bootstrap95_low=low[j],bootstrap95_high=high[j],
                             observed_sum=int(observed[idx,j].sum()),expected_sum=float(expected[idx,j].sum()),
                             n_patients_expected_ge5=int(np.sum(expected[idx,j]>=5)),
                             supported_for_ranking=bool(np.sum(expected[idx,j]>=5)>=int(np.ceil(len(idx)/2))),
                             null_stat95_low=np.quantile(null_stat[:,j],.025),null_stat95_high=np.quantile(null_stat[:,j],.975),
                             exceedances_upper=int(count[j]),p_upper=(count[j]+1)/(null.shape[1]+1),
                             p_two_sided=(count2[j]+1)/(null.shape[1]+1),
                             permutations=null.shape[1],pseudocount=pseudo,family=scope))
    out=pd.DataFrame(rows)
    out['p_upper_holm']=holm(out.p_upper);out['p_two_sided_holm']=holm(out.p_two_sided)
    out['rank_supported_mean_enrichment']=np.nan
    for group in ['NN','NP']:
        ix=(out.group==group)&out.supported_for_ranking
        out.loc[ix,'rank_supported_mean_enrichment']=out.loc[ix,'mean_enrichment'].rank(ascending=False,method='min')
    return out

def plots(frame,stats,spatial,out):
    colors={'NN':'#356aa0','NP':'#d57940'}
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,(a,b) in zip(axs,TARGETS):
        z=frame[(frame.code_a==a)&(frame.code_b==b)]
        for i,g in enumerate(['NN','NP']):
            x=z.loc[z.group==g,'proportion'].to_numpy()*100
            ax.scatter(i+np.linspace(-.15,.15,len(x)),x,c=colors[g],s=25,alpha=.8)
            ax.plot([i-.2,i+.2],[np.median(x)]*2,c='black')
        p=stats[(stats.code_a==a)&(stats.code_b==b)].iloc[0].p_holm
        ax.set(title=f'{NAMES[a]} / {NAMES[b]}\nHolm p={p:.3g}',xticks=[0,1],xticklabels=['NN','NP'],ylabel='Undirected edge proportion (%)')
    fig.suptitle('Handoff candidate interactions; exact S2 targets unverified',fontsize=11)
    fig.tight_layout();fig.savefig(out/'candidate_patient_proportions.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,(a,b) in zip(axs,TARGETS):
        z=frame[(frame.code_a==a)&(frame.code_b==b)]
        for i,g in enumerate(['NN','NP']):
            x=z.loc[z.group==g,'enrichment'].to_numpy()
            ax.scatter(i+np.linspace(-.15,.15,len(x)),x,c=colors[g],alpha=.8)
            row=spatial[(spatial.code_a==a)&(spatial.code_b==b)&(spatial.group==g)].iloc[0]
            ax.errorbar(i,row.mean_enrichment,yerr=np.array([[row.mean_enrichment-row.bootstrap95_low],[row.bootstrap95_high-row.mean_enrichment]]),fmt='ks',capsize=4)
        ax.axhline(1,c='gray',ls='--');ax.set(title=f'{NAMES[a]} / {NAMES[b]}',xticks=[0,1],xticklabels=['NN','NP'],ylabel='Observed / spatial-null expected')
    fig.suptitle('Patient ratios; squares = equal-patient mean, bars = bootstrap 95% CI',fontsize=11)
    fig.tight_layout();fig.savefig(out/'candidate_patient_enrichment.png',dpi=180);plt.close(fig)
    for group in ['NN','NP']:
        z=spatial[(spatial.group==group)&spatial.supported_for_ranking].sort_values('mean_enrichment',ascending=False).head(25)
        fig,ax=plt.subplots(figsize=(9,9));ax.barh(z.interaction[::-1],z.mean_enrichment[::-1],color=colors[group]);ax.axvline(1,color='black',ls='--');ax.set(xlabel='Equal-patient mean observed / expected',title=f'{group}: top 25 sufficiently supported interactions (descriptive rank)');fig.tight_layout();fig.savefig(out/f'{group}_enrichment_ranking.png',dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--graphs',type=Path,default=GRAPH_DIR);parser.add_argument('--labels',type=Path,default=LABELS);parser.add_argument('--permutations',type=int,default=10000);parser.add_argument('--degree-permutations',type=int,default=10000);parser.add_argument('--seed',type=int,default=20261109);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();out=ROOT/'interactions';out.mkdir(exist_ok=True)
    assert args.permutations>=10000 and args.degree_permutations>=10000
    labels=pd.read_csv(args.labels,header=None,names=['uid','group']).set_index('uid')['group']
    items=[]
    for p in sorted(args.graphs.glob('Patient_*.txt')):
        pid=int(re.search(r'Patient_(\d+)',p.name)[1]);g=labels.get('Melanoma'+str(pid))
        if g in ['NN','NP']:items.append((p,pid,g,len(items)))
    assert len(items)==76 and len(set(x[1] for x in items))==38
    # Compile once before concurrent dispatch; seeds are deterministic by sorted FOV index.
    permute_counts(np.array([0,1],np.int64),np.array([[0,1]],np.int64),LOOKUP,1,args.seed,np.array([0,0],np.int64))
    acc={};fov_rows=[];fov_edges=[];start=time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for n,res in enumerate(executor.map(lambda x:work(x,args.permutations,args.seed,args.degree_permutations),items),1):
            meta,obs,exp,null,dexp,dnull=res;pid=meta['patient_id'];fov_rows.append(meta)
            if pid not in acc:acc[pid]=[meta['group'],obs.copy(),exp.copy(),null.copy(),dexp.copy(),dnull.copy()]
            else:
                for j,v in enumerate([obs,exp,null,dexp,dnull],1):acc[pid][j]+=v
            for j,(a,b) in enumerate(PAIRS):fov_edges.append(dict(patient_id=pid,group=meta['group'],fov=meta['fov'],code_a=a,code_b=b,observed=int(obs[j]),expected=exp[j],proportion=obs[j]/meta['edges']))
            print(f'Completed {n}/{len(items)} FOVs; {time.time()-start:.1f}s',flush=True)
    ids=np.array(sorted(acc));groups=np.array([acc[i][0] for i in ids]);observed=np.stack([acc[i][1] for i in ids]);expected=np.stack([acc[i][2] for i in ids]);null=np.stack([acc[i][3] for i in ids]);dexpected=np.stack([acc[i][4] for i in ids]);dnull=np.stack([acc[i][5] for i in ids])
    np.savez_compressed(out/'patient_null_counts.npz',patient_ids=ids,groups=groups,observed=observed,expected=expected,null=null,degree_expected=dexpected,degree_null=dnull,pairs=np.array(PAIRS))
    pd.DataFrame(fov_rows).to_csv(out/'fov_manifest.csv',index=False);fov_df=pd.DataFrame(fov_edges);fov_df.to_csv(out/'fov_edges_audit.csv',index=False)
    rows=[]
    for i,pid in enumerate(ids):
        for j,(a,b) in enumerate(PAIRS):
            rows.append(dict(patient_id=int(pid),group=groups[i],code_a=a,code_b=b,cell_type_a=NAMES[a],cell_type_b=NAMES[b],observed=int(observed[i,j]),expected=expected[i,j],total_edges=int(observed[i].sum()),proportion=observed[i,j]/observed[i].sum(),enrichment=(observed[i,j]+.5)/(expected[i,j]+.5),null_count95_low=np.quantile(null[i,:,j],.025),null_count95_high=np.quantile(null[i,:,j],.975),null_mc_mean=null[i,:,j].mean(),null_mc_sd=null[i,:,j].std(ddof=1),degree_expected=dexpected[i,j],degree_enrichment=(observed[i,j]+.5)/(dexpected[i,j]+.5)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'patient_edges.csv',index=False)
    stats=compare_groups(frame,'proportion','120 eligible interaction proportions');stats.to_csv(out/'stats_group_proportions.csv',index=False)
    compare_groups(frame,'enrichment','120 exploratory enrichment-score group comparisons').to_csv(out/'stats_group_enrichment.csv',index=False)
    spatial=spatial_results(ids,groups,observed,expected,null,.5,args.seed+200000,'240 spatial enrichment tests: 120 pairs x 2 groups');spatial.to_csv(out/'stats_spatial_enrichment.csv',index=False)
    spatial_results(ids,groups,observed,dexpected,dnull,.5,args.seed+300000,'240 exploratory degree-stratified spatial tests').to_csv(out/'stats_degree_stratified.csv',index=False)
    for pseudo in [.1,1.]:
        spatial_results(ids,groups,observed,expected,null,pseudo,args.seed+400000,f'240 exploratory pseudocount {pseudo} tests').to_csv(out/f'stats_pseudocount_{pseudo:g}.csv',index=False)
    # Equal-FOV proportions within each patient are a sensitivity, still one row per patient.
    sens=fov_df.groupby(['patient_id','group','code_a','code_b'],as_index=False).proportion.mean()
    compare_groups(sens,'proportion','120 exploratory equal-FOV-within-patient tests').to_csv(out/'stats_equal_fov_weight_sensitivity.csv',index=False)
    # Historical symmetric-matrix entry: diagonal O/m, off-diagonal O/(2m).
    frame['historical_matrix_entry']=frame.proportion*np.where(frame.code_a==frame.code_b,1,.5)
    compare_groups(frame,'historical_matrix_entry','120 historical double-entry convention tests').to_csv(out/'stats_historical_matrix_sensitivity.csv',index=False)
    plots(frame,stats,spatial,out)
    meta=dict(graph_dir=str(args.graphs),labels=str(args.labels),permutations=args.permutations,degree_permutations=args.degree_permutations,seed=args.seed,pseudocount=.5,eligible_types=[NAMES[i] for i in ELIGIBLE],tested_pairs=len(PAIRS),target_status='Exact S2 manuscript targets not located. All 120 eligible pairs tested; handoff examples highlighted without target-only correction.',statistic='Equal-patient arithmetic mean observed/expected enrichment ratio',spatial_alternative='Upper tail for enrichment; two-sided deviations around 1 also provided',spatial_holm_family=2*len(PAIRS),group_holm_family=len(PAIRS),bootstrap='10000 patient resamples within each disease group, percentile 95% CI of mean ratio',degree_bins='0-3,4-5,6-7,>=8',versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),elapsed_seconds=time.time()-start)
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('DONE',json.dumps(meta),flush=True)

if __name__=='__main__':main()
