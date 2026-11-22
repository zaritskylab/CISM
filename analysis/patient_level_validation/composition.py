"""Patient-level Fig. 2B composition validation; no cell/FOV-level inference.

Run from repository root: python analysis/patient_level_validation/composition.py
Primary uses the original figure's retained graph vertices.
All raw cells and current tutorial graph vertices are explicit sensitivities.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cism_patient_validation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu
import scipy

ROOT = Path(__file__).resolve().parents[2]
CELLS = Path('/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv')
GRAPHS = ROOT / 'melanoma_tutorials/pet_data/Melanoma'
ORIGINAL_GRAPHS = Path('/groups/assafza_group/assafza/Omri/processed_data_23_10_2025/data/melanoma_distance_100_XX')
ORIGINAL_TYPES = dict(enumerate(['b cell','cd4 apc','cd4 t cell','treg','cd8 t cell','DC','gc','hevs','mac','memory%cd4 t cell','nk cell','neutrophil','stroma','tumor','unidentified','blood vessels']))

def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result

def tests(long, value='proportion', scale=100):
    rows = []
    for cell_type, table in long.groupby('cell_type', sort=True):
        assert not table.patient_id.duplicated().any()
        x = table.loc[table.group=='NN', value].to_numpy() * scale
        y = table.loc[table.group=='NP', value].to_numpy() * scale
        u, p = mannwhitneyu(x,y,alternative='two-sided',method='asymptotic',use_continuity=True)
        xq, yq = np.quantile(x,[.25,.5,.75]), np.quantile(y,[.25,.5,.75])
        rows.append(dict(cell_type=cell_type,n_NN=len(x),n_NP=len(y),
            NN_q25=xq[0],NN_median=xq[1],NN_q75=xq[2],NN_IQR=xq[2]-xq[0],
            NP_q25=yq[0],NP_median=yq[1],NP_q75=yq[2],NP_IQR=yq[2]-yq[0],
            median_difference_NN_minus_NP=xq[1]-yq[1],U_NN=u,
            rank_biserial_NN_minus_NP=2*u/(len(x)*len(y))-1,p_raw=p,
            n_zero_NN=int((x==0).sum()),n_zero_NP=int((y==0).sum()),
            n_distinct_pooled=len(np.unique(np.r_[x,y]))))
    result=pd.DataFrame(rows)
    result['p_holm']=holm(result.p_raw)
    return result

def global_test(long, permutations, seed):
    wide=long.pivot(index=['patient_id','group'],columns='cell_type',values='proportion')
    assert not wide.isna().any().any()
    a=wide.to_numpy()
    mask=wide.index.get_level_values('group')=='NN'
    n=int(mask.sum())
    observed=jensenshannon(a[mask].mean(0),a[~mask].mean(0),base=2)
    rng=np.random.default_rng(seed)
    null=np.empty(permutations)
    for k in range(permutations):
        order=rng.permutation(len(a))
        null[k]=jensenshannon(a[order[:n]].mean(0),a[order[n:]].mean(0),base=2)
    exceed=int((null>=observed-1e-15).sum())
    return dict(distance_base2=float(observed),p_empirical=(exceed+1)/(permutations+1),
                upper_tail_exceedances=exceed,permutations=permutations,seed=seed,
                n_NN=n,n_NP=int((~mask).sum()),null_95th_percentile=float(np.quantile(null,.95))),null

def aggregate(cells, types):
    labels=cells.groupby('patient number').Group.first().rename('group')
    counts=pd.crosstab(cells['patient number'],cells.pred_mapped).reindex(columns=types,fill_value=0)
    counts.index.name='patient_id'
    long=counts.stack().rename('count').reset_index().rename(columns={'pred_mapped':'cell_type'})
    long['group']=long.patient_id.map(labels)
    total=counts.sum(axis=1)
    assert (total>0).all()
    long['total_eligible_cells']=long.patient_id.map(total)
    long['proportion']=long['count']/long.total_eligible_cells
    long['percentage']=100*long.proportion
    assert np.allclose(long.groupby('patient_id').proportion.sum(),1)
    return long

def plot(long, result, path, ylabel='Eligible cells (%)', value='percentage'):
    fig, axes=plt.subplots(4,4,figsize=(15,12),squeeze=False)
    rng=np.random.default_rng(45)
    for ax,(cell_type,table) in zip(axes.flat,long.groupby('cell_type',sort=True)):
        arrays=[table.loc[table.group==g,value].to_numpy() for g in ['NN','NP']]
        ax.boxplot(arrays,positions=[0,1],widths=.45,showfliers=False)
        for i,(arr,color) in enumerate(zip(arrays,['#4c78a8','#e58a45'])):
            ax.scatter(i+rng.uniform(-.14,.14,len(arr)),arr,s=19,color=color,alpha=.8,zorder=3)
        q=float(result.loc[result.cell_type==cell_type,'p_holm'].iloc[0])
        ax.set_title(f'{cell_type}\nHolm p={q:.3g}',fontsize=10)
        ax.set_xticks([0,1],['NN','NP']);ax.set_ylabel(ylabel,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    for ax in list(axes.flat)[long.cell_type.nunique():]:ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)

def run_variant(cells, types, out, permutations, seed):
    out.mkdir(parents=True,exist_ok=True)
    long=aggregate(cells,types)
    long.to_csv(out/'patient_composition.csv',index=False)
    result=tests(long)
    result.to_csv(out/'rank_sum_tests.csv',index=False)
    global_result,null=global_test(long,permutations,seed)
    (out/'global_jensen_shannon_test.json').write_text(json.dumps(global_result,indent=2)+'\n')
    np.save(out/'global_permutation_null.npy',null)
    plot(long,result,out/'patient_composition')
    return long,result,global_result

def graph_subset(cells, graph_dir, original=False):
    """Read vertex types directly: graph vertex IDs are FOV row positions, not cell_id."""
    matched=[]; audit=[]
    for (patient,fov),table in cells.groupby(['patient number','fov']):
        fov_suffix=fov.replace('FOV','') if original else fov
        path=graph_dir/f'Patient_{patient}_FOV{fov_suffix}.txt'
        if not path.exists():raise FileNotFoundError(path)
        edges=np.loadtxt(path,dtype=np.int64,ndmin=2)
        pairs=np.concatenate([edges[:,[0,2]],edges[:,[1,3]]])
        vertices=np.unique(pairs,axis=0)
        assert len(vertices)==len(np.unique(vertices[:,0])), 'Vertex has conflicting types'
        mapping=ORIGINAL_TYPES if original else dict(enumerate(sorted(ORIGINAL_TYPES.values())))
        graph_types={int(i):mapping[int(t)] for i,t in vertices}
        kept=pd.DataFrame({'patient number':patient,'Group':table.Group.iloc[0],
                          'fov':fov,'cell_id':vertices[:,0],
                          'pred_mapped':[mapping[int(t)] for t in vertices[:,1]]})
        raw_counts=table.pred_mapped.value_counts()
        retained_counts=kept.pred_mapped.value_counts()
        delta=raw_counts.subtract(retained_counts,fill_value=0)
        assert (delta>=0).all(), f'Graph has more cells of a type than raw FOV: {patient} {fov}'
        matched.append(kept)
        audit.append(dict(patient_id=int(patient),group=table.Group.iloc[0],fov=fov,
                          raw_cells=len(table),graph_vertices=len(kept),omitted_cells=len(table)-len(kept),
                          raw_tumor_cells=int(raw_counts.get('tumor',0)),retained_tumor_cells=int(retained_counts.get('tumor',0))))
    return pd.concat(matched,ignore_index=True),pd.DataFrame(audit)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells-csv',type=Path,default=CELLS)
    ap.add_argument('--graph-dir',type=Path,default=GRAPHS)
    ap.add_argument('--original-graph-dir',type=Path,default=ORIGINAL_GRAPHS)
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).with_name('composition'))
    ap.add_argument('--permutations',type=int,default=20000)
    ap.add_argument('--seed',type=int,default=20261109)
    ap.add_argument('--fov-areas-csv',type=Path,help='Verified areas: patient_id,fov,area_mm2; never inferred from centroids')
    args=ap.parse_args()
    assert args.permutations>=10000
    cells=pd.read_csv(args.cells_csv,usecols=['Group','patient number','fov','cell_id','pred_mapped'])
    all_types=sorted(cells.pred_mapped.unique())
    cells=cells[cells.Group.isin(['NN','NP'])].copy()
    assert cells.groupby('patient number').Group.nunique().max()==1
    assert cells.groupby('fov')['patient number'].nunique().max()==1
    assert not cells.duplicated(['patient number','fov','cell_id']).any()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fovs=cells.groupby(['patient number','Group','fov']).size().rename('raw_cells').reset_index()
    fovs.to_csv(args.output_dir/'fov_cell_counts.csv',index=False)
    original_cells,original_audit=graph_subset(cells,args.original_graph_dir,original=True)
    assert 'tumor' not in set(original_cells.pred_mapped), 'Original tumor exclusion expectation violated'
    eligible_types=[t for t in all_types if t!='tumor']
    original_audit.to_csv(args.output_dir/'original_graph_node_coverage.csv',index=False)
    primary,result,global_result=run_variant(original_cells,eligible_types,args.output_dir/'primary_original_graph_retained_cells',args.permutations,args.seed)
    run_variant(cells,all_types,args.output_dir/'sensitivity_all_raw_cells',args.permutations,args.seed)
    graph_cells,audit=graph_subset(cells,args.graph_dir)
    audit.to_csv(args.output_dir/'graph_node_coverage.csv',index=False)
    run_variant(graph_cells,all_types,args.output_dir/'sensitivity_tutorial_graph_nodes',args.permutations,args.seed)
    run_variant(cells[cells.pred_mapped!='tumor'],[t for t in all_types if t!='tumor'],
                args.output_dir/'sensitivity_without_tumor',args.permutations,args.seed)
    if args.fov_areas_csv:
        areas=pd.read_csv(args.fov_areas_csv)
        assert not areas.duplicated(['patient_id','fov']).any()
        merged=fovs.rename(columns={'patient number':'patient_id'}).merge(areas,on=['patient_id','fov'],how='left',validate='one_to_one')
        assert merged.area_mm2.notna().all() and (merged.area_mm2>0).all()
        total_area=merged.groupby('patient_id').area_mm2.sum()
        density=primary.copy();density['area_mm2']=density.patient_id.map(total_area)
        density['cells_per_mm2']=density['count']/density.area_mm2
        density.to_csv(args.output_dir/'patient_density.csv',index=False)
        density_tests=tests(density,'cells_per_mm2',1)
        density_tests.to_csv(args.output_dir/'density_rank_sum_tests.csv',index=False)
        plot(density,density_tests,args.output_dir/'patient_density','Cells / mm²','cells_per_mm2')
    meta=dict(cells_csv=str(args.cells_csv),cells_sha256=hashlib.sha256(args.cells_csv.read_bytes()).hexdigest(),
        original_graph_dir=str(args.original_graph_dir),tutorial_graph_dir=str(args.graph_dir),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        patients=cells.groupby('Group')['patient number'].nunique().to_dict(),
        fovs=cells.groupby('Group').fov.nunique().to_dict(),cells=cells.groupby('Group').size().to_dict(),
        cell_types_primary=eligible_types,original_graph_omitted_cells=int(original_audit.omitted_cells.sum()),tutorial_graph_omitted_cells=int(audit.omitted_cells.sum()),
        eligible_cells=original_cells.groupby('Group').size().to_dict(),
        eligibility='Original graph endpoints only; NN/NP raw cells have no tumor annotation; three raw cells are absent from graph endpoints; tumor-cleanup code exists but its exact execution version for these graph files is uncertain',
        density_status='completed_verified_area_file' if args.fov_areas_csv else 'not_run_no_verified_physical_FOV_area',
        rank_sum_method='two-sided Mann–Whitney U; asymptotic normal approximation with tie correction and continuity correction; Holm over all 15 eligible non-tumor primary types; sensitivity families separately 16/15 types',
        effect_size='2 U_NN / (n_NN n_NP) - 1; positive means larger NN values',
        quantile_method='numpy linear interpolation; percentages; median difference NN minus NP in percentage points',
        global_method='Base-2 Jensen–Shannon distance between equally weighted patient-mean compositions; upper tail with plus-one correction; whole-patient labels shuffled preserving 20/18 group sizes',
        seed=args.seed,permutations=args.permutations)
    (args.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(result[['cell_type','NN_median','NP_median','p_raw','p_holm']].to_string(index=False))
    print(json.dumps(global_result,indent=2))
    print('Omitted original graph cells:',int(original_audit.omitted_cells.sum()))
    print('Omitted tutorial graph cells:',int(audit.omitted_cells.sum()))

if __name__=='__main__':main()
