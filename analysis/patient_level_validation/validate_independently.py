#!/usr/bin/env python3
"""Separate implementation of patient aggregation and tie-corrected rank tests.

This script deliberately does not import the analysis implementation. Run from
any directory with the same Python environment as the primary analysis.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from math import erfc, sqrt
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / 'validation'
LABELS = ROOT / 'melanoma_tutorials/pet_data/Melanoma/patient_class.csv'
GRAPH = Path('/groups/assafza_group/assafza/Omri/processed_data_23_10_2025/data/melanoma_distance_100_XX')
RAW = Path('/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv')
TYPES = ['b cell','cd4 apc','cd4 t cell','treg','cd8 t cell','DC','gc','hevs','mac','memory%cd4 t cell','nk cell','neutrophil','stroma','tumor','unidentified','blood vessels']


def rank_test(x, y):
    """Pairwise U, pooled tie variance and continuity-corrected normal p."""
    x, y = np.asarray(x), np.asarray(y)
    delta = x[:, None] - y[None, :]
    u = np.count_nonzero(delta > 0) + .5 * np.count_nonzero(delta == 0)
    n, m = len(x), len(y)
    ties = Counter(np.concatenate([x, y]))
    correction = sum(t**3-t for t in ties.values())
    variance = n*m/12 * (n+m+1-correction/((n+m)*(n+m-1)))
    p = 1. if variance <= 0 else erfc(max(0., abs(u-n*m/2)-.5)/sqrt(2*variance))
    return u, p, 2*u/(n*m)-1


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = np.empty(len(values))
    cumulative = 0.
    for rank, position in enumerate(order):
        cumulative = max(cumulative, min(1., values[position]*(len(values)-rank)))
        adjusted[position] = cumulative
    return adjusted


def js_distance(a, b):
    a, b = a/a.sum(), b/b.sum()
    midpoint = (a+b)/2
    positive_a, positive_b = a > 0, b > 0
    return np.sqrt((np.sum(a[positive_a]*np.log2(a[positive_a]/midpoint[positive_a])) + np.sum(b[positive_b]*np.log2(b[positive_b]/midpoint[positive_b])))/2)


def compare_primary():
    """Compare materialized primary results, retaining a machine-readable audit."""
    checks = []
    original_labels = pd.read_csv(GRAPH.parent.parent/'patient_class.csv',header=None,names=['uid','group']).sort_values('uid').reset_index(drop=True)
    current_labels = pd.read_csv(LABELS,header=None,names=['uid','group']).sort_values('uid').reset_index(drop=True)
    assert original_labels.equals(current_labels)
    checks.append({'check':'Original source patient labels equal tutorial patient labels','passed':True,'rows':len(original_labels)})
    own_comp = pd.read_csv(OUT/'independent_patient_composition.csv')
    own_edge = pd.read_csv(OUT/'independent_patient_edges.csv')
    own_stats = pd.read_csv(OUT/'independent_rank_tests.csv')
    def compare(name, own, primary, keys, columns):
        merged = own.merge(primary,on=keys,suffixes=('_independent','_primary'),validate='one_to_one',how='outer',indicator=True)
        assert merged._merge.eq('both').all(), name
        for col in columns:
            delta = float(np.max(np.abs(merged[col+'_independent']-merged[col+'_primary'])))
            passed = bool(np.allclose(merged[col+'_independent'],merged[col+'_primary'],rtol=1e-11,atol=1e-12))
            checks.append({'check':name+': '+col,'passed':passed,'max_absolute_difference':delta,'rows':len(merged)})
            assert passed, checks[-1]
    cp = BASE/'composition/primary_original_graph_retained_cells'
    ep = BASE/'interactions'
    if (cp/'patient_composition.csv').exists():
        compare('Composition aggregation',own_comp,pd.read_csv(cp/'patient_composition.csv'),['patient_id','cell_type'],['count','proportion'])
        compare('Composition rank tests',own_stats[own_stats.family.eq('composition')],pd.read_csv(cp/'rank_sum_tests.csv'),['cell_type'],['U_NN','p_raw','p_holm','rank_biserial_NN_minus_NP'])
        independent=json.loads((OUT/'independent_global_test.json').read_text())
        primary=json.loads((cp/'global_jensen_shannon_test.json').read_text())
        for key in ['distance_base2','p_empirical']:
            delta=abs(independent[key]-primary[key])
            checks.append({'check':'Global JS '+key,'passed':delta<1e-12,'max_absolute_difference':delta,'rows':20000})
            assert delta<1e-12
    if (ep/'patient_edges.csv').exists():
        own_edge['code_a']=own_edge.cell_type_a.map({t:i for i,t in enumerate(TYPES)})
        own_edge['code_b']=own_edge.cell_type_b.map({t:i for i,t in enumerate(TYPES)})
        compare('Undirected graph aggregation',own_edge,pd.read_csv(ep/'patient_edges.csv'),['patient_id','code_a','code_b'],['observed','expected','proportion','enrichment'])
        rank=own_stats[own_stats.family.eq('edges')].copy()
        rank['code_a']=rank.cell_type_a.map({t:i for i,t in enumerate(TYPES)})
        rank['code_b']=rank.cell_type_b.map({t:i for i,t in enumerate(TYPES)})
        compare('Interaction rank tests',rank,pd.read_csv(ep/'stats_group_proportions.csv'),['code_a','code_b'],['U_NN','p_raw','p_holm','rank_biserial_NN_minus_NP'])
        z=np.load(ep/'patient_null_counts.npz')
        # Sum over interaction types for each null realization must equal the
        # number of eligible undirected edges for that patient, exactly.
        assert np.all(z['null'].sum(axis=2)==z['observed'].sum(axis=1)[:,None])
        assert np.all(z['degree_null'].sum(axis=2)==z['observed'].sum(axis=1)[:,None])
        checks.append({'check':'Both null schemes conserve all patient edges in every realization','passed':True,'rows':int(z['null'].shape[0]*z['null'].shape[1]*2)})
        sd=z['null'].std(axis=1,ddof=1)
        se=sd/np.sqrt(z['null'].shape[1])
        mean_error=np.abs(z['null'].mean(axis=1)-z['expected'])
        standardized=np.divide(mean_error,se,out=np.zeros_like(se),where=se>0)
        # Diagnostic, not a test selected for statistical significance.
        checks.append({'check':'Monte Carlo mean vs exact analytic expectation (max standard errors)','passed':bool(standardized.max()<6),'max_absolute_difference':float(standardized.max()),'rows':int(sd.size)})
        spatial=pd.read_csv(ep/'stats_spatial_enrichment.csv')
        bootstrap_rng=np.random.default_rng(20261109+200000)
        for group in ['NN','NP']:
            ix=z['groups']==group
            ratios=(z['observed'][ix]+.5)/(z['expected'][ix]+.5)
            null=((z['null'][ix]+.5)/(z['expected'][ix,None,:]+.5)).mean(axis=0)
            obs=ratios.mean(axis=0)
            p=(1+(null>=obs).sum(axis=0))/(1+null.shape[0])
            primary=spatial[spatial.group.eq(group)].sort_values(['code_a','code_b'])
            assert np.allclose(primary.mean_enrichment,obs,rtol=1e-12)
            assert np.array_equal(primary.exceedances_upper,(null>=obs).sum(axis=0))
            assert np.allclose(primary.p_upper,p,rtol=1e-12)
            bootstrap=np.empty((10000,ratios.shape[1]))
            for replicate in range(10000):
                sample=bootstrap_rng.integers(len(ratios),size=len(ratios))
                bootstrap[replicate]=ratios[sample].sum(axis=0)/len(ratios)
            low, high=np.quantile(bootstrap,[.025,.975],axis=0)
            assert np.allclose(primary.bootstrap95_low,low,rtol=1e-12)
            assert np.allclose(primary.bootstrap95_high,high,rtol=1e-12)
        assert np.allclose(spatial.p_upper_holm,holm(spatial.p_upper.to_numpy()),rtol=1e-12)
        checks.append({'check':'Hierarchical spatial statistic, upper-tail exceedances and 240-test Holm','passed':True,'rows':len(spatial)})
        checks.append({'check':'Independent patient bootstrap reproduces both group confidence intervals','passed':True,'rows':20000})
        modern=pd.read_csv(ep/'stats_group_proportions.csv')
        historical=pd.read_csv(ep/'stats_historical_matrix_sensitivity.csv')
        assert np.array_equal(modern.U_NN,historical.U_NN)
        assert np.allclose(modern.p_holm,historical.p_holm,rtol=1e-12)
        checks.append({'check':'Historical diagonal convention preserves all within-pair rank tests','passed':True,'rows':120})
    (OUT/'primary_comparison_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


def validate_permutation_engine():
    """Exhaustively check a tiny path, then test the production shuffle engine."""
    from itertools import permutations
    import interactions as production
    labels = np.array([0,0,9,9],dtype=np.int64)
    edge_list = np.array([[0,1],[1,2],[2,3]],dtype=np.int64)
    all_labels = sorted(set(permutations(labels.tolist())))
    exact=[]
    for lab in all_labels:
        counts=Counter(tuple(sorted((lab[u],lab[v]))) for u,v in edge_list)
        exact.append([counts[a,b] for a,b in production.PAIRS])
    exact_mean=np.mean(exact,axis=0)
    assert np.isclose(exact_mean[production.PAIRS.index((0,0))],.5)
    assert np.isclose(exact_mean[production.PAIRS.index((0,9))],2.)
    assert np.allclose(production.expectation(labels,edge_list),exact_mean)
    result=production.permute_counts(labels,edge_list,production.LOOKUP,20000,20261109,np.zeros(4,dtype=np.int64))
    repeated=production.permute_counts(labels,edge_list,production.LOOKUP,20000,20261109,np.zeros(4,dtype=np.int64))
    assert np.array_equal(result,repeated)
    assert np.all(result.sum(axis=1)==3)
    exact_set={tuple(row) for row in exact}
    assert all(tuple(row) in exact_set for row in result)
    assert np.max(np.abs(result.mean(axis=0)-exact_mean)) < .03
    # Separate enumeration conditional on two strata, preserving type totals.
    stratum_labels=np.array([0,9,0,9],dtype=np.int64)
    strata=np.array([0,0,1,1],dtype=np.int64)
    restricted=[]
    for first in [(0,9),(9,0)]:
        for second in [(0,9),(9,0)]:
            lab=first+second
            counts=Counter(tuple(sorted((lab[u],lab[v]))) for u,v in edge_list)
            restricted.append([counts[a,b] for a,b in production.PAIRS])
    assert np.allclose(production.expectation(stratum_labels,edge_list,strata=strata),np.mean(restricted,axis=0))
    (OUT/'permutation_engine_checks.json').write_text(json.dumps({'passed':True,'exhaustive_labelings':len(all_labels),'homotypic_exact_expectation':.5,'heterotypic_exact_expectation':2.,'permutations':20000,'identical_repeated_seed':True,'max_mean_vs_exact_difference':float(np.max(np.abs(result.mean(axis=0)-exact_mean)))},indent=2))


def main():
    OUT.mkdir(exist_ok=True)
    classes = pd.read_csv(LABELS, header=None, names=['uid','group'])
    classes['patient_id'] = classes.uid.str.removeprefix('Melanoma').astype(int)
    labels = classes.set_index('patient_id')['group'].to_dict()
    eligible = {p for p, group in labels.items() if group in ['NN','NP']}
    data = pd.read_csv(RAW)
    assert set(data.pred_mapped.unique()) == set(TYPES)
    assert not data.duplicated(['patient number','fov','cell_id']).any()
    assert data.groupby('patient number').Group.nunique().eq(1).all()
    for p in eligible:
        assert set(data.loc[data['patient number'].eq(p), 'Group']) == {labels[p]}
    graph_counts = defaultdict(Counter)
    graph_expected = defaultdict(Counter)
    graph_nodes = defaultdict(Counter)
    fovs = []
    pairs = list(combinations_with_replacement([i for i in range(16) if i != 13], 2))
    for path in sorted(GRAPH.glob('Patient_*_FOV*.txt')):
        match = re.fullmatch(r'Patient_(\d+)_FOV(\d+)\.txt',path.name)
        p, fov = int(match[1]), 'FOV'+match[2]
        if p not in eligible:
            continue
        # Stream lines into a set and dictionary rather than use primary loader.
        vertices = {}
        occurrences = Counter()
        oriented = set()
        for line in path.open():
            u,v,a,b = map(int,line.split())
            assert u != v
            assert (u,v) not in oriented
            oriented.add((u,v))
            for node, code in [(u,a),(v,b)]:
                assert node not in vertices or vertices[node] == code
                vertices[node] = code
            occurrences[tuple(sorted((u,v)))] += 1
        assert set(occurrences.values()) == {2}
        assert all((v,u) in oriented for u,v in oriented)
        counts = Counter(vertices.values())
        assert counts[13] == 0, 'Original tumor-exclusion invariant'
        raw_fov = data.loc[data['patient number'].eq(p) & data.fov.eq(fov)]
        # Prepared graph IDs are zero-based row positions WITHIN each raw FOV.
        # They must not be joined to the biological `cell_id` column.
        raw_labels = dict(enumerate(raw_fov.pred_mapped))
        assert all(node in raw_labels and raw_labels[node] == TYPES[code] for node,code in vertices.items())
        observed = Counter(tuple(sorted((vertices[u],vertices[v]))) for u,v in occurrences)
        n, edges = len(vertices),len(occurrences)
        for a,b in pairs:
            graph_counts[p][a,b] += observed[a,b]
            numerator = counts[a]*(counts[a]-1) if a == b else 2*counts[a]*counts[b]
            graph_expected[p][a,b] += edges*numerator/(n*(n-1))
        graph_nodes[p].update(counts)
        assert sum(observed.values()) == edges
        assert np.isclose(sum(edges*(counts[a]*(counts[a]-1) if a == b else 2*counts[a]*counts[b])/(n*(n-1)) for a,b in pairs), edges)
        fovs.append({'patient_id':p,'group':labels[p],'fov':fov,'vertices':n,'edges':edges,'stored_rows':len(oriented)})
    fov_table = pd.DataFrame(fovs)
    assert not fov_table.fov.duplicated().any()
    selected = data.merge(fov_table[['patient_id','fov']],left_on=['patient number','fov'],right_on=['patient_id','fov'],validate='many_to_one')
    composition = []
    for p, counts in graph_nodes.items():
        for code,t in enumerate(TYPES):
            if code != 13:
                composition.append({'patient_id':p,'group':labels[p],'cell_type':t,'count':counts[code],'proportion':counts[code]/sum(counts.values())})
    composition = pd.DataFrame(composition)
    assert composition.groupby('patient_id').proportion.sum().sub(1).abs().max() < 1e-12
    assert set(composition.patient_id) == eligible
    composition.to_csv(OUT/'independent_patient_composition.csv',index=False)
    wide = composition.pivot(index=['patient_id','group'],columns='cell_type',values='proportion')
    array = wide.to_numpy()
    group = wide.index.get_level_values('group').to_numpy()
    n_nn = sum(group == 'NN')
    distance = js_distance(array[group == 'NN'].mean(0),array[group == 'NP'].mean(0))
    rng = np.random.default_rng(20261109)
    null = np.empty(20000)
    for i in range(len(null)):
        order = rng.permutation(len(array))
        null[i] = js_distance(array[order[:n_nn]].mean(0),array[order[n_nn:]].mean(0))
    (OUT/'independent_global_test.json').write_text(json.dumps({'distance_base2':float(distance),'p_empirical':float((1+sum(null>=distance-1e-15))/(len(null)+1)),'permutations':len(null),'seed':20261109},indent=2))
    edge_rows = []
    for p in sorted(eligible):
        total = sum(graph_counts[p].values())
        for a,b in pairs:
            observed, expected = graph_counts[p][a,b], graph_expected[p][a,b]
            edge_rows.append({'patient_id':p,'group':labels[p],'cell_type_a':TYPES[a],'cell_type_b':TYPES[b],'observed':observed,'expected':expected,'proportion':observed/total,'enrichment':(observed+.5)/(expected+.5)})
    edges = pd.DataFrame(edge_rows)
    edges.to_csv(OUT/'independent_patient_edges.csv',index=False)
    fov_table.to_csv(OUT/'independent_fov_audit.csv',index=False)
    results = []
    for family, table, key in [('composition',composition,['cell_type']),('edges',edges,['cell_type_a','cell_type_b'])]:
        records = []
        for name, sub in table.groupby(key):
            name = (name,) if isinstance(name,str) else name
            u,p,r = rank_test(sub.loc[sub.group.eq('NN'),'proportion'],sub.loc[sub.group.eq('NP'),'proportion'])
            records.append(dict(zip(key,name),family=family,U_NN=u,p_raw=p,rank_biserial_NN_minus_NP=r))
        adjusted = holm([r['p_raw'] for r in records])
        for row,p in zip(records,adjusted):
            row['p_holm'] = p
        results.extend(records)
    pd.DataFrame(results).to_csv(OUT/'independent_rank_tests.csv',index=False)
    metadata={'patients':len(eligible),'groups':classes[classes.patient_id.isin(eligible)].groupby('group').size().to_dict(),'fovs':len(fovs),'raw_cells':len(selected),'graph_vertices':int(fov_table.vertices.sum()),'unique_edges':int(fov_table.edges.sum()),'checks':'All assertions passed; unique patient/FOV mapping, labels, no loops, both orientations exactly once, edge/expectation totals, composition sums.'}
    (OUT/'independent_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
    compare_primary()


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-only',action='store_true')
    parser.add_argument('--permutation-engine-only',action='store_true')
    args=parser.parse_args()
    if args.compare_only:
        compare_primary()
    elif args.permutation_engine_only:
        validate_permutation_engine()
    else:
        main()
        validate_permutation_engine()
