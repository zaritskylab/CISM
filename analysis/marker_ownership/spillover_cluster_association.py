"""Exploratory cluster/patient associations of unidentified-cell spillover flags."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = _REPO_ROOT
BASE = ROOT/'analysis/outputs/unidentified_spillover_20748339660'
CLUSTERS = ROOT/'analysis/outputs/unidentified_neighbor_enrichment'


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty_like(adjusted); out[order] = np.minimum(adjusted, 1)
    return out


def permutation(frame, strata, iterations=10000, seed=42069):
    """Preserve flag totals within patient or FOV; exploratory exchangeability test."""
    frame = frame.reset_index(drop=True)
    groups = list(frame.groupby(strata, sort=True).indices.values())
    codes = frame.cluster_15.to_numpy(int)
    flags = frame.any_flag_q95.to_numpy(bool)
    n = np.bincount(codes, minlength=15)
    obs = np.bincount(codes[flags], minlength=15)
    expected = n*flags.mean()
    valid = (expected > 0) & (n-expected > 0)
    weight = np.zeros(15); weight[valid] = 1/expected[valid]+1/(n[valid]-expected[valid])
    statistic = np.sum((obs-expected)**2*weight)
    rng = np.random.default_rng(seed)
    sims = np.empty((iterations, 15), int)
    for i in range(iterations):
        shuffled = flags.copy()
        for ids in groups: shuffled[ids] = rng.permutation(flags[ids])
        sims[i] = np.bincount(codes[shuffled], minlength=15)
    stat_sims = ((sims-expected)**2*weight).sum(axis=1)
    result = pd.DataFrame({'cluster_15': range(15), 'observed_flags': obs,
                           'permutation_expected_flags': sims.mean(axis=0),
                           'one_sided_excess_p': (1+(sims >= obs).sum(axis=0))/(iterations+1)})
    result['excess_bh_q'] = bh(result.one_sided_excess_p)
    return result, float((1+np.count_nonzero(stat_sims >= statistic))/(iterations+1))


def main():
    output = BASE/'cluster_association'; output.mkdir(exist_ok=True)
    cells = pd.read_csv(BASE/'per_cell_summary.csv', low_memory=False)
    assignments = pd.read_csv(CLUSTERS/'cell_cluster_assignments.csv').drop(columns='Group')
    cells = cells.merge(assignments, on=['patient number', 'fov', 'cell_id'], how='left', validate='one_to_one')
    cells.to_csv(output/'cells_with_clusters.csv.gz', index=False)
    motif = cells[cells.in_motif]
    eligible = cells[cells.cluster_15.notna() & cells.status_q95.ne('unresolved')].copy()
    motif_eligible = eligible[eligible.in_motif].copy()
    rows = []
    for cluster in range(15):
        a = motif_eligible[motif_eligible.cluster_15.eq(cluster)]
        rest = motif_eligible[~motif_eligible.cluster_15.eq(cluster)]
        control = eligible[~eligible.in_motif & eligible.cluster_15.eq(cluster)]
        f, u = int(a.any_flag_q95.sum()), int((~a.any_flag_q95).sum())
        rf, ru = int(rest.any_flag_q95.sum()), int((~rest.any_flag_q95).sum())
        odds, p = fisher_exact([[f, u], [rf, ru]])
        all_cluster = motif[motif.cluster_15.eq(cluster)]
        rows.append(dict(cluster_15=cluster, motif_flagged=f, motif_low_evidence=u,
                         motif_unresolved=int(all_cluster.status_q95.eq('unresolved').sum()),
                         motif_flag_rate=f/len(a) if len(a) else np.nan,
                         control_flagged=int(control.any_flag_q95.sum()), control_assessable=len(control),
                         control_flag_rate=control.any_flag_q95.mean(),
                         flagged_patients=a.loc[a.any_flag_q95, 'patient number'].nunique(),
                         motif_patients=a['patient number'].nunique(),
                         odds_ratio_vs_other_motif_clusters=odds, fisher_p=p))
    table = pd.DataFrame(rows)
    table['fisher_bh_q'] = bh(table.fisher_p)
    perm_meta = {}
    for label, strata in [('patient', ['patient number']), ('FOV', ['patient number', 'fov'])]:
        result, global_p = permutation(motif_eligible, strata)
        result.to_csv(output/f'{label}_stratified_permutations.csv', index=False)
        perm_meta[label+'_global_p'] = global_p
    table.to_csv(output/'cluster_association.csv', index=False)
    complete = motif_eligible[motif_eligible.scored_markers.eq(35) & ~motif_eligible.multiple_dna_components.astype('boolean').fillna(True)]
    pd.crosstab(complete.cluster_15, complete.any_flag_q95).to_csv(output/'complete_case_cluster_counts.csv')
    marker_rows = []
    for _, row in motif[motif.any_flag_q95].iterrows():
        for marker in str(row.flagged_markers_q95).split(';'):
            marker_rows.append({'cluster_15': row.cluster_15, 'marker': marker,
                                'patient number': row['patient number'], 'fov': row.fov})
    markers = pd.DataFrame(marker_rows)
    pd.crosstab(markers.cluster_15, markers.marker).to_csv(output/'flagged_markers_by_cluster.csv')
    for label, keys in [('patient', ['Group', 'patient number']), ('FOV', ['Group', 'patient number', 'fov'])]:
        summary = motif.groupby(keys).agg(total=('cell_id', 'size'),
            flagged=('any_flag_q95', 'sum'), unresolved=('status_q95', lambda x: int(x.eq('unresolved').sum())))
        summary['assessable'] = summary.total-summary.unresolved
        summary['flag_rate_assessable'] = summary.flagged/summary.assessable.replace(0, np.nan)
        summary.to_csv(output/f'{label}_distribution.csv')
    centroids = pd.read_csv(CLUSTERS/'cluster_centroids_standardized.csv', index_col=0)
    centroids.loc[8].sort_values(ascending=False).to_csv(output/'cluster8_centroid.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(table.cluster_15, table.motif_low_evidence, label='Low evidence', color='#6eaa9b')
    axes[0].bar(table.cluster_15, table.motif_flagged, bottom=table.motif_low_evidence, label='Flagged', color='#b34f65')
    axes[0].bar(table.cluster_15, table.motif_unresolved, bottom=table.motif_low_evidence+table.motif_flagged, label='Unresolved', color='#afb7c0')
    axes[0].set(ylabel='Unique motif-associated cells', title='Cell counts by protein-expression cluster')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(table.cluster_15-.18, table.control_flag_rate*100, width=.36, label='Other unidentified cells', color='#9cabb6')
    axes[1].bar(table.cluster_15+.18, table.motif_flag_rate*100, width=.36, label='Motif-associated cells', color='#9672bb')
    axes[1].set(ylabel='Flagged / assessable cells (%)', title='Flag rates; small clusters can give unstable percentages')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(range(15)); ax.set_xlabel('Cluster ID (unique-cell clustering)')
    fig.suptitle('Exploratory association of spillover flags with protein-expression clusters')
    fig.tight_layout(); fig.savefig(output/'cluster_association.png', dpi=180); plt.close(fig)
    meta = dict(motif_cells=len(motif), flagged=int(motif.any_flag_q95.sum()),
                missing_cluster=int(motif.cluster_15.isna().sum()),
                assessable_clustered=len(motif_eligible), permutations=10000, seed=42069, **perm_meta)
    (output/'metadata.json').write_text(json.dumps(meta, indent=2))
    lines = ['# Spillover flags and cluster association', '',
        'Uses the existing protein clustering of unique unidentified cells, not the original motif-duplicated notebook clustering. Cluster IDs are not interchangeable with the earlier notebook IDs. No reclustering or reannotation was performed.', '',
        'Primary comparison: flagged versus low-evidence motif cells with cluster assignments. Unresolved cases are shown separately; two low-evidence motif cells lack protein-cluster assignments. Complete-case counts additionally restrict to all 35 scored markers and no multi-component-DNA concern.', '',
        '```', table.round(5).to_string(index=False), '```', '',
        'Permutation summaries:', '', '```', json.dumps(meta, indent=2), '```', '',
        'The permutations preserve flag totals within patient or patient/FOV and shuffle flags among assessable motif cells. Cluster-specific permutation tests assess excess flags and use BH adjustment over 15 clusters. Global tests use the Pearson contingency statistic. These are exploratory exchangeability tests: shared neighbors and spatially correlated cells can violate exchangeability even within FOV. Fisher tests also assume independent cells. Neither is a biological-replicate validation.', '',
        'Protein-expression clusters and spillover flags both use marker signal. An association, particularly with a marker-high cluster, is not independent proof of contamination. The controls within each cluster help distinguish cluster composition from motif-specific effects. Clustering can itself be influenced by spillover.', '',
        'Reproduce: `MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=1 python analysis/marker_ownership/spillover_cluster_association.py`.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(table.round(5).to_string(index=False)); print(json.dumps(meta, indent=2))
    print('Markers in cluster 8:',markers[markers.cluster_15.eq(8)].marker.value_counts().to_dict())


if __name__ == '__main__': main()
