"""Compare min-max cluster 4 with annotated HEV cells, NN/NP only."""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.marker_ownership.unidentified_neighbor_enrichment import FEATURES, KEY

ROOT = ANALYSIS_DIR
SOURCE = Path('/groups/assafza_group/assafza/Datasets/Melanoma')
RUN = ROOT/'outputs/unidentified_neighbor_enrichment_minmax'
OUT = RUN/'cluster4_vs_hev'
OUT.mkdir(parents=True,exist_ok=True)
assignments = pd.read_csv(RUN/'cell_cluster_assignments.csv')
cluster = assignments.loc[assignments.cluster_15.eq(4),KEY].copy()
cluster['population'] = 'Cluster 4'
cells = pd.read_csv(SOURCE/'mapped_cell_types.csv')
cells = cells.loc[cells.Group.isin(['NN','NP'])].copy()
hev = cells.loc[cells.pred_mapped.eq('hevs'),KEY].copy()
hev['population'] = 'HEV'
selected = pd.concat([cluster,hev],ignore_index=True).merge(cells[KEY+['centroid-0','centroid-1']],on=KEY,validate='one_to_one')
join = ['patient number','fov','centroid-0','centroid-1']
selected[['centroid-0','centroid-1']] = np.trunc(selected[['centroid-0','centroid-1']])
protein = pd.read_csv(SOURCE/'cleaned_expression_with_both_classification_prob_spatial_27_09_23.csv',usecols=join+FEATURES)
selected = selected.merge(protein,on=join,how='left',validate='many_to_one',indicator=True)
missing = selected.loc[selected._merge.ne('both'),KEY+['population']]
missing.to_csv(OUT/'unmatched_protein_cells.csv',index=False)
selected = selected.loc[selected._merge.eq('both')].copy()
assert np.isfinite(selected[FEATURES].to_numpy()).all()
params = pd.read_csv(RUN/'scaler_parameters.csv').set_index('feature').loc[FEATURES]
scaled = selected[FEATURES].mul(params.multiplier,axis=1).add(params.offset,axis=1)
scaled['population'] = selected.population.to_numpy()
means = scaled.groupby('population')[FEATURES].mean().reindex(['Cluster 4','HEV'])
raw_means = selected.groupby('population')[FEATURES].mean().reindex(means.index)
means.rename_axis('population').to_csv(OUT/'mean_protein_profiles_minmax.csv')
raw_means.to_csv(OUT/'mean_protein_profiles_raw.csv')
sizes = selected.population.value_counts()
assert sizes['Cluster 4']==len(cluster)
centroids = pd.read_csv(RUN/'cluster_centroids_minmax.csv',index_col=0)
# KMeans may stop at tolerance before centers equal final-label means exactly.
centroid_difference = float(np.max(np.abs(means.loc['Cluster 4'] - centroids.loc[4,FEATURES])))
np.testing.assert_allclose(means.loc['Cluster 4'],centroids.loc[4,FEATURES],rtol=0,atol=1e-3)
colors = ['#a23b72','#087e8b']
labels = [f'{name} (n={sizes[name]:,})' for name in means.index]
# A paired dot plot: dot position is mean expression on the shared scale.
fig,ax=plt.subplots(figsize=(11,11))
y=np.arange(len(FEATURES))
ax.hlines(y,means.iloc[0],means.iloc[1],color='#c5cbd0',lw=1.7,zorder=1)
for i,(name,row) in enumerate(means.iterrows()):
    ax.scatter(row,y,s=48,color=colors[i],label=labels[i],zorder=3)
ax.set_yticks(y,FEATURES);ax.invert_yaxis()
ax.set_xlabel('Mean protein expression (same min–max transform used for clustering)')
ax.set_title('Unidentified cluster 4 versus annotated HEV cells\nNN/NP patients · all 36 protein channels',pad=15)
ax.grid(axis='x',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.07),ncol=2);ax.set_xlim(left=0)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'cluster4_vs_hev_dotplot.png',dpi=180);fig.savefig(OUT/'cluster4_vs_hev_dotplot.pdf');plt.close(fig)
# All 36 channels in original feature order, without marker selection.
a=np.linspace(0,2*np.pi,len(FEATURES),endpoint=False)
fig,ax=plt.subplots(figsize=(13,13),subplot_kw={'projection':'polar'})
ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
for i,row in enumerate(means.to_numpy()):
    ax.plot(np.r_[a,a[0]],np.r_[row,row[0]],color=colors[i],lw=2,label=labels[i])
    ax.fill(np.r_[a,a[0]],np.r_[row,row[0]],color=colors[i],alpha=.09)
ax.set_xticks(a,FEATURES,fontsize=8);ax.tick_params(axis='x',pad=13)
ax.set_ylim(0,float(means.to_numpy().max())*1.13)
ax.set_title('Unidentified cluster 4 versus annotated HEV cells\nMean expression on the clustering min–max scale · NN/NP',pad=40)
ax.legend(loc='upper right',bbox_to_anchor=(1.25,1.12))
fig.tight_layout();fig.savefig(OUT/'cluster4_vs_hev_radar.png',dpi=180);fig.savefig(OUT/'cluster4_vs_hev_radar.pdf');plt.close(fig)
metadata={'cluster4_max_centroid_difference':centroid_difference,'expression_source':str(SOURCE/'cleaned_expression_with_both_classification_prob_spatial_27_09_23.csv'),'labels_source':str(SOURCE/'mapped_cell_types.csv'),'cluster_run':str(RUN),'matched_counts':sizes.to_dict(),'unmatched_counts':missing.population.value_counts().to_dict(),'scaling':'Stored min-max transform fitted on matched NN/NP unidentified cells; applied unchanged to HEV; no clipping.','aggregation':'Cell-weighted mean across NN/NP, not patient-weighted.'}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
(OUT/'README.md').write_text('''# Cluster 4 versus HEV protein profiles

Cluster 4 refers to the min–max run. HEV means pred_mapped == "hevs", restricted to NN/NP. Both profiles are cell-weighted means of the same 36 protein channels. The exact stored per-channel min–max transform fitted on unidentified cells is applied to both populations; HEV values are not separately rescaled or clipped and individual HEV values may exceed [0,1]. Coordinates are truncated for the protein join, matching the clustering pipeline. Missing HEV matches are listed separately. The cluster 4 profile uses the mean over its saved final cell assignments; it agrees with the saved KMeans centroid within 0.001 (small differences reflect KMeans stopping tolerance).

The paired dot plot uses dot position for mean expression (dot size has no additional meaning). The radar uses all 36 channels in feature order. Neither figure indicates a positivity threshold or statistical significance. Raw and transformed means, sample counts and source paths are saved alongside the plots. PDF versions are available for export.
''')
print(json.dumps(metadata,indent=2))
print(means.T.sort_values('Cluster 4',ascending=False).head(10).to_string())
