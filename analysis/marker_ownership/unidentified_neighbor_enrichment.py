"""NN/NP unidentified-cell clusters and spatial neighbor enrichment.

Run with .venv-noise/bin/python analysis/marker_ownership/unidentified_neighbor_enrichment.py.
Neighbors match the tutorial: Delaunay edges of length <=100, within patient/FOV.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CELL_TYPES = ['DC', 'b cell', 'blood vessels', 'cd4 apc', 'cd4 t cell', 'cd8 t cell', 'gc', 'hevs', 'mac', 'memory%cd4 t cell', 'neutrophil', 'nk cell', 'stroma', 'treg', 'tumor', 'unidentified']
KEY = ['patient number', 'fov', 'cell_id']
FEATURES = ['CD11c', 'CD14', 'CD163', 'CD20', 'CD206', 'CD21', 'CD3', 'CD31', 'CD4', 'CD45', 'CD45RA', 'CD45RO', 'CD56', 'CD68', 'CD69', 'CD8', 'COL1A1', 'DC-SIGN', 'Foxp3', 'Granzyme B', 'HLA-DR-DP-DQ', 'HLA-class-1-A-B-C', 'IDO-1', 'Ki67', 'LAG-3', 'MECA-79', 'MelanA', 'PD-1', 'S100A9-Calprotectin', 'SMA', 'SOX10', 'TCF1TCF7', 'TIM-3', 'Tox-Tox2', 'anti-Biotin', 'dsDNA']


def neighbor_matrices(cells, cutoff=100):
    """Count directed focal-cell/neighbor incidences; exclude self edges.

    Expected counts use each focal cell's degree times the fraction of other
    cells of each type in its FOV. Shared neighbors count once per focal cell.
    """
    types = CELL_TYPES
    observed = np.zeros((15, len(types)), dtype=np.int64)
    expected = np.zeros_like(observed, dtype=float)
    support = []
    for (patient, fov), frame in cells.groupby(['patient number', 'fov'], sort=True):
        xy = frame[['centroid-0', 'centroid-1']].to_numpy(float)
        codes = pd.Categorical(frame.pred_mapped, categories=types).codes
        clusters = frame.cluster_15.fillna(-1).to_numpy(int)
        n = len(frame)
        if n < 3:
            pairs = np.array([(i,j) for i in range(n) for j in range(i+1,n)], dtype=int).reshape(-1,2)
        else:
            tri = Delaunay(xy)  # Fail explicitly on degenerate geometry.
            pairs = np.unique(np.sort(np.concatenate([tri.simplices[:,[0,1]], tri.simplices[:,[1,2]], tri.simplices[:,[0,2]]]), axis=1), axis=0)
        pairs = pairs[np.linalg.norm(xy[pairs[:,0]] - xy[pairs[:,1]], axis=1) <= cutoff]
        edges = np.concatenate([pairs, pairs[:,::-1]])
        degree = np.bincount(edges[:,0], minlength=n)
        focal = np.flatnonzero(clusters >= 0)
        totals = np.bincount(codes, minlength=len(types))
        for i in focal:
            c = clusters[i]
            available = totals.copy(); available[codes[i]] -= 1
            if n > 1:
                expected[c] += degree[i] * available / (n-1)
            support.append((patient, fov, frame.iloc[i].cell_id, c, degree[i]))
        selected = clusters[edges[:,0]] >= 0
        np.add.at(observed, (clusters[edges[selected,0]], codes[edges[selected,1]]), 1)
    index = pd.Index(range(15), name='cluster_15')
    return (pd.DataFrame(observed,index=index,columns=types),
            pd.DataFrame(expected,index=index,columns=types),
            pd.DataFrame(support,columns=KEY+['cluster_15','neighbor_count']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/groups/assafza_group/assafza/Datasets/Melanoma'))
    parser.add_argument('--output', type=Path, default=ANALYSIS_DIR/'outputs/unidentified_neighbor_enrichment')
    parser.add_argument('--max-distance', type=float, default=100)
    parser.add_argument('--scaling', choices=['standard', 'minmax'], default='standard')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.data_dir/'mapped_cell_types.csv')
    cells = cells[cells.Group.isin(['NN','NP'])].copy()
    if cells.duplicated(KEY).any():
        raise ValueError('Duplicate spatial cell keys')
    if not cells.pred_mapped.isin(CELL_TYPES).all():
        raise ValueError('Unknown mapped cell types')
    protein = pd.read_csv(args.data_dir/'cleaned_expression_with_both_classification_prob_spatial_27_09_23.csv', usecols=['patient number','fov','centroid-0','centroid-1']+FEATURES)
    focal = cells[cells.pred_mapped.eq('unidentified')].copy()
    # Reproduce the old notebook coordinate join, without motif-instance duplication.
    join = ['patient number','fov','centroid-0','centroid-1']
    focal[['centroid-0','centroid-1']] = np.trunc(focal[['centroid-0','centroid-1']])
    focal = focal.merge(protein, on=join, how='left', validate='many_to_one', indicator=True)
    missing = focal.loc[~focal._merge.eq('both'), KEY].copy()
    missing.to_csv(args.output/'unmatched_protein_cells.csv', index=False)
    focal = focal.loc[focal._merge.eq('both')].copy()
    if not np.isfinite(focal[FEATURES].to_numpy()).all():
        raise ValueError('Nonfinite protein values')
    scaler = MinMaxScaler() if args.scaling == 'minmax' else StandardScaler()
    model = KMeans(n_clusters=15, random_state=42069, n_init=10)
    focal['cluster_15'] = model.fit_predict(scaler.fit_transform(focal[FEATURES]))
    focal[KEY+['Group','cluster_15']].to_csv(args.output/'cell_cluster_assignments.csv', index=False)
    pd.DataFrame(model.cluster_centers_,columns=FEATURES).rename_axis('cluster_15').to_csv(args.output/('cluster_centroids_minmax.csv' if args.scaling == 'minmax' else 'cluster_centroids_standardized.csv'))
    cells = cells.merge(focal[KEY+['cluster_15']], on=KEY, how='left', validate='one_to_one')
    pd.DataFrame({'feature': FEATURES, 'multiplier': scaler.scale_ if args.scaling == 'minmax' else 1 / scaler.scale_, 'offset': scaler.min_ if args.scaling == 'minmax' else -scaler.mean_ / scaler.scale_}).to_csv(args.output/'scaler_parameters.csv', index=False)
    metadata = {'scaling':args.scaling, 'scaling_fit_population':'matched NN/NP unidentified cells', 'groups':['NN','NP'], 'seed':42069,'n_init':10,'features':FEATURES,'cells':len(cells),'unidentified_cells':len(focal),'unmatched_unidentified_cells':len(missing),'max_distance':args.max_distance,'patients':cells['patient number'].nunique(),'fovs':cells.groupby(['patient number','fov']).ngroups}
    for name, subset in [('pooled',cells),('NN',cells[cells.Group.eq('NN')]),('NP',cells[cells.Group.eq('NP')])]:
        obs, exp, support = neighbor_matrices(subset,args.max_distance)
        # Keep all 16 columns even if a group lacks a type.
        cols = CELL_TYPES
        obs, exp = obs.reindex(columns=cols,fill_value=0), exp.reindex(columns=cols,fill_value=0)
        prop = obs.div(obs.sum(axis=1).replace(0,np.nan),axis=0)
        ratio = obs / exp.replace(0,np.nan)
        for label, matrix in [('counts',obs),('expected',exp),('proportions',prop),('enrichment',ratio)]:
            matrix.to_csv(args.output/f'{name}_{label}.csv')
        support.to_csv(args.output/f'{name}_focal_support.csv',index=False)
        fig, axes = plt.subplots(1,2,figsize=(17,7))
        for ax, matrix, title, cmap, vmax in [(axes[0],prop,'Neighbor proportions','viridis',1),(axes[1],np.log2(ratio.where(ratio>0)),'log2(observed / FOV-adjusted expected)','RdBu_r',3)]:
            im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=(-3 if cmap=='RdBu_r' else 0),vmax=vmax)
            ax.set_xticks(range(16),cols,rotation=65,ha='right'); ax.set_yticks(range(15)); ax.set_ylabel('Cluster'); ax.set_title(title)
            fig.colorbar(im,ax=ax)
        fig.suptitle(f'{name}: unidentified-cell neighborhoods (blank = undefined; zero enrichment also blank in log plot)')
        fig.tight_layout(); fig.savefig(args.output/f'{name}_heatmaps.png',dpi=180); plt.close(fig)
    (args.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__':
    main()
