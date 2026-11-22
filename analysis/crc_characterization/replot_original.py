"""Reproduce the historical 19-motif weighted pairwise figure and restyle ticks.

Run with .venv-noise/bin/python analysis/crc_characterization/replot_original.py
Uses the historical notebook's stored selected IDs and its original helper function.
Does not rerun or change motif selection, edge orientation, weighting or normalization.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

import ast
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR','/tmp/cism_original_pairwise_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

OUT=ANALYSIS_DIR/'original_motif_induced_pairwise'
NOTEBOOK=Path('/home/omriavit/CISM/CISM/analysis/Melanoma_MIBITOF/Melanoma_100_distance_NN_NP_depth_4_nodes.ipynb')
HELPERS=Path('/home/omriavit/CISM/CISM/cism/helpers.py')
DATA=Path('/groups/assafza_group/assafza/Omri/processed_data_23_10_2025')

def notebook_ids(cells,index):
    text=''.join(''.join(o.get('text',[])) for o in cells[index].get('outputs',[]))
    return [int(i) for i in re.findall(r'motif_id: (\d+)',text)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notebook=json.loads(NOTEBOOK.read_text());cells=notebook['cells']
    codebook_node=ast.parse(''.join(cells[2]['source'])).body[0]
    codebook=ast.literal_eval(codebook_node.value)
    groups={g:notebook_ids(cells,i) for g,i in [('NN',37),('NP',38)]}
    assert [len(groups[g]) for g in ['NN','NP']]==[8,11]
    selected=set(groups['NN']+groups['NP']);assert len(selected)==19
    labels=pd.read_csv(DATA/'patient_class.csv',header=None,names=['uid','group'])
    labels['patient']=labels.uid.str.replace('Melanoma','',regex=False).astype(int)
    eligible=set(labels.loc[labels.group.isin(['NN','NP']),'patient'])
    parts=[];sources=[]
    for path in sorted((DATA/'parse_cache/melanoma_distance_100_XX/4').glob('*.df_cache')):
        pid=int(path.name.split('_')[0])
        if pid not in eligible:continue
        df=pd.read_csv(path,usecols=['Patient','FOV','ID','Count','Freq','Mean_Freq','p_value','motif'])
        # Original helper import_data filters: nonmissing Mean_Freq and p < .05;
        # notebook quantile threshold=1 does not discard any remaining row.
        df=df[df.ID.isin(selected)&df.Mean_Freq.notna()&(df.p_value<.05)].copy()
        parts.append(df);sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(parts)==76
    data=pd.concat(parts,ignore_index=True);assert set(data.ID)==selected
    data['Total']=data.Count/data.Freq
    freq=data.groupby(['Patient','ID'],observed=True)[['Count','Total']].sum().reset_index()
    freq['Freq']=freq.Count/freq.Total
    weights=freq.groupby('ID').Freq.mean()
    # Separate aggregation proves the historical Count/Freq, then equal-patient
    # frequency mean, rather than weighting by motif counts or including absent zeros.
    independent={}
    for mid in selected:
        per_patient={}
        for row in data.loc[data.ID==mid].itertuples():
            n,d=per_patient.get(row.Patient,(0.,0.))
            per_patient[row.Patient]=(n+row.Count,d+row.Count/row.Freq)
        independent[mid]=sum(n/d for n,d in per_patient.values())/len(per_patient)
    assert np.allclose([independent[i] for i in weights.index],weights,rtol=1e-14,atol=1e-16)
    weights.rename('historical_mean_patient_frequency').to_csv(OUT/'motif_weights.csv')
    freq.to_csv(OUT/'motif_patient_frequencies.csv',index=False)
    graph_lookup=data.drop_duplicates('ID').set_index('ID').motif
    identity_map={mid:[mid] for mid in groups['NN']+groups['NP']}
    discriminator=SimpleNamespace(cism=SimpleNamespace(motifs_dataset=data))
    # Execute the actual historical helper function in an isolated namespace to
    # avoid loading unrelated CISM/ML modules; function body remains unchanged.
    tree=ast.parse(HELPERS.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_weight_matrix_from_motifs')
    namespace={'np':np,'pd':pd,'nx':nx,'string_base64_pickle':lambda s:pickle.loads(base64.b64decode(s))}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(HELPERS),'exec'),namespace)
    matrices={};checks={}
    for group,ids in groups.items():
        with contextlib.redirect_stdout(io.StringIO()):
            matrix=namespace['create_weight_matrix_from_motifs'](discriminator,codebook,identity_map,ids,weights)
        separate=np.zeros((16,16))
        for mid in ids:
            graph=pickle.loads(base64.b64decode(graph_lookup.loc[mid]))
            for u,v in nx.Graph(graph).edges():
                separate[int(graph.nodes[u]['type']),int(graph.nodes[v]['type'])]+=independent[mid]
        separate/=separate.sum()
        difference=float(np.max(np.abs(matrix.to_numpy()-separate)))
        assert difference<1e-14
        # Original plotting code drops Tumor after normalization, without renormalizing.
        assert np.all(matrix.loc['Tumor']==0) and np.all(matrix['Tumor']==0)
        matrix=matrix.drop(index='Tumor',columns='Tumor');matrices[group]=matrix
        matrix.to_csv(OUT/f'original_motif_induced_pairwise_{group}.csv')
        checks[group]=dict(selected_motifs=ids,n_motifs=len(ids),max_abs_difference_independent=difference,
                           matrix_sum=float(matrix.to_numpy().sum()),n_nonzero=int(np.count_nonzero(matrix)),
                           vmin=float(matrix.to_numpy()[matrix.to_numpy()!=0].min()),vmax=float(matrix.to_numpy().max()))
    # Same strict-upper-triangle Spearman result stored in historical cell 39.
    from scipy.stats import spearmanr
    upper=np.triu_indices(15,k=1)
    result=spearmanr(matrices['NN'].to_numpy()[upper],matrices['NP'].to_numpy()[upper])
    assert np.isclose(result.statistic,-.015790637411579066,atol=1e-14)
    assert np.isclose(result.pvalue,.8729764344358347,atol=1e-14)
    checks['historical_spearman_check']=dict(rho=float(result.statistic),pvalue=float(result.pvalue),matches_notebook=True)
    # Export original embedded panels as a direct visual reference.
    for group,i in [('NN',37),('NP',38)]:
        for output in cells[i].get('outputs',[]):
            if 'image/png' in output.get('data',{}):
                (OUT/f'original_notebook_{group}.png').write_bytes(base64.b64decode(output['data']['image/png']))
    checks['archived_panel_color_checks']=validate_archived_panels(matrices)
    plot(matrices,checks)
    payload=dict(notebook=str(NOTEBOOK),notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
                 helper=str(HELPERS),helper_sha256=hashlib.sha256(HELPERS.read_bytes()).hexdigest(),
                 source_cache_files=sources,checks=checks,
                 method='Archived NN8/NP11 motif IDs; original Count/Freq patient frequencies; original mean frequencies; original oriented motif-edge accumulation and global matrix normalization.',
                 tick_changes='18pt category labels; 20pt colorbar labels rotated45 degrees; at most4 colorbar intervals; categorical labels retained to identify every cell type.')
    (OUT/'reproduction_checks.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

def validate_archived_panels(matrices):
    """Compare all 225 cell colors to each original embedded notebook figure.

    Fit the archived raster grid from unique-color cells; colors are not used
    to estimate numerical matrix values. Those come only from source data.
    """
    results={}
    for group,cmap in [('NN','Blues'),('NP','Reds')]:
        a=matrices[group].to_numpy()
        image=np.asarray(Image.open(OUT/f'original_notebook_{group}.png').convert('RGB'))
        colors=np.rint(matplotlib.colormaps[cmap](Normalize(a[a!=0].min(),a.max())(a))[...,:3]*255).astype(int)
        h,w=image.shape[:2];crop=image[:int(h*.67),:int(w*.89)];anchors=[]
        for r,c in zip(*np.nonzero(a)):
            if np.sum(np.all(colors==colors[r,c],axis=2)&(a!=0))!=1:continue
            ys,xs=np.where(np.all(crop==colors[r,c],axis=2))
            if len(xs)>100 and np.ptp(xs)<150 and np.ptp(ys)<150:
                anchors.append((r,c,float(np.median(ys)),float(np.median(xs))))
        anchors=np.asarray(anchors);assert len(anchors)>=4
        dx,x0=np.polyfit(anchors[:,1],anchors[:,3],1)
        dy,y0=np.polyfit(anchors[:,0],anchors[:,2],1)
        actual=np.array([[image[round(y0+dy*r),round(x0+dx*c)] for c in range(15)] for r in range(15)])
        expected=colors.copy();expected[a==0]=255
        assert np.array_equal(actual,expected),f'Archived {group} panel differs'
        results[group]=dict(exact_cell_color_matches=225,total_cells=225,max_rgb_difference=0)
    return results

def plot(matrices,checks):
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(24,10))
    gs=fig.add_gridspec(1,5,width_ratios=[1,.035,.23,1,.035],left=.155,right=.955,bottom=.40,top=.94,wspace=.08)
    for group,col,cmap in [('NN',0,'Blues'),('NP',3,'Reds')]:
        ax=fig.add_subplot(gs[0,col]);cax=fig.add_subplot(gs[0,col+1])
        a=matrices[group].to_numpy();masked=np.ma.masked_equal(a,0)
        im=ax.imshow(masked,cmap=cmap,norm=Normalize(checks[group]['vmin'],checks[group]['vmax']),aspect='auto',interpolation='nearest')
        ax.set_xticks(range(15),matrices[group].columns,rotation=90,fontsize=18)
        ax.set_yticks(range(15),matrices[group].index,fontsize=18)
        ax.tick_params(length=0,pad=8)
        if group=='NP':ax.set_yticklabels([])
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title(f'{group} disease state',fontsize=22,pad=15,weight='bold')
        cb=fig.colorbar(im,cax=cax)
        lo,hi=checks[group]['vmin'],checks[group]['vmax']
        ticks=MaxNLocator(nbins=4,min_n_ticks=3).tick_values(lo,hi)
        ticks=ticks[(ticks>=lo)&(ticks<=hi)]
        cb.set_ticks(ticks);cb.formatter=FormatStrFormatter('%.2f');cb.update_ticks()
        cb.ax.tick_params(labelsize=20,length=4,pad=10)
        for label in cb.ax.get_yticklabels():label.set_rotation(45);label.set_ha('left');label.set_rotation_mode('anchor')
        cb.outline.set_visible(False)
        checks[group]['new_colorbar_ticks']=ticks.tolist()
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'motif_induced_pairwise_clean.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
