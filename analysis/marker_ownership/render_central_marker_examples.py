"""Raw-image candidates selected explicitly for central staining, not validation."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from zipfile import ZipFile
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import find_central_marker_examples as search
from analysis.marker_ownership import figure_marker_ownership_stacked as stacked

base=search.base
OUT=search.OUT


def square_bounds(mask,ids,padding):
    """Shift a square crop inward at FOV edges rather than truncate its width."""
    yy,xx=np.where(np.isin(mask,ids))
    if not len(yy):raise ValueError('Missing requested masks')
    side=min(int(max(yy.max()-yy.min(),xx.max()-xx.min())+2*padding+1),*mask.shape)
    y0=int(np.floor((yy.min()+yy.max())/2-(side-1)/2))
    x0=int(np.floor((xx.min()+xx.max())/2-(side-1)/2))
    y0=max(0,min(y0,mask.shape[0]-side));x0=max(0,min(x0,mask.shape[1]-side))
    return slice(y0,y0+side),slice(x0,x0+side)


def load_cases(selected,square_crops=False):
    cases=[]
    for row in selected.to_dict('records'):
        config=next(c for c in base.study.CONFIG if c[0]==row['motif'])
        hits=pd.read_csv(base.study.HITS/f'{config[1]}_hits_wide.csv')
        hits=hits[hits.FOV.eq(row['fov'])&hits[config[2]].eq(row['cell_id'])].sort_values(['cell1','cell2','cell3','cell4'])
        instance=hits.iloc[0]
        columns={'I':['cell4'],'II':['cell3'],'III':['cell4'],'IV':['cell2','cell3']}[row['motif']]
        partners=[int(instance[c]) for c in columns]
        markers=base.study.CORE[row['candidate_kind']]+['dsDNA']
        with ZipFile(base.study.DATA/'raw_data'/f'{row["fov"]}.zip') as archive:
            mask=base.imaging.read_image(archive,'TIFs/segmentation_labels.tiff')
            images={m:base.imaging.read_image(archive,f'TIFs/{m}.tif') for m in markers}
        limits={m:max(float(np.quantile(im[im>0],.995)),1) if (im>0).any() else 1 for m,im in images.items()}
        row['selected_instance_json']=json.dumps({f'cell{i}':int(instance[f'cell{i}']) for i in range(1,5)})
        case=base.make_case(mask,images,limits,row,partners)
        if square_crops:
            for region,ids,padding in [('zoom',[int(row['cell_id'])],11),('context',[int(row['cell_id'])]+partners,20)]:
                sl=square_bounds(mask,ids,padding)
                case[region]=dict(mask=mask[sl].copy(),images={m:im[sl].copy() for m,im in images.items()})
        cases.append(case)
    return cases


def focused_figure(cases):
    fig=plt.figure(figsize=(12,10))
    fig.text(.025,.975,'Interior marker-signal candidates',fontsize=17,weight='bold')
    for i,case in enumerate(cases):
        top=.92-i*.28;row=case['row'];roman=row['motif']
        fig.text(.025,top,f'Motif {roman}',fontsize=13,weight='bold')
        art=fig.add_axes([.015,top-.215,.205,.20])
        with Image.open(base.OUT/'graphics'/f'motif_{roman}.png') as im:art.imshow(np.array(im))
        art.axis('off')
        grid=fig.add_gridspec(1,4,left=.24,right=.985,bottom=top-.23,top=top-.008,wspace=.025)
        for j,column in enumerate([0,1,2,4]):
            ax=fig.add_subplot(grid[0,j]);title=base.draw(ax,case,column)
            if i==0:ax.set_title('MECA-79' if title=='MECA-79 / PNAd' else title,fontsize=11,pad=3)
            elif j in [1,2]:ax.set_title('HLA-II' if column==2 else 'CD11c',fontsize=11,pad=3)
        fig.text(.24,top-.247,f'P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]}',fontsize=10)
    fig.text(.025,.060,'Selection: at least 3 above-background pixels after 2-pixel erosion in each core-marker channel; not a validated positivity call.',fontsize=9)
    fig.text(.025,.041,'Dashed white: unidentified mask. Cyan: annotated motif neighbor. Bars: merge 20 px; individual channels 10 px.',fontsize=9)
    fig.text(.025,.022,'These deliberately selected examples include weak and peripheral signal and do not establish absence of spillover or HEV/DC identity.',fontsize=9)
    return fig


def detail(case,measurements):
    fig,axes=plt.subplots(1,4,figsize=(12,4.4));row=case['row']
    for j,column in enumerate([0,1,2,4]):
        title=base.draw(axes[j],case,column);axes[j].set_title(title.replace(' / PNAd',''),fontsize=11)
        if column in [1,2]:
            own=case['zoom']['mask']==case['cell_id'];inner=search.ndi.binary_erosion(own,structure=np.ones((3,3),bool),iterations=2)
            if inner.any():axes[j].contour(inner.astype(float),levels=[.5],colors=['#ffdf60'],linewidths=.6,linestyles=':')
    fig.suptitle(f'Motif {row["motif"]} · P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]}',fontsize=14,y=.97)
    m=measurements[(measurements['patient number']==row['patient number'])&measurements.fov.eq(row['fov'])&measurements.cell_id.eq(case['cell_id'])]
    lines=[]
    for r in m.itertuples():
        radius=f'{r.relative_signal_radius:.2f}' if np.isfinite(r.relative_signal_radius) else 'undefined (no excess signal)'
        lines.append(f'{r.marker}: {r.interior_positive_pixels} interior pixels above background ({r.background_threshold:g}); relative signal radius {radius}')
    fig.text(.02,.16,'\n'.join(lines),fontsize=9,linespacing=1.6)
    fig.text(.02,.055,'Yellow dotted: 2-pixel-eroded interior. White dashed: focal mask. Cyan: motif neighbor. Raw display limits fixed per FOV/channel.',fontsize=8)
    fig.text(.02,.025,'Central staining was a selection criterion; these images do not independently validate annotation or exclude spillover.',fontsize=8)
    fig.subplots_adjust(left=.015,right=.99,top=.82,bottom=.27,wspace=.04)
    return fig


def main():
    rank=pd.read_csv(OUT/'all_candidate_cell_ranking.csv')
    pool=rank[rank.display_geometry_eligible&~rank.any_core_risk_flag].copy()
    selected=pool.groupby('motif',sort=False).head(2).copy()
    assert len(selected)==8
    selected['example_id']=selected.motif+'_'+(selected.groupby('motif').cumcount()+1).astype(str)
    cases=load_cases(selected)
    pd.DataFrame([c['row'] for c in cases]).to_csv(OUT/'selected_examples.csv',index=False)
    fig=stacked.stacked_figure(cases)
    fig.text(.25,.988,'Selected for interior marker signal; best available examples include incomplete marker profiles',fontsize=10)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'best_available_by_motif.{ext}',dpi=260,facecolor='white')
    plt.close(fig)
    dual=[c for c in cases if c['row']['central_markers']==2]
    fig=focused_figure(dual)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'interior_marker_candidates.{ext}',dpi=280,facecolor='white')
    plt.close(fig)
    measurements=pd.read_csv(OUT/'all_candidate_marker_centrality.csv')
    with PdfPages(OUT/'central_marker_review.pdf') as pdf:
        fig=focused_figure(dual);pdf.savefig(fig);plt.close(fig)
        for case in cases:
            fig=detail(case,measurements);pdf.savefig(fig);plt.close(fig)
    counts=rank.groupby('motif').agg(cells=('cell_id','size'),both_channels_interior_detected=('central_markers',lambda s:int(s.eq(2).sum())))
    counts.to_csv(OUT/'interior_detection_counts.csv')
    (OUT/'render_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        display='Original raw TIFFs, no denoising; original FOV/channel positive-pixel 99.5th percentile display limits.',
        candidates_with_both_marker_interior_detection=len(dual),selection='Top 2 per motif from existing fixed ranking, valid geometry/one DNA component, no prior core flag. Not a random sample.',
        motif_partners='Lexicographically first saved motif instance for this focal cell.',source_art=str(base.ART)),indent=2))
    print(counts.to_string());print('Images and nine-page review PDF complete.',flush=True)


if __name__=='__main__':main()
