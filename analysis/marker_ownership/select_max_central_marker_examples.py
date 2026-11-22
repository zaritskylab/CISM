"""Select cells using the maximum eligible single-marker centrality score."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import render_central_marker_examples as render

base=render.base
SOURCE=render.OUT
OUT=SOURCE/'max_aggregation'
KEY=base.study.KEY


def rank_markers(markers,geometry):
    markers=markers.copy()
    markers['marker_eligible']=markers.interior_positive_pixels.ge(3)&np.isfinite(markers.relative_signal_radius)
    markers['marker_centrality_score']=np.where(markers.marker_eligible,1/(1+markers.relative_signal_radius),np.nan)
    # A missing/absent other marker never reduces the winning marker's score.
    winners=markers.sort_values(KEY+['marker_centrality_score','interior_excess_fraction','interior_positive_pixels','marker'],
        ascending=[True]*len(KEY)+[False,False,False,True],na_position='last').drop_duplicates(KEY)
    cells=winners[KEY+['motif','candidate_kind','marker','marker_centrality_score','relative_signal_radius',
        'interior_excess_fraction','interior_positive_pixels']].rename(columns={
            'marker':'winning_marker','marker_centrality_score':'max_centrality_score',
            'relative_signal_radius':'winning_relative_radius','interior_excess_fraction':'winning_interior_fraction',
            'interior_positive_pixels':'winning_interior_pixels'})
    grouped=markers.groupby(KEY).agg(eligible_marker_count=('marker_eligible','sum'),any_core_risk_flag=('ownership_risk_q95',lambda s:s.eq(1).any())).reset_index()
    cells=cells.merge(grouped,on=KEY,validate='one_to_one').merge(geometry,on=KEY,validate='one_to_one')
    cells['display_eligible']=cells.geometry_valid&cells.bright_dna_components.eq(1)&~cells.any_core_risk_flag&cells.max_centrality_score.notna()
    cells=cells.sort_values(['motif','display_eligible','max_centrality_score','winning_interior_fraction','winning_interior_pixels']+KEY,
        ascending=[True,False,False,False,False]+[True]*len(KEY),na_position='last')
    cells['motif_rank']=cells.groupby('motif').cumcount()+1
    return markers,cells


def figure(cases,two_per_row=False):
    fig=plt.figure(figsize=(24,12.5) if two_per_row else (12,22))
    for k,roman in enumerate(['I','II','III','IV']):
        pair=sorted([c for c in cases if c['row']['motif']==roman],key=lambda c:c['row']['example_id'])
        left=.01+(k%2)*.50 if two_per_row else .02
        top=.94-(k//2)*.455 if two_per_row else .955-k*.232
        height=.37 if two_per_row else .202
        art_width=.105 if two_per_row else .205
        art_height=.27 if two_per_row else .147
        xgrid=left+(.12 if two_per_row else .23)
        right=left+(.485 if two_per_row else .97)
        fig.text(left+.005,top+(.018 if two_per_row else .014),f'{"ABCD"[k]}  Motif {roman}',fontsize=17,weight='bold')
        art=fig.add_axes([left,top-height*.76,art_width,art_height])
        with Image.open(base.OUT/'graphics'/f'motif_{roman}.png') as im:art.imshow(np.array(im))
        art.axis('off')
        fig.text(left+.005,top-height*.86,'HEV-associated motif' if k<2 else 'DC-associated motif',fontsize=10,
            color='#21853c' if k<2 else '#be3333')
        fig.text(left+.005,top-height*.94,'Merge: MECA-79 + DNA' if k<2 else 'Merge: CD11c + DNA',fontsize=8.5)
        bottom=top-height;gridtop=top
        if len(pair)==1:
            bottom=top-height*.75;gridtop=top-height*.25
        grid=fig.add_gridspec(len(pair),4,left=xgrid,right=right,bottom=bottom,top=gridtop,wspace=.025,hspace=.14)
        for i,case in enumerate(pair):
            first=None
            for j,column in enumerate([0,1,2,4]):
                ax=fig.add_subplot(grid[i,j]);title=base.draw(ax,case,column)
                if first is None:first=ax
                if i==0:ax.set_title(title.replace(' / PNAd',''),fontsize=13,pad=4)
            pos=first.get_position();row=case['row']
            label=f'P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]} | selected by {row["winning_marker"].replace("HLA-DR-DP-DQ","HLA-II")}'
            fig.text(pos.x0,pos.y0-(.014 if two_per_row else .008),label,fontsize=8.5)
    fig.text(.025,.031,'Selection uses the maximum centrality score across the two core markers; one eligible marker is sufficient. Display limits are unchanged.',fontsize=10)
    fig.text(.025,.018,'Dashed white: unidentified cell. Cyan: annotated motif neighbor. Bars: merges 20 px; individual channels 10 px. Selection does not validate absence of spillover.',fontsize=9)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_csv(SOURCE/'all_candidate_marker_centrality.csv')
    geometry=pd.read_csv(base.OUT/'cell_geometry.csv.gz',usecols=KEY+['geometry_valid','bright_dna_components'])
    markers,rank=rank_markers(raw,geometry)
    markers.to_csv(OUT/'per_marker_scores.csv',index=False);rank.to_csv(OUT/'all_cell_ranking.csv',index=False)
    selected=rank[rank.display_eligible].groupby('motif',sort=False).head(2).copy()
    selected['example_id']=selected.motif+'_'+(selected.groupby('motif').cumcount()+1).astype(str)
    assert set(selected.motif)=={'I','II','III','IV'}
    cases=render.load_cases(selected)
    pd.DataFrame([c['row'] for c in cases]).to_csv(OUT/'selected_examples.csv',index=False)
    for two_per_row,stem in [(False,'max_centrality_examples'),(True,'max_centrality_examples_two_per_row')]:
        fig=figure(cases,two_per_row)
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=260,facecolor='white')
        plt.close(fig)
    with PdfPages(OUT/'max_centrality_review.pdf') as pdf:
        fig=figure(cases,True);pdf.savefig(fig);plt.close(fig)
        for case in cases:
            fig=render.detail(case,raw);row=case['row']
            fig.text(.02,.205,f'Selected by {row["winning_marker"]}; max centrality score {row["max_centrality_score"]:.3f}. The other marker does not reduce the score.',fontsize=9)
            pdf.savefig(fig);plt.close(fig)
    (OUT/'provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_measurement_sha256=hashlib.sha256((SOURCE/'all_candidate_marker_centrality.csv').read_bytes()).hexdigest(),
        score='Marker score = 1/(1 + signal-weighted mean radius / uniform mask RMS radius); cell score = max of eligible core-marker scores.',
        marker_eligibility='At least 3 above-background pixels in the two-pixel-eroded mask and finite relative radius.',
        cell_eligibility='Existing valid geometry and one bright DNA component; no existing core-marker q95 risk flag. Absence of a flag is not validation.',
        tie_break='Winning-marker interior signal fraction, then interior pixel count, then patient/FOV/cell ID. Number of eligible markers is not a rank criterion.',
        maximum_per_motif=2,selected_counts=selected.groupby('motif').size().to_dict(),
        original_cohort_outcomes='Unchanged; selection for illustration only.'),indent=2))
    print(selected[['motif','fov','cell_id','winning_marker','max_centrality_score','winning_relative_radius','winning_interior_fraction']].to_string(index=False),flush=True)


if __name__=='__main__':main()
