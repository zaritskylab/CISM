"""One top max-centrality cell per motif, four vertically stacked image rows."""

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

from analysis.marker_ownership import select_max_central_marker_examples as selection


def main():
    base=selection.base;out=selection.OUT
    ranking=pd.read_csv(out/'all_cell_ranking.csv')
    selected=ranking[ranking.display_eligible].groupby('motif',sort=False).head(1).copy()
    selected['example_id']=selected.motif+'_1'
    assert len(selected)==4 and selected.motif.nunique()==4
    cases=selection.render.load_cases(selected,square_crops=True)
    for case in cases:
        for region in ['context','zoom']:
            assert case[region]['mask'].shape[0]==case[region]['mask'].shape[1]
    fig=plt.figure(figsize=(12,12.4))
    panel_sizes=[]
    for k,roman in enumerate(['I','II','III','IV']):
        case=next(c for c in cases if c['row']['motif']==roman);row=case['row']
        top=.945-k*.229
        fig.text(.025,top+.017,f'{"ABCD"[k]}  Motif {roman}',fontsize=16,weight='bold')
        art=fig.add_axes([.018,top-.17,.205,.15])
        with Image.open(base.OUT/'graphics'/f'motif_{roman}.png') as im:art.imshow(np.array(im))
        art.axis('off')
        fig.text(.025,top-.191,'Merge: MECA-79 + DNA' if k<2 else 'Merge: CD11c + DNA',fontsize=9)
        grid=fig.add_gridspec(1,4,left=.25,right=.99,bottom=top-.175,top=top,wspace=.025)
        first=None
        for j,column in enumerate([0,1,2,4]):
            ax=fig.add_subplot(grid[0,j]);title=base.draw(ax,case,column)
            ax.set_title(title.replace(' / PNAd',''),fontsize=13,pad=4)
            position=ax.get_position()
            panel_sizes.append((position.width*12,position.height*12.4))
            if first is None:first=ax
        pos=first.get_position()
        marker=row['winning_marker'].replace('HLA-DR-DP-DQ','HLA-II')
        fig.text(pos.x0,pos.y0-.014,f'P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]} | selected by {marker}',fontsize=9)
    fig.text(.025,.034,'Dashed white: unidentified cell. Cyan: annotated motif neighbor. Green: HEV markers; red: DC markers; blue: DNA in merges.',fontsize=9)
    fig.text(.025,.020,'Bars: merges 20 px; individual channels 10 px. Selected by maximum single-marker centrality; selection does not establish absence of spillover.',fontsize=9)
    stem='one_cell_per_motif_stacked'
    assert np.allclose(panel_sizes,panel_sizes[0]),panel_sizes
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{stem}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    pd.DataFrame([c['row'] for c in cases]).to_csv(out/f'{stem}_examples.csv',index=False)
    (out/f'{stem}_provenance.json').write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ranking_sha256=hashlib.sha256((out/'all_cell_ranking.csv').read_bytes()).hexdigest(),
        selection='Top eligible max-centrality cell per motif; original ordering and filters preserved.',
        layout='Four rows; one supplied motif graphic and merge / core marker 1 / core marker 2 / DNA per row.',
        raw_images_and_display_limits='Original raw images and original display limits. Square crops shift inward at FOV edges; no stretching or generated padding.',
        displayed_panel_sizes_inches=panel_sizes),indent=2))
    print(selected[['motif','fov','cell_id','winning_marker']].to_string(index=False),flush=True)


if __name__=='__main__':main()
