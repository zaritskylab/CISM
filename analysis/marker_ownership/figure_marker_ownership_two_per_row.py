"""Alternative layout: two motif blocks per row, each with a 2 x 4 matrix."""

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
from PIL import Image
import matplotlib.pyplot as plt

from analysis.marker_ownership import figure_marker_ownership_stacked as source


def main():
    base=source.base
    cases=source.load_examples()
    fig=plt.figure(figsize=(24,12.5))
    for k,roman in enumerate(['I','II','III','IV']):
        left=.01+(k%2)*.50
        top=.94-(k//2)*.455
        pair=sorted([c for c in cases if c['row']['motif']==roman],key=lambda c:c['row']['example_id'])
        fig.text(left+.005,top+.018,f'{"ABCD"[k]}  Motif {roman}',fontsize=17,weight='bold')
        art=fig.add_axes([left,top-.28,.105,.27])
        with Image.open(base.OUT/'graphics'/f'motif_{roman}.png') as image: art.imshow(np.array(image))
        art.axis('off')
        fig.text(left+.005,top-.317,'HEV-associated motif' if k<2 else 'DC-associated motif',fontsize=10,
                 color='#21853c' if k<2 else '#be3333')
        fig.text(left+.005,top-.346,'Merge: MECA-79 + DNA' if k<2 else 'Merge: CD11c + DNA',fontsize=8.5)
        grid=fig.add_gridspec(2,4,left=left+.12,right=left+.485,bottom=top-.37,top=top,wspace=.025,hspace=.12)
        for i,case in enumerate(pair):
            first=None
            for j,column in enumerate([0,1,2,4]):
                ax=fig.add_subplot(grid[i,j]);title=base.draw(ax,case,column)
                if first is None:first=ax
                if i==0:ax.set_title('MECA-79' if title=='MECA-79 / PNAd' else title,fontsize=13,pad=4)
            pos=first.get_position();row=case['row']
            fig.text(pos.x0,pos.y0-.014,f'P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]}',fontsize=9)
    fig.text(.015,.035,'Dashed white: unidentified cell. Cyan: annotated motif neighbor. Green: HEV markers; red: DC markers; blue: DNA in merges.',fontsize=11)
    fig.text(.015,.017,'Bars: merges 20 px; individual channels 10 px. Images are illustrative; proposed cell identities remain unvalidated.',fontsize=11)
    stem='motif_ownership_figure_two_per_row'
    for ext in ['png','pdf','svg']:fig.savefig(base.OUT/f'{stem}.{ext}',dpi=260,facecolor='white')
    plt.close(fig)
    (base.OUT/f'{stem}_provenance.json').write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layout='Two motifs per row, two rows; each motif left of a 2 x 4 matrix',
        examples_unchanged=True,display_limits_unchanged=True,status_text_removed=True),indent=2))
    print(base.OUT/f'{stem}.pdf',flush=True)


if __name__=='__main__':main()
