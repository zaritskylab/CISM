"""Four motif rows, original artwork left and two-by-four raw image panels right."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import figure_marker_ownership as base


def load_examples():
    requests=pd.read_csv(base.OUT/'render_manifest.csv')
    requests=requests[requests.main_example.fillna(False)].copy()
    selected=pd.read_csv(base.PREVIOUS)
    cases=[]
    for row in requests.to_dict('records'):
        known=selected[selected.fov.eq(row['fov'])&selected.cell_id.eq(row['cell_id'])&selected.roman.eq(row['motif'])].iloc[0]
        instance=json.loads(known.selected_instance_json)
        partner_columns={'I':['cell4'],'II':['cell3'],'III':['cell4'],'IV':['cell2','cell3']}[row['motif']]
        partners=[int(instance[c]) for c in partner_columns]
        markers=base.study.CORE[row['candidate_kind']]+['dsDNA']
        with ZipFile(base.study.DATA/'raw_data'/f'{row["fov"]}.zip') as archive:
            mask=base.imaging.read_image(archive,'TIFs/segmentation_labels.tiff')
            images={m:base.imaging.read_image(archive,f'TIFs/{m}.tif') for m in markers}
        limits={m:max(float(np.quantile(im[im>0],.995)),1) if (im>0).any() else 1 for m,im in images.items()}
        cases.append(base.make_case(mask,images,limits,row,partners))
    assert len(cases)==8
    return cases


def stacked_figure(cases):
    fig=plt.figure(figsize=(12,22))
    for k,roman in enumerate(['I','II','III','IV']):
        top=.965-k*.235
        pair=sorted([c for c in cases if c['row']['motif']==roman],key=lambda c:c['row']['example_id'])
        fig.text(.025,top+.010,f'{"ABCD"[k]}  Motif {roman}',fontsize=17,weight='bold')
        art=fig.add_axes([.018,top-.152,.215,.147])
        with Image.open(base.OUT/'graphics'/f'motif_{roman}.png') as image: art.imshow(np.array(image))
        art.axis('off')
        fig.text(.025,top-.173,'HEV-associated motif' if k<2 else 'DC-associated motif',fontsize=11,
                 color='#21853c' if k<2 else '#be3333')
        fig.text(.025,top-.191,'Merge: MECA-79 + DNA' if k<2 else 'Merge: CD11c + DNA',fontsize=9)
        grid=fig.add_gridspec(2,4,left=.25,right=.99,bottom=top-.202,top=top,wspace=.025,hspace=.12)
        for i,case in enumerate(pair):
            first=None
            for j,column in enumerate([0,1,2,4]):
                ax=fig.add_subplot(grid[i,j]); title=base.draw(ax,case,column)
                if first is None:first=ax
                if i==0:ax.set_title('MECA-79' if title=='MECA-79 / PNAd' else title,fontsize=13,pad=4)
            pos=first.get_position(); row=case['row']
            fig.text(pos.x0,pos.y0-.008,f'P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]}',fontsize=9)
    fig.text(.025,.025,'Dashed white: unidentified cell. Cyan: annotated motif neighbor. Green: HEV markers; red: DC markers; blue: DNA in merges.',fontsize=10)
    fig.text(.025,.014,'Bars: merges 20 px; individual channels 10 px. Images are illustrative; proposed cell identities remain unvalidated.',fontsize=10)
    return fig


def main():
    archive=base.OUT/'previous_layout';archive.mkdir(exist_ok=True)
    names=['motif_ownership_figure.png','motif_ownership_figure.pdf','motif_ownership_figure.svg','results_page_1.png']
    for name in names:
        if not (archive/name).exists():shutil.copy2(base.OUT/name,archive/name)
    cases=load_examples();fig=stacked_figure(cases)
    for ext in ['png','pdf','svg']:fig.savefig(base.OUT/f'motif_ownership_figure.{ext}',dpi=300,facecolor='white')
    fig.savefig(base.OUT/'results_page_1.png',dpi=220,facecolor='white');plt.close(fig)
    with PdfPages(base.OUT/'marker_ownership_results.pdf') as pdf:
        for number in [1,2,3]:
            with Image.open(base.OUT/f'results_page_{number}.png') as im:pixels=np.array(im)
            fig=plt.figure(figsize=(10,10*pixels.shape[0]/pixels.shape[1]));ax=fig.add_axes([0,0,1,1]);ax.imshow(pixels);ax.axis('off');pdf.savefig(fig);plt.close(fig)
    (base.OUT/'stacked_layout_provenance.json').write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),layout='Four rows; supplied motif left; two by four microscopy matrix right',
        examples_unchanged=True,display_limits_unchanged=True,status_text_removed=True,
        displayed_channels={'HEV':['merge','CD31','MECA-79','dsDNA'],'DC':['merge','CD11c','HLA-DR-DP-DQ','dsDNA']},
        omitted_from_compact_figure='CD45/CD3 alternative-lineage channels remain in the detailed image packets; numerical analysis unchanged.'),indent=2))
    print('Updated main figure and three-page results PDF',flush=True)


if __name__=='__main__':main()
