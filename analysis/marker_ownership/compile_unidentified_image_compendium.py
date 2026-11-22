"""Collect every microscopy image variant and linked supplementary image set."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
import gc
import hashlib
import json
import textwrap

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import export_marker_focused_motif_figure as focused
from analysis.marker_ownership import export_compact_motif_figure as compact

ROOT=_REPO_ROOT
OUTPUTS=ROOT/'analysis/outputs'
OUT=OUTPUTS/'unidentified_cells_image_compendium'
ORIGINAL=OUTPUTS/'unidentified_spillover_20748339660/paper_style_examples'
FOUR=OUTPUTS/'unidentified_motif_I_IV_examples'
MECA=OUTPUTS/'meca79_motif_I_II'
EXTRA=OUT/'supplementary_images'


def supplemental(cases,module,prefix):
    paths=[]
    for case in cases:
        markers=list(dict.fromkeys(['dsDNA']+case['extra']))
        rows=int(np.ceil(len(markers)/4))
        fig,axes=plt.subplots(rows,4,figsize=(8,rows*2+.8),squeeze=False)
        for ax,marker in zip(axes.flat,markers):
            module.image_panel(ax,case,marker)
            ax.set_title(module.label(marker),fontsize=9)
        for ax in list(axes.flat)[len(markers):]:ax.axis('off')
        fig.suptitle(f'{case["row"].example_id}: additional profile channels',fontsize=12)
        fig.text(.5,.015,'Dashed white: unidentified mask. Bars: 10 px. Display limits fixed per FOV/channel.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.94))
        path=EXTRA/f'{prefix}_{case["row"].example_id}.png'
        fig.savefig(path,dpi=200);plt.close(fig);paths.append(path)
    return paths


def prepare_additional():
    EXTRA.mkdir(parents=True,exist_ok=True)
    old_out=focused.OUT;old_selection=focused.selection_path
    focused.OUT=OUT/'historical_selection';focused.OUT.mkdir(exist_ok=True)
    focused.selection_path=lambda:FOUR/'example_manifest.csv'
    cases=focused.load_cases()
    for outlines,suffix in [(True,''),(False,'_no_outlines')]:
        fig=focused.compact(cases,outlines)
        fig.savefig(EXTRA/f'HEV_DC_layout_before_replacement{suffix}.png',dpi=300)
        plt.close(fig)
    del cases;gc.collect()
    focused.OUT=old_out;focused.selection_path=old_selection
    old_compact_out=compact.OUT
    compact.OUT=OUT/'historical_compact_profiles';compact.OUT.mkdir(exist_ok=True)
    cases=compact.load_cases()
    original_supp=supplemental(cases,compact,'original_compact_extra')
    del cases;gc.collect();compact.OUT=old_compact_out
    focused.OUT=OUT/'current_profiles';focused.OUT.mkdir(exist_ok=True)
    # Explicit selection source: the saved replacement manifest, with seven original
    # examples and I_1 = P14/FOV132/7775. Never resample during compilation.
    focused.selection_path=lambda:old_out/'selection_manifest.csv'
    cases=focused.load_cases()
    current_supp=supplemental(cases,focused,'current_HEV_DC_extra')
    complete=[]
    for case in cases:
        fig=plt.figure(figsize=(11,4.5))
        grid=fig.add_gridspec(2,8,left=.025,right=.985,bottom=.12,top=.83,wspace=.10,hspace=.30)
        for i,markers in enumerate([focused.HEV_MARKERS,focused.DC_MARKERS]):
            for j,marker in enumerate(markers):
                ax=fig.add_subplot(grid[i,j]);focused.image_panel(ax,case,marker)
                ax.set_title(focused.label(marker),fontsize=8)
        fig.suptitle(f'{case["row"].example_id}: complete highlighted HEV and DC marker sets',fontsize=12)
        fig.text(.5,.035,'Top: HEV-focused set, green. Bottom: DC-focused set, red. DNA is nuclear context. Bars: 10 px.',ha='center',fontsize=8)
        path=EXTRA/f'complete_marker_set_{case["row"].example_id}.png'
        fig.savefig(path,dpi=220);plt.close(fig);complete.append(path)
    del cases;gc.collect();focused.OUT=old_out;focused.selection_path=old_selection
    return original_supp,current_supp,complete


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original_supp,current_supp,complete=prepare_additional()
    entries=[];seen=set()
    def add(path,title,request,section='Figures shared in the conversation'):
        path=Path(path)
        assert path.exists(),path
        if path in seen:return
        entries.append(dict(path=str(path),title=title,request=request,section=section));seen.add(path)
    add(ORIGINAL/'unflagged_examples.png','Initial request: visually inspect unidentified cells in motif 20748339660',
        'Examples unflagged by the existing spatial screen, displayed with individual marker channels and focal segmentation.')
    add(ORIGINAL/'additional_unflagged_examples.png','Initial request: see additional unidentified-cell examples',
        'Additional examples from the same prespecified selection; these are illustrative images, not an accuracy estimate.')
    add(ORIGINAL/'spillover_suspect_examples.png','Comparison request: inspect computationally flagged examples',
        'Cells flagged for CD56-associated patterns, shown for comparison with the unflagged examples.')
    for roman in ['I','II','III','IV']:
        add(FOUR/f'motif_{roman}.png',f'Request: show two unidentified-cell examples for motif {roman}',
            'Original expanded layout: neighborhood merge, focal-cell zoom and six individual channels. Original I_1 is P4/FOV16/cell1767.')
    add(FOUR/'compact/compact_motifs.png','Request: make the four-motif figure compact',
        'First compact proposal: four motif blocks, two example rows each, a neighborhood merge and four close-up channels.')
    add(EXTRA/'HEV_DC_layout_before_replacement.png','Request: focus on HEV markers in green and DC markers in red',
        'Historical layout before the requested replacement: motif-I first example is P4/FOV16/cell1767. Reconstructed from the saved original selection and unchanged raw images.')
    add(FOUR/'marker_focused/compact_motifs.png','Request: replace the top-left unidentified-cell example',
        'Current layout: I_1 is P14/FOV132/cell7775; the other seven examples are retained. HEV-focused channels are green and DC-focused channels red.')
    add(MECA/'neighbors_with_segmentation.png','Request: show the segmentation of the annotated HEV neighbors',
        'Both cells are outlined. White dashed: unidentified; cyan solid: annotated HEV. Includes MECA-79, CD31 and DNA neighbor close-ups.')
    add(MECA/'requested_cells/requested_cells.png','Request: show FOV36/cell651 and FOV124/cell1197',
        'The two specifically requested motif-I cells and their annotated HEV neighbors, including MECA-79, CD31, SMA and DNA.')
    # Include every associated PNG, including outline-free viewer variants and full
    # motif context panels. Duplicated paths are omitted, not distinct image versions.
    for folder in [ORIGINAL,FOUR,MECA]:
        for path in sorted(folder.rglob('*.png')):
            stem=path.stem
            if path in seen:continue
            if 'no_outlines' in stem:
                section='Appendix: corresponding views without focal outlines'
                title=f'Request: inspect raw signal without focal outlines — {stem.replace("_no_outlines","")}'
                request='Companion view with focal outlines removed; raw-image display limits and the selected cells are retained.'
            elif 'motif_context' in stem:
                section='Appendix: locations of all cells in the motif'
                title=f'Request: see the complete motif neighborhood — {stem.replace("_motif_context","")}'
                request='Spatial locations of the four selected motif partners. Labels are the existing annotations, not independent validation.'
            elif path.parent.name=='requested_cells':
                section='Appendix: individually requested cells'
                title=f'Request: inspect {stem.replace("_cell_",", cell ")}'
                request='Individual export with the unidentified and annotated HEV masks, plus HEV-associated marker channels.'
            else:
                section='Appendix: individual-cell microscopy sheets'
                title=f'Request: inspect an individual unidentified-cell example — {stem}'
                request='Expanded individual-cell view accompanying the earlier montage. Source set: '+str(path.parent.relative_to(OUTPUTS))+'.'
            add(path,title,request,section)
    add(EXTRA/'HEV_DC_layout_before_replacement_no_outlines.png',
        'Request: inspect the earlier HEV/DC layout without focal outlines',
        'Historical marker-focused layout before replacing I_1; original I_1 is P4/FOV16/cell1767.',
        'Appendix: corresponding views without focal outlines')
    for paths,section,request in [
        (original_supp,'Appendix: additional channels for the first compact proposal',
         'Additional profile channels from the original eight-cell selection; corresponds to the first compact figure supplemental PDF.'),
        (current_supp,'Appendix: additional channels for the current HEV/DC layout',
         'Additional profile channels for the current selection, including the replacement I_1. Reference-threshold positivity is descriptive.'),
        (complete,'Appendix: complete highlighted HEV and DC marker sets',
         'All twelve requested channels for each current example. Top: CD31, MECA-79, SMA, dsDNA. Bottom: eight highlighted DC-profile channels.')]:
        for path in paths:add(path,'Request: examine the remaining marker evidence — '+path.stem,request,section)
    main_count=12
    entries[main_count:]=sorted(entries[main_count:],key=lambda e:(e['section'],e['path']))
    pdf_path=OUT/'all_unidentified_cell_images.pdf'
    manifest=[]
    with PdfPages(pdf_path,metadata={'Title':'Unidentified-cell microscopy: complete figure collection',
        'Author':'CISM analysis','Subject':'Requested images, figure revisions, neighbors and additional channels'}) as pdf:
        fig=plt.figure(figsize=(8.5,11));fig.text(.08,.91,'Unidentified-cell microscopy',fontsize=22,weight='bold')
        fig.text(.08,.86,'Complete figure collection',fontsize=17)
        intro='All microscopy image exports associated with this conversation, including figure revisions, individual examples, full motif context, outline-free variants, and the additional channels from the linked supplemental PDFs. Each image page begins with the request it addresses.'
        fig.text(.08,.79,textwrap.fill(intro,83),fontsize=11,va='top',linespacing=1.6)
        notes='The opening pages follow the figure-development sequence. Earlier and current selections are identified explicitly. I_1 originally refers to P4/FOV16/cell1767; after replacement it refers to P14/FOV132/cell7775. All later HEV-neighbor images use the replacement where relevant.'
        fig.text(.08,.63,textwrap.fill(notes,83),fontsize=11,va='top',linespacing=1.6)
        notes='Images are measurements from the original TIFFs and existing masks. Existing annotations are not independent ground truth. Scale bars are pixels. A negative computational screen or a selected image does not establish absence of spillover or validate an HEV/DC identity.'
        fig.text(.08,.47,textwrap.fill(notes,83),fontsize=11,va='top',linespacing=1.6)
        fig.text(.08,.30,f'{len(entries)} image pages plus this introduction.\nSources and page-by-page titles: image_manifest.csv.\nComplete motif-I/II MECA-79 pair measurements: ../meca79_motif_I_II/index.html',fontsize=10,linespacing=1.7)
        pdf.savefig(fig);plt.close(fig)
        for i,entry in enumerate(entries,2):
            path=Path(entry['path'])
            with Image.open(path) as im:
                ratio=im.height/im.width
                im=im.convert('RGB');im.thumbnail((2600,2900),Image.Resampling.LANCZOS)
                pixels=np.array(im)
            landscape=ratio<.83
            size=(12,8.5) if landscape else (8.5,11)
            fig=plt.figure(figsize=size)
            width=108 if landscape else 80
            fig.text(.04,.965,entry['section'],fontsize=9,color='#546677',va='top')
            fig.text(.04,.932,textwrap.fill(entry['title'],width),fontsize=13,weight='bold',va='top')
            fig.text(.04,.865,textwrap.fill(entry['request'],145 if landscape else 103),fontsize=8.5,va='top',linespacing=1.4)
            ax=fig.add_axes([.025,.065,.95,.72]);ax.imshow(pixels,interpolation='nearest');ax.axis('off')
            fig.text(.04,.025,str(path.relative_to(ROOT)),fontsize=5.3,color='#666666')
            fig.text(.96,.025,f'{i} / {len(entries)+1}',fontsize=8,ha='right')
            pdf.savefig(fig,dpi=240);plt.close(fig)
            manifest.append(dict(page=i,section=entry['section'],title=entry['title'],request=entry['request'],
                image=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if i%10==0:print(f'Compiled {i} pages',flush=True)
    pd.DataFrame(manifest).to_csv(OUT/'image_manifest.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pdf=str(pdf_path),image_pages=len(entries),total_pages=len(entries)+1,
        coverage='Every PNG under the three associated image-output trees, plus historical replacement and PDF-only supplementary channel panels.'),indent=2))
    print(f'Complete: {pdf_path} ({pdf_path.stat().st_size/1e6:.1f} MB; {len(entries)+1} pages)',flush=True)


if __name__=='__main__':main()
