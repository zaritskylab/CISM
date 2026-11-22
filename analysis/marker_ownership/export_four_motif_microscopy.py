"""Two actual microscopy examples for each attached motif I–IV.

Uses existing whole-cohort unidentified-cell screen for illustrative selection;
does not compute new motif-specific spillover thresholds or change annotations.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import html
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import export_spillover_microscopy_panels as imaging

ROOT = _REPO_ROOT
AUDIT = ROOT/'analysis/outputs/unidentified_spillover_20748339660'
SOURCE = Path('/groups/assafza_group/assafza/Omri/melanoma_motif_instances')
KEY = ['patient number', 'fov', 'cell_id']
CONFIG = [
    dict(roman='I', motif=4071751116, focal='C',
         description='DC · B cell · unidentified · HEV',
         types={'A':'b cell', 'B':'DC', 'C':'unidentified', 'D':'hevs'},
         labels={'A':'B', 'B':'DC', 'C':'U', 'D':'HEV'},
         merge=['CD11c', 'MECA-79', 'dsDNA'],
         markers=['CD11c', 'HLA-DR-DP-DQ', 'CD20', 'MECA-79', 'CD31', 'dsDNA']),
    dict(roman='II', motif=15341317580, focal='B',
         description='CD8 T · Treg · unidentified · HEV',
         types={'A':'cd8 t cell', 'B':'unidentified', 'C':'hevs', 'D':'treg'},
         labels={'A':'CD8', 'B':'U', 'C':'HEV', 'D':'Treg'},
         merge=['CD3', 'MECA-79', 'dsDNA'],
         markers=['CD3', 'CD8', 'CD4', 'Foxp3', 'MECA-79', 'dsDNA']),
    dict(roman='III', motif=17763644908, focal='C',
         description='DC · unidentified · vessel · CD8 T',
         types={'A':'cd8 t cell', 'B':'blood vessels', 'C':'unidentified', 'D':'DC'},
         labels={'A':'CD8', 'B':'Vessel', 'C':'U', 'D':'DC'},
         merge=['CD11c', 'CD31', 'dsDNA'],
         markers=['CD11c', 'HLA-DR-DP-DQ', 'CD31', 'CD3', 'CD8', 'dsDNA']),
    dict(roman='IV', motif=4880694732, focal='A',
         description='CD4 APC · unidentified · DC · DC',
         types={'A':'unidentified', 'B':'DC', 'C':'DC', 'D':'cd4 apc'},
         labels={'A':'U', 'B':'DC1', 'C':'DC2', 'D':'APC'},
         merge=['CD11c', 'CD4', 'dsDNA'],
         markers=['CD11c', 'HLA-DR-DP-DQ', 'DC-SIGN', 'CD3', 'CD4', 'dsDNA']),
]


def parse_definition(path):
    types = {}; edges = []
    for line in path.read_text().splitlines():
        match = re.match(r'\s*([A-D])\.type\s*=\s*(.+?)\s*$', line)
        if match: types[match[1]] = match[2]
        match = re.match(r'\s*([A-D])\s*->\s*([A-D])\s*$', line)
        if match: edges.append(tuple(match.groups()))
    return types, edges


def select_cells(audit):
    rng = np.random.default_rng(42069)
    selections = []; coverage = []; used_cells = set(); used_patients = set()
    for config in CONFIG:
        mid = config['motif']
        types, edges = parse_definition(SOURCE/f'{mid}.txt')
        if types != config['types']: raise ValueError(f'Definition mismatch for motif {mid}')
        config['edges'] = edges
        hits = pd.read_csv(SOURCE/f'{mid}_hits_wide.csv')
        focal_column = f'cell{ord(config["focal"])-ord("A")+1}'
        focal = hits[['patient', 'FOV', focal_column]].rename(columns={'patient':'patient number','FOV':'fov',focal_column:'cell_id'}).drop_duplicates()
        matched = focal.merge(audit, on=KEY, how='inner', validate='one_to_one')
        pool = matched[matched.geometry_valid.eq(True) & matched.bright_dna_components.eq(1)
                       & matched.mask_area.between(100, 800) & matched.status_q90.eq('low_evidence')].copy()
        coverage.append(dict(roman=config['roman'], motif_id=mid, all_hit_rows=len(hits),
            unique_NN_NP_unidentified=len(matched), eligible_unflagged=len(pool),
            NN_eligible=int(pool.Group.eq('NN').sum()), NP_eligible=int(pool.Group.eq('NP').sum())))
        groups = ['NN', 'NP'] if set(pool.Group) == {'NN','NP'} else [pool.Group.iloc[0]]*2 if len(pool) else []
        if len(groups) != 2: raise ValueError(f'Cannot select two examples for motif {mid}')
        selected_this = set()
        for example_number, group in enumerate(groups, 1):
            candidates = pool[pool.Group.eq(group)]
            candidates = candidates[~candidates.apply(lambda r: tuple(r[k] for k in KEY) in used_cells, axis=1)]
            preferred = candidates[~candidates['patient number'].isin(used_patients)]
            if len(preferred): candidates = preferred
            candidates = candidates[~candidates['patient number'].isin(selected_this)]
            if candidates.empty: raise ValueError(f'Insufficient distinct-patient candidates for motif {mid}')
            row = candidates.loc[rng.choice(candidates.index)].copy()
            row['motif_id'] = mid; row['roman'] = config['roman']
            row['example_id'] = f'{config["roman"]}_{example_number}'
            row['focal_variable'] = config['focal']
            used_cells.add(tuple(row[k] for k in KEY)); used_patients.add(row['patient number']); selected_this.add(row['patient number'])
            selections.append(row)
    return pd.DataFrame(selections).reset_index(drop=True), pd.DataFrame(coverage)


def color_map(config):
    colors = {'CD11c': (1,.15,.1), 'CD3': (1,.15,.1), 'CD4': (1,.1,.85),
              'HLA-DR-DP-DQ': (0,.85,1), 'DC-SIGN': (1,.65,.05),
              'CD20': (.85,.1,1), 'CD8': (1,.65,.05), 'Foxp3': (0,.85,1),
              'CD31': (.15,1,.1), 'MECA-79': (.15,1,.1), 'dsDNA': (.2,.35,1)}
    colors[config['merge'][0]] = (1,.15,.1)
    colors[config['merge'][1]] = (.15,1,.1)
    return colors


def merged(case, sl):
    result = np.zeros(case['mask'][sl].shape+(3,))
    for marker in case['config']['merge']:
        result += imaging.normalized(case['images'][marker], case['limits'][marker], sl)[:,:,None]*np.asarray(case['colors'][marker])
    return np.clip(result, 0, 1)


def draw(ax, case, column, outlines=True, title=True):
    sl = case['context'] if column == 0 else case['zoom']
    if column < 2:
        image = merged(case, sl); label = 'Neighborhood' if column == 0 else 'Focal-cell zoom'
    else:
        marker = case['config']['markers'][column-2]
        color = np.ones(3) if marker == 'dsDNA' else case['colors'][marker]
        image = imaging.normalized(case['images'][marker], case['limits'][marker], sl)[:,:,None]*color
        label = marker
    ax.imshow(image, interpolation='nearest')
    if outlines: imaging.outline(ax, case['mask'][sl], int(case['row'].cell_id))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    if title: ax.set_title(label, fontsize=10)
    if column < 2: imaging.scale_bar(ax, case['mask'][sl].shape, 20 if column == 0 else 10)


def legend(config):
    return f'Merge: {config["merge"][0]} red · {config["merge"][1]} green · DNA blue'


def two_example_figure(cases, outlines=True):
    config = cases[0]['config']
    fig, axes = plt.subplots(2, 8, figsize=(18.3, 6.8))
    for i, case in enumerate(cases):
        row = case['row']
        for j, ax in enumerate(axes[i]): draw(ax, case, j, outlines, title=i == 0)
    fig.suptitle(f'Motif {config["roman"]} · {config["description"]}\nTwo motif-associated unidentified cells', fontsize=14, y=.98)
    fig.text(.55,.06,legend(config)+'   |   Dashed white: unidentified-cell mask   |   Bars: 20 px (context), 10 px (zoom)',ha='center',fontsize=9)
    fig.text(.55,.025,'Examples were selected as unflagged by the existing spatial screen; these images do not establish absence of spillover.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.105,right=.99,bottom=.16,top=.80,hspace=.25,wspace=.10)
    for i, case in enumerate(cases):
        row = case['row']
        pos = axes[i,0].get_position()
        fig.text(.05, (pos.y0 + pos.y1) / 2,
                 f'Example {i+1} · {row.Group}\nP{int(row["patient number"])} · {row.fov}\ncell {int(row.cell_id)}',
                 ha='center', va='center', fontsize=9)
    return fig


def single_figure(case, outlines=True):
    config, row = case['config'], case['row']
    fig, axes = plt.subplots(2,4,figsize=(11.8,7.6))
    for j, ax in enumerate(axes.ravel()): draw(ax,case,j,outlines)
    fig.suptitle(f'Motif {config["roman"]} · {config["description"]}\n{row.example_id} · {row.Group} · P{int(row["patient number"])} · {row.fov} · unidentified cell {int(row.cell_id)}',fontsize=12)
    fig.text(.5,.055,legend(config)+'   |   Dashed white: focal-cell mask',ha='center',fontsize=9)
    fig.text(.5,.025,'No added smoothing; scale bars in pixels. Not flagged by the existing screen does not mean spillover is excluded.',ha='center',fontsize=8,color='#444444')
    fig.subplots_adjust(left=.02,right=.985,bottom=.14,top=.83,hspace=.28,wspace=.12)
    return fig


def context_figure(case):
    config, row, instance, frame = case['config'], case['row'], case['instance'], case['spatial']
    ids = [int(instance[f'cell{i}']) for i in range(1,5)]
    members = frame.set_index('cell_id').loc[ids]
    yy, xx = members['centroid-0'], members['centroid-1']
    sl = (slice(max(0,int(yy.min())-22),min(case['mask'].shape[0],int(np.ceil(yy.max()))+23)),
          slice(max(0,int(xx.min())-22),min(case['mask'].shape[1],int(np.ceil(xx.max()))+23)))
    fig, axes = plt.subplots(1,2,figsize=(10,5.3))
    for ax in axes:
        ax.imshow(merged(case,sl),interpolation='nearest');ax.axis('off')
        imaging.scale_bar(ax,case['mask'][sl].shape,20)
    imaging.outline(axes[1],case['mask'][sl],int(row.cell_id),1)
    for var,label in config['labels'].items():
        cell = members.loc[int(instance[f'cell{ord(var)-ord("A")+1}'])]
        x,y=cell['centroid-1']-sl[1].start,cell['centroid-0']-sl[0].start
        color='#ffe66d' if var==config['focal'] else '#53eeff'
        axes[1].plot(x,y,'o',markerfacecolor='none',markeredgecolor=color,markersize=12,markeredgewidth=.8)
        axes[1].annotate(label,(x,y),xytext=(7,-12),textcoords='offset points',color=color,fontsize=11,fontweight='bold',arrowprops=dict(arrowstyle='-',color=color,lw=.5))
    axes[0].set_title('Marker image'); axes[1].set_title('Same image + motif cell locations')
    fig.suptitle(f'Motif {config["roman"]} · {row.example_id} · {row.fov} · unidentified cell {int(row.cell_id)}',fontsize=12)
    fig.text(.5,.055,legend(config),ha='center',fontsize=9)
    fig.text(.5,.025,'U = unidentified; other labels show existing cell annotations, not independent visual validation.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.92))
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=imaging.DATA)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/outputs/unidentified_motif_I_IV_examples')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    audit=pd.read_csv(AUDIT/'per_cell_summary.csv',low_memory=False).rename(columns={'in_motif':'original_20748339660_member'})
    selected,coverage=select_cells(audit)
    selected.to_csv(args.output/'selected_cells.csv',index=False)
    coverage.to_csv(args.output/'selection_coverage.csv',index=False)
    spatial=pd.read_csv(args.data_dir/'mapped_cell_types.csv')
    cases=[]; records=[]; source_stats={}
    for config in CONFIG:
        hits=pd.read_csv(SOURCE/f'{config["motif"]}_hits_wide.csv')
        subset=selected[selected.roman.eq(config['roman'])]
        focal_column=f'cell{ord(config["focal"])-ord("A")+1}'
        for _,row in subset.iterrows():
            archive_path=args.data_dir/'raw_data'/f'{row.fov}.zip'
            checkpoint=json.loads((AUDIT/'fovs'/f'{int(row["patient number"])}_{row.fov}'/'complete.json').read_text())
            if archive_path.stat().st_size!=checkpoint['archive_size'] or archive_path.stat().st_mtime_ns!=checkpoint['archive_mtime_ns']:
                raise ValueError('Raw archive has changed since the spatial screen')
            source_stats[str(archive_path)]=dict(size=archive_path.stat().st_size,mtime_ns=archive_path.stat().st_mtime_ns)
            with ZipFile(archive_path) as archive:
                mask=imaging.read_image(archive,'TIFs/segmentation_labels.tiff')
                channels=list(dict.fromkeys(config['markers']+config['merge']))
                images={marker:imaging.read_image(archive,f'TIFs/{marker}.tif') for marker in channels}
            if any(image.shape!=mask.shape for image in images.values()):raise ValueError('Channel/mask dimension mismatch')
            limits={marker:max(float(np.quantile(image[image>0],.995)),1) if (image>0).any() else 1 for marker,image in images.items()}
            frame=spatial[spatial.fov.eq(row.fov)&spatial['patient number'].eq(row['patient number'])]
            lookup=frame.set_index('cell_id')
            candidates=hits[hits.FOV.eq(row.fov)&hits.patient.eq(row['patient number'])&hits[focal_column].eq(row.cell_id)]
            def compactness(hit):
                coords={var:lookup.loc[int(hit[f'cell{ord(var)-ord("A")+1}'])][['centroid-0','centroid-1']].to_numpy(float) for var in config['types']}
                return sum(np.linalg.norm(coords[a]-coords[b]) for a,b in config['edges'])
            instance=candidates.loc[candidates.apply(compactness,axis=1).idxmin()]
            for var,expected in config['types'].items():
                cellid=int(instance[f'cell{ord(var)-ord("A")+1}']);cell=lookup.loc[cellid]
                if cell.pred_mapped!=expected:raise ValueError(f'Motif partner annotation mismatch: {row.example_id} {var}')
                yy,xx=np.where(mask==cellid)
                if not len(yy) or np.hypot(yy.mean()-cell['centroid-0'],xx.mean()-cell['centroid-1'])>.75:
                    raise ValueError('Motif partner mask mismatch')
            yy,xx=np.where(mask==int(row.cell_id));y,x=row['centroid-0'],row['centroid-1']
            half=max(20,int(np.ceil(max(abs(yy-y).max(),abs(xx-x).max())))+9)
            case=dict(config=config,row=row,instance=instance,spatial=frame,images=images,mask=mask,limits=limits,colors=color_map(config),
                      context=imaging.bounds(y,x,58,mask.shape),zoom=imaging.bounds(y,x,half,mask.shape))
            cases.append(case)
            for outlines,suffix in [(True,''),(False,'_no_outlines')]:
                fig=single_figure(case,outlines)
                for ext in ['png','pdf']:fig.savefig(args.output/f'{row.example_id}{suffix}.{ext}',dpi=220,bbox_inches='tight')
                plt.close(fig)
            fig=context_figure(case)
            for ext in ['png','pdf']:fig.savefig(args.output/f'{row.example_id}_motif_context.{ext}',dpi=220,bbox_inches='tight')
            plt.close(fig)
            record=row.to_dict();record.update(archive=str(archive_path),mask_member='TIFs/segmentation_labels.tiff',
                channel_members=json.dumps({marker:f'TIFs/{marker}.tif' for marker in channels}),display_limits_json=json.dumps(limits),
                motif_definition=str(SOURCE/f'{config["motif"]}.txt'),motif_hits=str(SOURCE/f'{config["motif"]}_hits_wide.csv'),
                selected_instance_json=json.dumps({f'cell{i}':int(instance[f'cell{i}']) for i in range(1,5)}),
                focal_zoom_halfwidth_px=half,selection='seeded existing-screen low-evidence q90; valid geometry; one DNA component; 100–800 mask pixels')
            records.append(record)
            print(f'{row.example_id}: motif {config["motif"]}, {row.Group} P{int(row["patient number"])} {row.fov} cell {int(row.cell_id)}',flush=True)
    with PdfPages(args.output/'all_four_motifs.pdf') as pdf:
        for config in CONFIG:
            pair=[c for c in cases if c['config']['roman']==config['roman']]
            for outlines,suffix in [(True,''),(False,'_no_outlines')]:
                fig=two_example_figure(pair,outlines)
                for ext in ['png','pdf']:fig.savefig(args.output/f'motif_{config["roman"]}{suffix}.{ext}',dpi=220,bbox_inches='tight')
                if outlines:pdf.savefig(fig,dpi=220)
                plt.close(fig)
    pd.DataFrame(records).to_csv(args.output/'example_manifest.csv',index=False)
    options=''.join(f'<option value="{r.example_id}">Motif {r.roman}, example {r.example_id.split("_")[-1]} · {r.Group} · {r.fov} · U{int(r.cell_id)}</option>' for _,r in selected.iterrows())
    (args.output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Two examples per motif I–IV</title>
<style>body{font:16px/1.5 system-ui;max-width:1450px;margin:25px auto;padding:0 20px}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #ddd}img{max-width:100%}select{font:inherit;margin:5px}</style><h1>Unidentified cells: two examples per motif I–IV</h1><p>Actual marker TIFFs and existing segmentation masks. Selected as unflagged by the existing spatial screen; this is not proof of absence of spillover.</p><p><a href="all_four_motifs.pdf">All four motifs (PDF)</a> · <a href="example_manifest.csv">Source manifest</a></p><header><select id="case" onchange="show()">'''+options+'''</select><select id="view" onchange="show()"><option value="pair">Two examples for motif</option><option value="single">Individual example</option><option value="context">All four motif-cell locations</option></select><label><input type="checkbox" id="outline" checked onchange="show()">Show focal-cell outlines</label></header><img id="image" alt="Microscopy panels"><p>Dashed white = unidentified-cell boundary. The composite channels and their colors are listed in each image. Bars are pixels; no physical calibration is assumed. Display limits are fixed per whole FOV/channel at the positive-pixel 99.5th percentile. No added smoothing or background subtraction.</p><script>function show(){let id=document.getElementById('case').value;let view=document.getElementById('view').value;let suffix=document.getElementById('outline').checked?'':'_no_outlines';let src=view==='pair'?'motif_'+id.split('_')[0]+suffix:view==='context'?id+'_motif_context':id+suffix;document.getElementById('image').src=src+'.png'}show()</script>''')
    (args.output/'README.md').write_text('''# Two unidentified-cell microscopy examples per motif I–IV

The attached diagrams match these saved definitions:

| Diagram | Motif ID | Focal node | Other cell types |
|---|---:|---|---|
| I | 4071751116 | C (cell3) | DC, B cell, HEV |
| II | 15341317580 | B (cell2) | CD8 T, HEV, Treg |
| III | 17763644908 | C (cell3) | CD8 T, vessel, DC |
| IV | 4880694732 | A (cell1) | DC, DC, CD4 APC |

Two unique cells per motif were selected before viewing images, with seed 42069. Eligibility: existing low-evidence status at the 90th-percentile screen (and therefore no flag at 95th/99th), valid mask geometry, one bright DNA component, and mask area 100–800 pixels. Patients are distinct within a motif and distinct across motifs when possible. One NN and one NP are selected if both are available; motif I has only NP matches in the hit file, so both examples are NP. Selection coverage is in selection_coverage.csv. These illustrative selections do not estimate motif-wide annotation accuracy.

Screen status is reused from the existing whole-NN/NP unidentified-cell analysis. Its matched reference populations excluded motif 20748339660, not the four motifs shown here; no motif-specific recalibration, new spillover prevalence estimate, or reannotation is performed. original_20748339660_member records overlap with that previous motif; all current examples belong to their separately recorded motif_id.

Raw images and masks are read from the original FOV archives. Archive size and mtime must match the earlier screen's checkpoint. Every selected motif instance is verified against its four existing mapped annotations and all four segmentation-mask centroids. Multiple instances containing the same unidentified cell are resolved by the shortest sum of the motif's edge distances, for compact visual context, not by image appearance. Exact partners, sources and display limits are in example_manifest.csv.

Each motif has a neighborhood composite, focal zoom, and six relevant single channels, using the same layout as the preceding microscopy figure. DC-associated markers are CD11c, HLA-DR-DP-DQ and/or DC-SIGN; B cells use CD20; HEVs use MECA-79; vessels use CD31; T-cell/Treg context uses CD3, CD4, CD8 and Foxp3. Marker patterns and existing annotations are not independently validated by their inclusion here. Composite colors are explicit per motif; single-marker colors can differ across motifs. DNA is blue in composites and grayscale in the separate DNA panel.

Display is linear and clipped to each whole FOV/channel's positive-pixel 99.5th percentile. No smoothing, denoising, new segmentation or background removal is added. Original pixels remain in the source archives. Dashed white contours show the focal mask; outline-free views are included. Bars are pixels because calibrated physical pixel size has not been established. Use within-image localization, rather than brightness comparisons across FOVs, to assess possible neighboring signal.

Open index.html to view each two-example motif figure, an individual example, or the full motif context with cell locations. all_four_motifs.pdf contains four two-example pages. PNG and PDF exports are available separately, with and without focal outlines. These images support visual review; they do not prove absence of spillover or correctness of unidentified annotations.

Reproduce: `MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=1 python analysis/marker_ownership/export_four_motif_microscopy.py`.
''')
    (args.output/'provenance.json').write_text(json.dumps(dict(seed=42069,source_archives=source_stats,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configuration=CONFIG,audit_source=str(AUDIT/'per_cell_summary.csv'),
        imported_imaging_sha256=hashlib.sha256(Path(imaging.__file__).read_bytes()).hexdigest()),indent=2))
    print(f'Complete: {args.output}',flush=True)


if __name__=='__main__':main()
