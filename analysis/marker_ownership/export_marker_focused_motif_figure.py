"""HEV/DC marker-focused microscopy layout using the same eight selected cells."""

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
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import export_four_motif_microscopy as source
from analysis.marker_ownership import export_spillover_microscopy_panels as imaging

BASE = source.ROOT/'analysis/outputs/unidentified_motif_I_IV_examples'
OUT = BASE/'marker_focused'
HEV_MARKERS = ['CD31', 'MECA-79', 'SMA', 'dsDNA']
DC_MARKERS = ['CD11c', 'CD3', 'CD4', 'CD45', 'HLA-DR-DP-DQ',
              'HLA-class-1-A-B-C', 'IDO-1', 'TIM-3']
MARKERS = {
    'I': HEV_MARKERS,
    'II': HEV_MARKERS,
    'III': ['CD11c', 'CD4', 'HLA-DR-DP-DQ', 'IDO-1'],
    'IV': ['CD11c', 'CD4', 'HLA-DR-DP-DQ', 'IDO-1'],
}
COLORS = {**{m: (.15,1,.1) for m in HEV_MARKERS},
          **{m: (1,.15,.1) for m in DC_MARKERS}}
LABEL = {'HLA-DR-DP-DQ': 'HLA-II', 'HLA-class-1-A-B-C': 'HLA-I'}


def label(marker):
    return LABEL.get(marker, marker)


def selection_path():
    override = OUT/'selection_manifest.csv'
    return override if override.exists() else BASE/'example_manifest.csv'


def replace_first_example(seed=42070):
    selected = pd.read_csv(selection_path())
    previous = selected.loc[selected.example_id.eq('I_1')].iloc[0]
    audit = pd.read_csv(source.AUDIT/'per_cell_summary.csv', low_memory=False).rename(
        columns={'in_motif':'original_20748339660_member'})
    hits = pd.read_csv(source.SOURCE/'4071751116_hits_wide.csv')
    focal = hits[['patient','FOV','cell3']].rename(columns={
        'patient':'patient number','FOV':'fov','cell3':'cell_id'}).drop_duplicates()
    pool = focal.merge(audit,on=source.KEY,validate='one_to_one')
    pool = pool[pool.geometry_valid.eq(True) & pool.bright_dna_components.eq(1)
                & pool.mask_area.between(100,800) & pool.status_q90.eq('low_evidence')
                & ~pool['patient number'].isin(selected['patient number'])]
    if pool.empty: raise ValueError('No eligible replacement from an unused patient')
    cell = pool.sort_values(source.KEY).sample(n=1,random_state=seed).iloc[0]
    row = previous.copy()
    for name,value in cell.items(): row[name] = value
    archive_path = imaging.DATA/'raw_data'/f'{cell.fov}.zip'
    with ZipFile(archive_path) as archive:
        mask = imaging.read_image(archive,'TIFs/segmentation_labels.tiff')
    yy,xx = np.where(mask==int(cell.cell_id))
    half=max(20,int(np.ceil(max(abs(yy-cell['centroid-0']).max(),abs(xx-cell['centroid-1']).max())))+9)
    spatial=pd.read_csv(imaging.DATA/'mapped_cell_types.csv')
    lookup=spatial[spatial.fov.eq(cell.fov)&spatial['patient number'].eq(cell['patient number'])].set_index('cell_id')
    config=source.CONFIG[0]
    _,edges=source.parse_definition(source.SOURCE/'4071751116.txt')
    candidates=hits[hits.FOV.eq(cell.fov)&hits.patient.eq(cell['patient number'])&hits.cell3.eq(cell.cell_id)]
    def compactness(hit):
        coords={v:lookup.loc[int(hit[f'cell{ord(v)-ord("A")+1}'])][['centroid-0','centroid-1']].to_numpy(float) for v in config['types']}
        return sum(np.linalg.norm(coords[a]-coords[b]) for a,b in edges)
    instance=candidates.loc[candidates.apply(compactness,axis=1).idxmin()]
    for var,expected in config['types'].items():
        cid=int(instance[f'cell{ord(var)-ord("A")+1}']); partner=lookup.loc[cid]
        assert partner.pred_mapped==expected
        y,x=np.where(mask==cid)
        assert len(y) and np.hypot(y.mean()-partner['centroid-0'],x.mean()-partner['centroid-1'])<.75
    row['archive']=str(archive_path)
    row['channel_members']='{}'; row['display_limits_json']='{}'
    row['focal_zoom_halfwidth_px']=half
    row['selected_instance_json']=json.dumps({f'cell{i}':int(instance[f'cell{i}']) for i in range(1,5)})
    row['selection']=f'user-requested replacement of I_1; seed {seed}; existing low-evidence q90; valid geometry; one DNA component; 100–800 pixels; unused patient'
    for name in row.index:
        if name not in selected.columns: selected[name]=None
    selected.loc[selected.example_id.eq('I_1'),row.index]=row.values
    selected.to_csv(OUT/'selection_manifest.csv',index=False)
    (OUT/'replacement_record.json').write_text(json.dumps(dict(
        example_id='I_1',seed=seed,previous={k:str(previous[k]) for k in source.KEY},
        replacement={k:str(row[k]) for k in source.KEY},
        reason='User requested replacement; sampled before viewing the replacement image.'),indent=2))
    print(f'Replaced I_1: P{int(cell["patient number"])} {cell.fov} cell {int(cell.cell_id)}',flush=True)


def load_cases():
    selected = pd.read_csv(selection_path())
    assert len(selected) == 8 and not selected.duplicated(source.KEY).any()
    assert selected.groupby('roman').size().eq(2).all()
    cases = []; profiles = []
    for _, row in selected.iterrows():
        config = next(c for c in source.CONFIG if c['roman'] == row.roman).copy()
        config['merge'] = (['CD11c', 'MECA-79', 'dsDNA'] if row.roman == 'I'
                           else ['MECA-79', 'dsDNA'] if row.roman == 'II'
                           else ['CD11c', 'dsDNA'])
        types, edges = source.parse_definition(source.SOURCE/f'{config["motif"]}.txt')
        assert types == config['types']
        config['edges'] = edges
        table = pd.read_csv(source.AUDIT/'fovs'/f'{int(row["patient number"])}_{row.fov}'/'markers.csv.gz', low_memory=False)
        profile = table[table.cell_id.eq(row.cell_id)].copy()
        profile['example_id'] = row.example_id
        profile['in_main_figure'] = profile.marker.isin(MARKERS[row.roman])
        profiles.append(profile)
        # Show additional lineage-threshold-positive channels; these are descriptive
        # reference comparisons, not independent cell-type validation.
        extra = profile.loc[profile.lineage_whole_positive.eq(True)
                            & ~profile.marker.isin(MARKERS[row.roman]), 'marker'].tolist()
        # SMA is prominent in the existing III_2 whole-mask profile.
        if row.example_id == 'III_2' and 'SMA' not in extra:
            extra.append('SMA')
        channels = list(dict.fromkeys(MARKERS[row.roman] + config['merge'] + extra + HEV_MARKERS + DC_MARKERS))
        path = Path(row.archive)
        checkpoint = json.loads((source.AUDIT/'fovs'/f'{int(row["patient number"])}_{row.fov}'/'complete.json').read_text())
        assert path.stat().st_size == checkpoint['archive_size']
        assert path.stat().st_mtime_ns == checkpoint['archive_mtime_ns']
        with ZipFile(path) as archive:
            mask = imaging.read_image(archive, row.mask_member)
            images = {m: imaging.read_image(archive, f'TIFs/{m}.tif') for m in channels}
        assert all(im.shape == mask.shape for im in images.values())
        yy, xx = np.where(mask == int(row.cell_id))
        assert len(yy) and np.hypot(yy.mean()-row['centroid-0'], xx.mean()-row['centroid-1']) < .75
        limits = {m: max(float(np.quantile(im[im > 0], .995)), 1) if (im > 0).any() else 1
                  for m, im in images.items()}
        old_limits = json.loads(row.display_limits_json)
        for m in set(old_limits) & set(limits):
            assert np.isclose(old_limits[m], limits[m])
        cases.append(dict(row=row, config=config, images=images, mask=mask, limits=limits,
                          colors={**COLORS, 'dsDNA': (.2,.35,1)}, extra=extra,
                          context=imaging.bounds(row['centroid-0'], row['centroid-1'], 58, mask.shape),
                          zoom=imaging.bounds(row['centroid-0'], row['centroid-1'], int(row.focal_zoom_halfwidth_px), mask.shape)))
        print(f'Loaded {row.example_id}; additional profile channels: {", ".join(extra)}', flush=True)
    pd.concat(profiles).to_csv(OUT/'selected_cell_marker_profiles.csv', index=False)
    return cases


def image_panel(ax, case, marker=None, outlines=True):
    sl = case['context'] if marker is None else case['zoom']
    if marker is None:
        pixels = source.merged(case, sl)
    else:
        color = COLORS.get(marker, (1,1,1))
        pixels = imaging.normalized(case['images'][marker], case['limits'][marker], sl)[:,:,None]*np.array(color)
    ax.imshow(pixels, interpolation='nearest')
    if outlines:
        imaging.outline(ax, case['mask'][sl], int(case['row'].cell_id), width=.45)
    if marker is None:
        zoom = case['zoom']
        ax.add_patch(Rectangle((zoom[1].start-sl[1].start-.5, zoom[0].start-sl[0].start-.5),
                               zoom[1].stop-zoom[1].start, zoom[0].stop-zoom[0].start,
                               fill=False, edgecolor='white', linewidth=.45))
    # Same pixel scale per view; draw one bar in each panel without repeating text.
    h,w = case['mask'][sl].shape; length = 20 if marker is None else 10
    ax.plot([w*.08,w*.08+length],[h*.91,h*.91],color='white',lw=1.2,solid_capstyle='butt')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def diagram(ax, config):
    positions = {'A': (.12,.70), 'B': (.80,.70), 'C': (.80,.12), 'D': (.12,.12)}
    for a,b in config['edges']:
        x,y = positions[a]; u,v = positions[b]
        other = b if a == config['focal'] else a if b == config['focal'] else None
        target = 'hevs' if config['roman'] in ['I','II'] else 'DC'
        highlighted = other is not None and config['types'][other] == target
        color = ('#16923b' if target == 'hevs' else '#d62628') if highlighted else '#aaaaaa'
        ax.plot([x,u],[y,v],color=color,lw=1.4 if highlighted else .7,zorder=1)
    short = {'CD8':'CD8 T', 'APC':'CD4 APC', 'DC1':'DC', 'DC2':'DC'}
    for node,(x,y) in positions.items():
        focal = node == config['focal']
        ax.scatter(x,y,s=75 if focal else 53,c='#b4a6d8' if focal else '#d7e4df',
                   edgecolors='#624b91' if focal else '#83978e',linewidths=.6,zorder=2)
        text = 'U' if focal else short.get(config['labels'][node], config['labels'][node])
        ax.text(x,y+.17,text,ha='center',va='center',fontsize=5)
    ax.set_xlim(-.2,1.15); ax.set_ylim(-.1,1.05); ax.axis('off')


def compact(cases, outlines=True):
    fig = plt.figure(figsize=(7.2,5.8))
    for k, config in enumerate(source.CONFIG):
        pair = [c for c in cases if c['row'].roman == config['roman']]
        left = .025 + (k%2)*.50
        bottom = .54 if k < 2 else .105
        diagram(fig.add_axes([left,bottom+.29,.12,.10]), pair[0]['config'])
        fig.text(left+.137,bottom+.373,f'{"ABCD"[k]}  Motif {config["roman"]}',fontsize=9,weight='bold')
        is_hev = config['roman'] in ['I', 'II']
        fig.text(left+.137,bottom+.345,'HEV-focused channels' if is_hev else 'DC-focused channels',
                 fontsize=6.5,color='#137c32' if is_hev else '#b52222')
        merge_text = ('Merge: CD11c red; MECA-79 green; DNA blue' if config['roman']=='I'
                      else 'Merge: MECA-79 green; DNA blue' if config['roman']=='II'
                      else 'Merge: CD11c red; DNA blue')
        fig.text(left+.137,bottom+.323,merge_text,fontsize=4.7,color='#555555')
        grid = fig.add_gridspec(2,5,left=left,right=left+.45,bottom=bottom,top=bottom+.29,
                               wspace=.065,hspace=.26)
        for i, case in enumerate(pair):
            for j, marker in enumerate([None]+MARKERS[config['roman']]):
                ax = fig.add_subplot(grid[i,j]); image_panel(ax,case,marker,outlines)
                if i == 0: ax.set_title('Merge' if marker is None else label(marker),fontsize=6,pad=3)
                if j == 0:
                    row=case['row']
                    ax.set_xlabel(f'{row.Group} · P{int(row["patient number"])} · {row.fov}',fontsize=5,labelpad=2)
    fig.text(.025,.035,'Dashed: unidentified-cell mask   |   Box: close-up region   |   Bars: merge 20 px; individual channels 10 px',fontsize=6)
    fig.text(.025,.014,'Single channels: HEV set green; DC set red. DNA is nuclear context. Illustrative examples do not exclude spillover.',fontsize=5.6,color='#444444')
    return fig


def supplementary(cases):
    with PdfPages(OUT/'complete_HEV_DC_marker_panels.pdf') as pdf:
        for case in cases:
            fig = plt.figure(figsize=(10.8,4.4))
            grid = fig.add_gridspec(2,8,left=.025,right=.985,bottom=.13,top=.83,wspace=.09,hspace=.35)
            for i,markers in enumerate([HEV_MARKERS, DC_MARKERS]):
                for j,marker in enumerate(markers):
                    ax=fig.add_subplot(grid[i,j]); image_panel(ax,case,marker)
                    ax.set_title(label(marker),fontsize=8)
            fig.suptitle(f'{case["row"].example_id} · complete highlighted HEV and DC marker sets',fontsize=11)
            fig.text(.025,.065,'Top: CD31 / MECA-79 / SMA / dsDNA (green). Bottom: highlighted DC-profile channels (red).',fontsize=8)
            fig.text(.025,.025,'Dashed: unidentified-cell mask. Bars: 10 px. DNA is nuclear context; these channels are not individually cell-type-specific.',fontsize=7)
            pdf.savefig(fig,dpi=350); plt.close(fig)
    with PdfPages(OUT/'additional_profile_channels.pdf') as pdf:
        for case in cases:
            markers = ['dsDNA'] + case['extra']
            cols=4; rows=int(np.ceil(len(markers)/cols))
            fig,axes=plt.subplots(rows,cols,figsize=(7.2,rows*1.9+.7),squeeze=False)
            for ax,marker in zip(axes.flat,markers):
                image_panel(ax,case,marker); ax.set_title(label(marker),fontsize=8)
            for ax in list(axes.flat)[len(markers):]:ax.axis('off')
            fig.suptitle(f'{case["row"].example_id}: additional profile channels',fontsize=11)
            fig.text(.03,.015,'HEV-set channels green; DC-set channels red; others grayscale. Dashed: focal mask. Bars: 10 px. Positivity is descriptive.',fontsize=6)
            fig.tight_layout(rect=(0,.04,1,.93)); pdf.savefig(fig,dpi=350); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-top-left',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.replace_top_left: replace_first_example()
    cases=load_cases()
    for outlines,suffix in [(True,''),(False,'_no_outlines')]:
        fig=compact(cases,outlines)
        for ext in ['png','pdf','svg']:
            fig.savefig(OUT/f'compact_motifs{suffix}.{ext}',dpi=450,facecolor='white')
        plt.close(fig)
    supplementary(cases)
    manifest=pd.read_csv(selection_path())
    manifest['main_channels']=manifest.roman.map(lambda r:';'.join(MARKERS[r]))
    manifest['additional_channels']=[';'.join(c['extra']) for c in cases]
    manifest['channel_members']=[json.dumps({m:f'TIFs/{m}.tif' for m in c['images']}) for c in cases]
    manifest['display_limits_json']=[json.dumps(c['limits']) for c in cases]
    manifest['merge_channels']=[';'.join(c['config']['merge']) for c in cases]
    manifest.to_csv(OUT/'example_manifest.csv',index=False)
    (OUT/'README.md').write_text('''# HEV/DC marker-focused motif microscopy figure

Four blocks (I–IV), each with two examples and five columns: neighborhood merge and four aligned close-ups. Seven original examples are retained; I_1 was replaced at the user’s request by a seeded draw (42070) from eligible motif-I cells in an unused patient, before viewing that replacement. The current selection is stored in selection_manifest.csv and the old/new cell IDs in replacement_record.json. Segmentation and annotations are unchanged. I–II focus on the unidentified–HEV relationship: CD31, MECA-79, SMA and dsDNA. III–IV focus on the unidentified–DC relationship: CD11c, CD4, HLA-II and IDO-1. complete_HEV_DC_marker_panels.pdf shows all twelve requested channels for all eight examples, including the additional red-highlighted CD3, CD45, HLA-I and TIM-3 channels. additional_profile_channels.pdf shows further channels positive against the existing lineage reference threshold, with SMA for III_2; thresholds are descriptive rather than identity validation. selected_cell_marker_profiles.csv contains all 36 measured markers for each example.

Single channels use green for the HEV-focused set (CD31, MECA-79, SMA, dsDNA) and red for the DC-focused set (CD11c, CD3, CD4, CD45, HLA-II, HLA-I, IDO-1, TIM-3). These sets follow the user's highlighted expression profiles, not a claim that every marker is cell-type-specific. In particular, dsDNA provides nuclear context, and CD3/CD4 are not independent DC identity markers. DNA remains blue only in neighborhood composites to distinguish nuclei from neighboring marker signal; merge channels and colors are explicit above each block. HLA-II abbreviates HLA-DR-DP-DQ; HLA-I abbreviates HLA-class-1-A-B-C. Other supplemental markers are grayscale. Diagram edges highlight unidentified–HEV green in I–II and unidentified–DC red in III–IV. Thin dashed white outlines show the original focal mask; solid rectangles locate the close-up within the neighborhood. Bars: 20 pixels in neighborhoods; 10 pixels in close-ups. Physical pixel size has not been established.

Original TIFFs are read directly from the archived FOVs. Mask centroid and dimensions are verified; archive size and mtime must match the previous spatial-screen checkpoint. Linear display limits are each full FOV/channel's positive-pixel 99.5th percentile and match previous exports. No new smoothing, background correction, or per-crop intensity scaling. Brightness is not comparable between different FOVs or channels. See example_manifest.csv for exact provenance and cell IDs. The original figure and supplementary panels remain available in the parent directory.

Caption suggestion: Representative motif-associated unidentified cells. For each motif, two cells from distinct patients are shown as a neighborhood composite and aligned individual-channel close-ups. DNA is blue in composites; the other composite channels and colors are listed above each block. Dashed outlines indicate the unidentified-cell segmentation; solid boxes locate the close-up. Scale bars: 20 pixels in composites and 10 pixels in individual channels. Examples were selected before viewing from cells unflagged by the existing spatial screen and are illustrative rather than a population-level validation of annotation accuracy.

Reproduce: MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=1 python analysis/marker_ownership/export_marker_focused_motif_figure.py
''')
    (OUT/'provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_manifest=str(selection_path()),input_manifest_sha256=hashlib.sha256(selection_path().read_bytes()).hexdigest(),
        main_channels=MARKERS,complete_HEV_channels=HEV_MARKERS,complete_DC_channels=DC_MARKERS,
        physical_size_mm=[182.88,147.32],dpi=450),indent=2))
    print(f'Complete: {OUT}',flush=True)


if __name__=='__main__': main()
