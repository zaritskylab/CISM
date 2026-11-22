"""Compact, traceable microscopy layout using the eight previously selected cells."""

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
OUT = BASE/'compact'
MARKERS = {
    'I': ['MECA-79', 'CD11c', 'HLA-DR-DP-DQ', 'CD20'],
    'II': ['MECA-79', 'CD3', 'CD8', 'Foxp3'],
    'III': ['CD11c', 'CD31', 'CD3', 'CD8'],
    'IV': ['CD11c', 'HLA-DR-DP-DQ', 'CD4', 'CD3'],
}
COLORS = {'MECA-79': (.15,1,.1), 'CD11c': (1,.15,.1),
          'HLA-DR-DP-DQ': (0,.85,1), 'CD20': (.85,.1,1),
          'CD3': (1,.15,.1), 'CD8': (1,.65,.05), 'Foxp3': (0,.85,1),
          'CD31': (.15,1,.1), 'CD4': (.9,.1,.8), 'dsDNA': (.2,.35,1)}
LABEL = {'HLA-DR-DP-DQ': 'HLA-II'}


def label(marker):
    return LABEL.get(marker, marker)


def load_cases():
    selected = pd.read_csv(BASE/'example_manifest.csv')
    assert len(selected) == 8 and not selected.duplicated(source.KEY).any()
    assert selected.groupby('roman').size().eq(2).all()
    cases = []; profiles = []
    for _, row in selected.iterrows():
        config = next(c for c in source.CONFIG if c['roman'] == row.roman).copy()
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
        channels = list(dict.fromkeys(MARKERS[row.roman] + config['merge'] + extra))
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
                          colors=COLORS, extra=extra,
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
        ax.plot([x,u],[y,v],color='#888888',lw=.7,zorder=1)
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
        merge = config['merge']
        fig.text(left+.137,bottom+.345,f'{merge[0]} / {merge[1]} / DNA',fontsize=5.7)
        fig.text(left+.137,bottom+.323,'red / '+('magenta' if merge[1]=='CD4' else 'green')+' / blue',fontsize=5.5,color='#555555')
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
    fig.text(.025,.014,'Illustrative cells unflagged by the spatial screen; images alone do not exclude spillover.',fontsize=6,color='#444444')
    return fig


def supplementary(cases):
    with PdfPages(OUT/'additional_profile_channels.pdf') as pdf:
        for case in cases:
            markers = ['dsDNA'] + case['extra']
            cols=4; rows=int(np.ceil(len(markers)/cols))
            fig,axes=plt.subplots(rows,cols,figsize=(7.2,rows*1.9+.7),squeeze=False)
            for ax,marker in zip(axes.flat,markers):
                image_panel(ax,case,marker); ax.set_title(label(marker),fontsize=8)
            for ax in list(axes.flat)[len(markers):]:ax.axis('off')
            fig.suptitle(f'{case["row"].example_id}: additional profile channels',fontsize=11)
            fig.text(.03,.015,'Main-figure colors retained; other markers grayscale. Dashed: focal mask. Bars: 10 px. Positivity is descriptive.',fontsize=6)
            fig.tight_layout(rect=(0,.04,1,.93)); pdf.savefig(fig,dpi=350); plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=load_cases()
    for outlines,suffix in [(True,''),(False,'_no_outlines')]:
        fig=compact(cases,outlines)
        for ext in ['png','pdf','svg']:
            fig.savefig(OUT/f'compact_motifs{suffix}.{ext}',dpi=450,facecolor='white')
        plt.close(fig)
    supplementary(cases)
    manifest=pd.read_csv(BASE/'example_manifest.csv')
    manifest['main_channels']=manifest.roman.map(lambda r:';'.join(MARKERS[r]))
    manifest['additional_channels']=[';'.join(c['extra']) for c in cases]
    manifest.to_csv(OUT/'example_manifest.csv',index=False)
    (OUT/'README.md').write_text('''# Compact motif microscopy figure

Four blocks (I–IV), each with two examples and five columns: neighborhood merge and four aligned close-ups. Uses the same eight cells as the existing figure, with no new selection, segmentation or annotation. Channels were chosen for motif context; additional channels positive against the existing lineage reference threshold are in additional_profile_channels.pdf, along with SMA for III_2 because it is prominent in the saved profile. These thresholds are descriptive and do not validate cell identity. selected_cell_marker_profiles.csv contains all 36 measured markers for each example.

Main marker colors are consistent across motifs: CD11c/CD3 red, MECA-79/CD31 green, HLA-II/Foxp3 cyan, CD20/CD4 magenta, CD8 orange. DNA is blue in neighborhood merges. HLA-II abbreviates HLA-DR-DP-DQ. Additional channels use grayscale unless they have a defined main-figure color. Thin dashed white outlines show the original focal mask; solid rectangles locate the close-up within the neighborhood. Bars: 20 pixels in neighborhoods; 10 pixels in close-ups. Physical pixel size has not been established.

Original TIFFs are read directly from the archived FOVs. Mask centroid and dimensions are verified; archive size and mtime must match the previous spatial-screen checkpoint. Linear display limits are each full FOV/channel's positive-pixel 99.5th percentile and match previous exports. No new smoothing, background correction, or per-crop intensity scaling. Brightness is not comparable between different FOVs or channels. See example_manifest.csv for exact provenance and cell IDs. The original figure and supplementary panels remain available in the parent directory.

Caption suggestion: Representative motif-associated unidentified cells. For each motif, two cells from distinct patients are shown as a neighborhood composite and aligned individual-channel close-ups. DNA is blue in composites; the other composite channels and colors are listed above each block. Dashed outlines indicate the unidentified-cell segmentation; solid boxes locate the close-up. Scale bars: 20 pixels in composites and 10 pixels in individual channels. Examples were selected before viewing from cells unflagged by the existing spatial screen and are illustrative rather than a population-level validation of annotation accuracy.

Reproduce: MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=1 python analysis/marker_ownership/export_compact_motif_figure.py
''')
    (OUT/'provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_manifest_sha256=hashlib.sha256((BASE/'example_manifest.csv').read_bytes()).hexdigest(),
        main_channels=MARKERS,physical_size_mm=[182.88,147.32],dpi=450),indent=2))
    print(f'Complete: {OUT}',flush=True)


if __name__=='__main__': main()
