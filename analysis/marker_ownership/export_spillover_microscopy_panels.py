"""CellTune-style microscopy examples from actual motif-associated cells.

No new segmentation, denoising, or annotation is performed. Selection is
seeded within prespecified group/cluster/status strata, before image inspection.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import argparse
import hashlib
import html
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = _REPO_ROOT
BASE = ROOT/'analysis/outputs/unidentified_spillover_20748339660'
DATA = Path('/groups/assafza_group/assafza/Datasets/Melanoma')
HITS = Path('/groups/assafza_group/assafza/Omri/melanoma_motif_instances/20748339660_hits_wide.csv')
KEY = ['patient number', 'fov', 'cell_id']
CHANNELS = ['CD3', 'CD4', 'CD45RO', 'CD8', 'CD56', 'dsDNA', 'CD45', 'CD20', 'CD68', 'CD163', 'Foxp3', 'S100A9-Calprotectin']
COLORS = {'CD3': (1, .12, .10), 'CD4': (1, .10, .85), 'CD45RO': (0, .9, 1),
          'CD8': (1, .6, .05), 'CD56': (.15, 1, .1), 'dsDNA': (.2, .35, 1)}
ROLE_COLORS = {'U': '#ffe66d', 'M': '#39dcff', 'C': '#ff9c40', 'N': '#54ff70'}


def read_image(archive, name):
    with Image.open(BytesIO(archive.read(name))) as im:
        return np.array(im)


def choose_examples(cells):
    eligible = cells[cells.in_motif & cells.geometry_valid.eq(True)
                     & cells.bright_dna_components.eq(1) & cells.mask_area.between(120, 650)].copy()
    rng = np.random.default_rng(42069)
    selected = []; patients = set()
    for group in ['NN', 'NP']:
        for cluster8 in [True, False]:
            pool = eligible[eligible.Group.eq(group) & eligible.cluster_15.eq(8).eq(cluster8)
                            & eligible.status_q90.eq('low_evidence')]
            for _ in range(2):
                options = pool[~pool['patient number'].isin(patients)]
                if options.empty:
                    raise ValueError('Not enough distinct-patient examples in prespecified stratum')
                index = rng.choice(options.index)
                row = options.loc[index].copy()
                row['selection_stratum'] = f'unflagged_{group}_'+('cluster8' if cluster8 else 'other_cluster')
                row['panel_category'] = 'not_flagged'
                selected.append(row); patients.add(row['patient number'])
    for group in ['NN', 'NP']:
        pool = eligible[eligible.Group.eq(group) & eligible.flagged_markers_q95.fillna('').str.split(';').apply(lambda s: 'CD56' in s)]
        for _ in range(2):
            options = pool[~pool['patient number'].isin(patients)]
            if options.empty: options = pool[~pool.index.isin([r.name for r in selected])]
            if options.empty: raise ValueError('Not enough CD56-suspect examples')
            index = rng.choice(options.index)
            row = options.loc[index].copy()
            row['selection_stratum'] = f'CD56_flagged_{group}'
            row['panel_category'] = 'spillover_suspect'
            selected.append(row); patients.add(row['patient number'])
    result = pd.DataFrame(selected).reset_index(drop=True)
    result['example_id'] = [f'E{i:02d}' for i in range(1, len(result)+1)]
    assert not result.duplicated(KEY).any()
    return result


def bounds(y, x, half, shape):
    return slice(max(0, int(np.floor(y-half))), min(shape[0], int(np.ceil(y+half))+1)), slice(max(0, int(np.floor(x-half))), min(shape[1], int(np.ceil(x+half))+1))


def normalized(image, limit, sl):
    return np.clip(image[sl].astype(float)/limit, 0, 1)


def composite(images, limits, sl, channels=('CD3', 'CD56', 'dsDNA')):
    rgb = np.zeros(images[channels[0]][sl].shape+(3,))
    for marker in channels:
        rgb += normalized(images[marker], limits[marker], sl)[:, :, None]*np.asarray(COLORS[marker])
    return np.clip(rgb, 0, 1)


def outline(ax, local_mask, cell, width=.7):
    binary = local_mask == cell
    if binary.any() and (~binary).any():
        ax.contour(binary.astype(float), levels=[.5], colors=['white'], linewidths=width, linestyles='--', alpha=.9)


def scale_bar(ax, shape, length):
    height, width = shape
    x, y = width*.07, height*.90
    ax.plot([x, x+length], [y, y], color='white', lw=2, solid_capstyle='butt')
    ax.text(x, y-2, f'{length} px', color='white', fontsize=7, ha='left', va='bottom')


def draw_panel(ax, case, panel, outlines=True, title=True):
    row, images, limits, mask = case['row'], case['images'], case['limits'], case['mask']
    sl = case['context'] if panel == 0 else case['zoom']
    if panel in [0, 1]:
        rgb = composite(images, limits, sl)
        label = 'Neighborhood' if panel == 0 else 'Focal-cell zoom'
    else:
        marker = ['CD3', 'CD4', 'CD45RO', 'CD8', 'CD56', 'dsDNA'][panel-2]
        color = np.ones(3) if marker == 'dsDNA' else COLORS[marker]
        rgb = normalized(images[marker], limits[marker], sl)[:, :, None]*color
        label = marker
    ax.imshow(rgb, interpolation='nearest')
    if outlines: outline(ax, mask[sl], int(row.cell_id))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('black')
    for spine in ax.spines.values(): spine.set_visible(False)
    if title: ax.set_title(label, fontsize=10, pad=6)
    if panel in [0, 1]: scale_bar(ax, mask[sl].shape, 20 if panel == 0 else 10)


def status_text(row):
    return 'Not flagged at 90th, 95th or 99th percentile' if row.panel_category == 'not_flagged' else f'Spillover-suspect: {row.flagged_markers_q95}'


def individual_figure(case, outlines=True):
    row = case['row']
    fig, axes = plt.subplots(2, 4, figsize=(11.8, 7.6))
    for i, ax in enumerate(axes.ravel()): draw_panel(ax, case, i, outlines)
    fig.suptitle(f'{row.example_id} · {row.Group} · patient {int(row["patient number"])} · {row.fov} · unidentified cell {int(row.cell_id)}\n{status_text(row)}', fontsize=12)
    fig.text(.5, .035, 'Merged panels: CD3 red · CD56 green · DNA blue. Dashed white: focal segmentation. Scale bars are pixels.', ha='center', fontsize=9)
    fig.text(.5, .009, 'Raw channel TIFFs; linear display; no smoothing. A negative computational screen does not establish absence of spillover.', ha='center', fontsize=8, color='#444444')
    fig.subplots_adjust(left=.02, right=.985, bottom=.13, top=.83, hspace=.28, wspace=.12)
    return fig


def overview(cases, title, output, filename):
    for outlines, suffix in [(True, ''), (False, '_no_outlines')]:
        fig, axes = plt.subplots(len(cases), 8, figsize=(18.2, 2.65*len(cases)+1.2), squeeze=False)
        for i, case in enumerate(cases):
            row = case['row']
            for j, ax in enumerate(axes[i]): draw_panel(ax, case, j, outlines, title=i == 0)
            axes[i, 0].set_ylabel(f'{row.example_id} · {row.Group}\nP{int(row["patient number"])} · {row.fov}\ncell {int(row.cell_id)}', rotation=0, labelpad=55, fontsize=9, va='center')
        fig.suptitle(title, fontsize=15, y=.99)
        fig.text(.53, .035, 'Merge: CD3 red · CD56 green · DNA blue   |   Dashed white: focal-cell mask   |   Bars: 20 px (context), 10 px (zoom)', ha='center', fontsize=10)
        fig.text(.53, .012, 'Selection was based on computational status before viewing images. These examples are illustrative, not an estimate of annotation accuracy.', ha='center', fontsize=9, color='#444444')
        fig.subplots_adjust(left=.105, right=.99, bottom=.09, top=.91, hspace=.22, wspace=.10)
        for extension in ['png', 'pdf']:
            fig.savefig(output/f'{filename}{suffix}.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)


def motif_context(case, output):
    row, instance, frame = case['row'], case['instance'], case['spatial']
    members = frame.set_index('cell_id').loc[[int(instance[f'cell{i}']) for i in range(1, 5)]]
    ys, xs = members['centroid-0'], members['centroid-1']
    sl = (slice(max(0, int(ys.min())-20), min(case['mask'].shape[0], int(np.ceil(ys.max()))+21)),
          slice(max(0, int(xs.min())-20), min(case['mask'].shape[1], int(np.ceil(xs.max()))+21)))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5))
    for ax in axes:
        ax.imshow(composite(case['images'], case['limits'], sl), interpolation='nearest'); ax.axis('off')
        scale_bar(ax, case['mask'][sl].shape, 20)
    outline(axes[1], case['mask'][sl], int(row.cell_id), 1)
    for i, role in enumerate(['N', 'C', 'M', 'U'], start=1):
        cid = int(instance[f'cell{i}']); cell = frame[frame.cell_id.eq(cid)].iloc[0]
        x, y = cell['centroid-1']-sl[1].start, cell['centroid-0']-sl[0].start
        axes[1].plot(x, y, marker='o', markerfacecolor='none', markeredgecolor=ROLE_COLORS[role], markersize=12, markeredgewidth=.8)
        axes[1].annotate(role, (x, y), xytext=(8, -12), textcoords='offset points', color=ROLE_COLORS[role], fontsize=12, fontweight='bold', arrowprops=dict(arrowstyle='-', color=ROLE_COLORS[role], lw=.5))
    axes[0].set_title('Marker image'); axes[1].set_title('Same image + motif cell locations')
    fig.suptitle(f'{row.example_id} · motif 20748339660 · {row.fov}', fontsize=12)
    fig.text(.5, .05, 'U: Unidentified   M: Memory CD4 T   C: CD8 T   N: NK   (existing annotations)', ha='center', fontsize=9)
    fig.text(.5, .015, 'Merge: CD3 red · CD56 green · DNA blue. Location labels do not independently validate the annotations.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, .92))
    for ext in ['png', 'pdf']: fig.savefig(output/f'{row.example_id}_motif_context.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=BASE/'paper_style_examples')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(BASE/'cluster_association/cells_with_clusters.csv.gz', low_memory=False)
    selected = choose_examples(cells)
    spatial = pd.read_csv(args.data_dir/'mapped_cell_types.csv')
    hits = pd.read_csv(HITS)
    metadata = []; all_cases = []
    for fov, selected_fov in selected.groupby('fov'):
        archive_path = args.data_dir/'raw_data'/f'{fov}.zip'
        with ZipFile(archive_path) as archive:
            mask = read_image(archive, 'TIFs/segmentation_labels.tiff')
            images = {marker: read_image(archive, f'TIFs/{marker}.tif') for marker in CHANNELS}
            limits = {marker: max(float(np.quantile(image[image > 0], .995)), 1) if (image > 0).any() else 1 for marker, image in images.items()}
            for _, row in selected_fov.iterrows():
                y, x = row['centroid-0'], row['centroid-1']; own = mask == int(row.cell_id)
                yy, xx = np.where(own)
                if not len(yy) or np.hypot(yy.mean()-y, xx.mean()-x) > .75:
                    raise ValueError('Cell-mask correspondence failed')
                half = max(20, int(np.ceil(max(abs(yy-y).max(), abs(xx-x).max())))+9)
                frame = spatial[spatial.fov.eq(fov) & spatial['patient number'].eq(row['patient number'])]
                candidates = hits[hits.FOV.eq(fov) & hits.patient.eq(row['patient number']) & hits.cell4.eq(row.cell_id)]
                if candidates.empty: raise ValueError('Selected cell absent from motif hits')
                lookup = frame.set_index('cell_id')[['centroid-0', 'centroid-1']]
                def path_length(hit):
                    xy = lookup.loc[[int(hit[f'cell{i}']) for i in range(1, 5)]].to_numpy()
                    return np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
                instance = candidates.loc[candidates.apply(path_length, axis=1).idxmin()]
                case = dict(row=row, images=images, limits=limits, mask=mask,
                            context=bounds(y, x, 58, mask.shape), zoom=bounds(y, x, half, mask.shape),
                            instance=instance, spatial=frame)
                for outlines, suffix in [(True, ''), (False, '_no_outlines')]:
                    fig = individual_figure(case, outlines)
                    for extension in ['png', 'pdf']:
                        fig.savefig(args.output/f'{row.example_id}{suffix}.{extension}', dpi=220, bbox_inches='tight')
                    plt.close(fig)
                motif_context(case, args.output)
                record = row.to_dict()
                record.update(archive_path=str(archive_path), mask_member='TIFs/segmentation_labels.tiff',
                              zoom_halfwidth_px=half, context_halfwidth_px=58,
                              display_limits_json=json.dumps(limits), motif_N_cell=int(instance.cell1),
                              motif_C_cell=int(instance.cell2), motif_M_cell=int(instance.cell3), motif_U_cell=int(instance.cell4))
                metadata.append(record); all_cases.append(case)
                print(f'{row.example_id}: {fov} cell {int(row.cell_id)} ({row.panel_category})', flush=True)
    cases = {case['row'].example_id: case for case in all_cases}
    overview([cases[x] for x in ['E01', 'E03', 'E05', 'E07']],
             'Motif-associated unidentified cells · examples not flagged by the spatial screen', args.output, 'unflagged_examples')
    overview([cases[x] for x in ['E02', 'E04', 'E06', 'E08']],
             'Additional motif-associated unidentified cells · examples not flagged by the spatial screen', args.output, 'additional_unflagged_examples')
    overview([cases[x] for x in ['E09', 'E10', 'E11', 'E12']],
             'Comparison cases · CD56 spillover-suspect motif-associated unidentified cells', args.output, 'spillover_suspect_examples')
    manifest = pd.DataFrame(metadata).sort_values('example_id')
    manifest.to_csv(args.output/'example_manifest.csv', index=False)
    with PdfPages(args.output/'all_examples.pdf') as pdf:
        for eid in sorted(cases):
            fig = individual_figure(cases[eid]); pdf.savefig(fig, dpi=220); plt.close(fig)
    options = ''.join(f'<option value="{r.example_id}">{r.example_id} · {r.Group} · P{int(r["patient number"])} · {r.fov} · cell {int(r.cell_id)} · {r.panel_category}</option>' for _, r in manifest.iterrows())
    (args.output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Motif-cell microscopy panels</title>
<style>body{font:16px/1.5 system-ui;max-width:1450px;margin:25px auto;padding:0 20px;color:#26323b}header{position:sticky;top:0;background:white;padding:12px;border-bottom:1px solid #ddd}img{max-width:100%}select,button{font:inherit;margin:4px}</style>
<h1>Unidentified cells in motif 20748339660</h1><p>Actual channel images and existing segmentation masks. CellTune-style display inspired by <a href="https://www.nature.com/articles/s41592-026-03162-2">Bussi et al., figure 5f/k</a>. Eight examples were not flagged at any screening threshold; four comparison examples were flagged for CD56. These illustrative images do not establish population-wide absence of spillover.</p>
<p><a href="unflagged_examples.png">Unflagged overview</a> · <a href="spillover_suspect_examples.png">Suspect overview</a> · <a href="all_examples.pdf">All examples (PDF)</a> · <a href="example_manifest.csv">Source and selection manifest</a></p>
<header><button onclick="move(-1)">Previous</button><select id="case" onchange="show()">'''+options+'''</select><button onclick="move(1)">Next</button><label><input type="checkbox" id="outlines" checked onchange="show()">Focal outline</label><label><input type="checkbox" id="context" onchange="show()">Full motif context</label></header>
<img id="image" alt="Marker channels"><p>Dashed white: focal cell. Merge: CD3 red, CD56 green, DNA blue. Intensities use a fixed whole-FOV/channel positive-pixel 99.5th percentile display limit; brightness is not directly comparable between FOVs. No smoothing or background subtraction was added. Scale bars are pixels because physical pixel size has not been verified.</p>
<script>function show(){let id=document.getElementById('case').value;let s=document.getElementById('context').checked?'_motif_context':document.getElementById('outlines').checked?'':'_no_outlines';document.getElementById('image').src=id+s+'.png'}function move(n){let s=document.getElementById('case');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+n));show()}show()</script>''')
    (args.output/'README.md').write_text('''# Motif-associated unidentified-cell microscopy examples

Displays use actual existing TIFF data, not generated or reconstructed microscopy. Presentation is inspired by [CellTune figure 5f/k](https://www.nature.com/articles/s41592-026-03162-2): black-background marker composites, individual channels, and thin segmentation contours.

Eight unflagged examples were sampled with seed 42069, two within each NN/NP × cluster-8/other-cluster stratum. They must be low evidence at the 90th percentile (and therefore unflagged at 95th/99th), have valid geometry, one bright DNA component, and mask area 120–650 pixels. Four CD56-flagged comparison examples were sampled, two per group, with the same geometry/size/DNA criteria. Distinct patients are preferred. Selection occurs before inspecting images and must not be interpreted as representative sampling of all 485 motif cells. Cluster IDs are from the unique-cell clustering, not the old notebook.

Each cell is checked against motif hit cell4 and its mask centroid. When it belongs to several hits, the context panel uses the instance with the shortest sum of three centroid-edge lengths, purely to keep the neighborhood compact. Context annotations are the existing labels, not independently verified cell identities. U=Unidentified; M=Memory CD4 T; C=CD8 T; N=NK.

Source: `/groups/assafza_group/assafza/Datasets/Melanoma/raw_data/FOV*.zip`; segmentation `TIFs/segmentation_labels.tiff`; channels `TIFs/<marker>.tif`. Display is linear, normalized to each entire FOV/channel's positive-pixel 99.5th percentile and clipped at that upper display limit. No smoothing, denoising or background removal was added. The unoutlined views preserve the same display. Comparison of signal location within an image is appropriate; intensity comparisons across FOVs require care. Scale bars are in pixels; TIFF metadata did not establish a calibrated micrometer scale. RGB merges add the explicitly named channel colors; white dashed contours show only the focal-cell boundary. All raw pixels remain in the original archives.

Open `index.html` to toggle focal outlines and inspect full motif context. `all_examples.pdf` contains the 12 individual sheets. The overview PNG/PDF pairs separate unflagged and suspect examples. `example_manifest.csv` records exact source cells, motif partners, strata, and display limits.

These figures support visual assessment of whether apparent marker expression belongs to the focal cell or its neighbor. A negative computational screen or selected image does not establish absence of spillover, segmentation errors, or annotation errors. No cell labels were changed and no manual review verdict was assigned by this export.

Reproduce: `MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=1 python analysis/marker_ownership/export_spillover_microscopy_panels.py`.
''')
    (args.output/'provenance.json').write_text(json.dumps(dict(seed=42069,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        examples=len(manifest), reference='https://www.nature.com/articles/s41592-026-03162-2',
        source_archive_stats={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in {Path(r['archive_path']) for r in metadata}}), indent=2))
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__': main()
