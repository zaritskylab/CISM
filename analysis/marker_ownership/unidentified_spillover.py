"""Pixel-based spillover-risk assessment; no automatic reannotation.

Run: MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=2 python analysis/marker_ownership/unidentified_spillover.py
The raw archives are read in place. Outputs are resumable per patient/FOV.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import argparse
import hashlib
import json
import time
import base64
import html
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = _REPO_ROOT
KEY = ['patient number', 'fov', 'cell_id']
FEATURES = ['CD11c', 'CD14', 'CD163', 'CD20', 'CD206', 'CD21', 'CD3', 'CD31', 'CD4', 'CD45', 'CD45RA', 'CD45RO', 'CD56', 'CD68', 'CD69', 'CD8', 'COL1A1', 'DC-SIGN', 'Foxp3', 'Granzyme B', 'HLA-DR-DP-DQ', 'HLA-class-1-A-B-C', 'IDO-1', 'Ki67', 'LAG-3', 'MECA-79', 'MelanA', 'PD-1', 'S100A9-Calprotectin', 'SMA', 'SOX10', 'TCF1TCF7', 'TIM-3', 'Tox-Tox2', 'anti-Biotin', 'dsDNA']
NUCLEAR = {'Foxp3', 'Ki67', 'SOX10', 'TCF1TCF7', 'Tox-Tox2', 'dsDNA'}
CYTOPLASMIC = {'CD68', 'Granzyme B', 'IDO-1', 'MelanA', 'S100A9-Calprotectin'}
OTHER = {'COL1A1', 'SMA', 'anti-Biotin'}
QUANTILES = [.90, .95, .99]
METRICS = ['boundary_log2_ratio', 'interface_log2_asymmetry', 'donor_log2_ratio']
TYPES = {
    'CD3': ['cd4 t cell', 'memory%cd4 t cell', 'cd8 t cell', 'treg'],
    'CD4': ['cd4 t cell', 'memory%cd4 t cell', 'treg'],
    'CD8': ['cd8 t cell'], 'CD45RO': ['memory%cd4 t cell'],
    'CD56': ['nk cell'], 'CD20': ['b cell'],
    'CD68': ['mac'], 'CD163': ['mac'], 'CD206': ['mac'],
    'CD31': ['blood vessels', 'hevs'], 'SOX10': ['tumor'], 'MelanA': ['tumor'],
    'CD45': ['cd4 t cell', 'memory%cd4 t cell', 'cd8 t cell', 'treg', 'nk cell', 'b cell', 'mac', 'DC'],
}
VERSION = '1.0'


def load_cohort(data_dir, hits):
    cells = pd.read_csv(data_dir/'mapped_cell_types.csv')
    cells = cells[cells.Group.isin(['NN', 'NP'])].copy()
    if cells.duplicated(KEY).any():
        raise ValueError('Spatial keys must be unique')
    h = pd.read_csv(hits).rename(columns={'patient': 'patient number', 'FOV': 'fov', 'cell4': 'cell_id'})
    counts = h.groupby(KEY).size().rename('motif_occurrences').reset_index()
    cells = cells.merge(counts, on=KEY, how='left', validate='one_to_one')
    cells['motif_occurrences'] = cells.motif_occurrences.fillna(0).astype(int)
    if not cells.loc[cells.motif_occurrences.gt(0), 'pred_mapped'].eq('unidentified').all():
        raise ValueError('Motif cell4 must be unidentified')
    cells['in_motif'] = cells.motif_occurrences.gt(0)
    return cells


def read_image(archive, name):
    # BytesIO avoids extracting absolute or traversal paths from third-party zips.
    with Image.open(BytesIO(archive.read(name))) as im:
        return np.array(im)


def erode_labels(labels):
    lo = ndi.minimum_filter(labels, size=3, mode='constant', cval=0)
    hi = ndi.maximum_filter(labels, size=3, mode='constant', cval=0)
    return np.where((labels == lo) & (labels == hi), labels, 0)


def geometry(labels, focal_ids, radius=3):
    """All focal boundary-pixel/donor pairs within a disk, deduplicated per pixel.

    Interfaces can overlap when a pixel faces multiple cells. Each *pair's*
    interface is deduplicated; it is subtracted only from that focal boundary.
    """
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError('Expected nonnegative integer 2-D segmentation labels')
    n = int(labels.max()) + 1
    erode1 = erode_labels(labels)
    erode2 = erode_labels(erode1)
    boundary = np.where((labels > 0) & (erode2 == 0), labels, 0)
    focal_lookup = np.zeros(n, bool)
    focal_lookup[np.asarray(focal_ids)[np.asarray(focal_ids) < n]] = True
    eligible = focal_lookup[boundary] & (boundary > 0)
    height, width = labels.shape
    pixel_grid = np.arange(labels.size).reshape(labels.shape)
    chunks = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if not 0 < dx*dx+dy*dy <= radius*radius:
                continue
            fy = slice(max(0, -dy), min(height, height-dy))
            fx = slice(max(0, -dx), min(width, width-dx))
            other = labels[slice(max(0, dy), min(height, height+dy)),
                           slice(max(0, dx), min(width, width+dx))]
            own = labels[fy, fx]
            keep = eligible[fy, fx] & (other > 0) & (other != own)
            chunks.append(pixel_grid[fy, fx][keep].astype(np.int64)*n + other[keep])
    packed = np.unique(np.concatenate(chunks)) if chunks else np.array([], np.int64)
    pixels, donor = packed//n, packed % n
    pair_codes, inverse = np.unique(labels.ravel()[pixels].astype(np.int64)*n+donor, return_inverse=True)
    focal, donor_ids = pair_codes//n, pair_codes % n
    counts = np.bincount(inverse, minlength=len(pair_codes))
    return dict(labels=labels, n=n, erode1=erode1, erode2=erode2, boundary=boundary,
                pixels=pixels, inverse=inverse, pair_focal=focal, pair_donor=donor_ids,
                interface_pixels=counts)


def sums(image, labels, n):
    return np.bincount(labels.ravel(), weights=image.ravel(), minlength=n)


def mean_by_label(image, labels, n):
    counts = np.bincount(labels.ravel(), minlength=n)
    return np.divide(sums(image, labels, n), counts, out=np.full(n, np.nan), where=counts > 0)


def extract_marker(g, image, focal_ids):
    """Measure all candidate interfaces, then retain the brightest donor per marker."""
    if image.shape != g['labels'].shape or not np.isfinite(image).all() or image.min() < 0:
        raise ValueError('Channel must match the mask and contain finite nonnegative pixels')
    n = g['n']
    whole = mean_by_label(image, g['labels'], n)
    inner1 = mean_by_label(image, g['erode1'], n)
    inner2 = mean_by_label(image, g['erode2'], n)
    bsum = sums(image, g['boundary'], n)
    bcount = np.bincount(g['boundary'].ravel(), minlength=n)
    bmean = np.divide(bsum, bcount, out=np.full(n, np.nan), where=bcount > 0)
    positive = whole[1:][whole[1:] > 0]
    eps = max(float(np.median(positive))*.01, 1e-6) if len(positive) else 1e-6
    f, d = g['pair_focal'], g['pair_donor']
    icount = g['interface_pixels']
    isum = np.bincount(g['inverse'], weights=image.ravel()[g['pixels']], minlength=len(f))
    imean = np.divide(isum, icount, out=np.full(len(f), np.nan), where=icount > 0)
    ocount = bcount[f]-icount
    omean = np.divide(bsum[f]-isum, ocount, out=np.full(len(f), np.nan), where=ocount > 0)
    interfaces = pd.DataFrame({'cell_id': f, 'donor_cell_id': d, 'interface_pixels': icount,
                               'other_boundary_pixels': ocount, 'interface_sum': isum,
                               'interface_mean': imean, 'other_boundary_mean': omean,
                               'donor_mean': whole[d]})
    viable = interfaces[(interfaces.interface_pixels >= 3) & (interfaces.other_boundary_pixels >= 3)]
    # Geometry-valid brightest donor; deterministic donor-ID tie break.
    best = viable.sort_values(['cell_id', 'donor_mean', 'donor_cell_id'], ascending=[True, False, True]).drop_duplicates('cell_id')
    result = pd.DataFrame({'cell_id': focal_ids})
    for name, values in [('whole_mean', whole), ('erode1_mean', inner1), ('erode2_mean', inner2), ('boundary_mean', bmean)]:
        result[name] = [values[i] if i < n else np.nan for i in focal_ids]
    result = result.merge(best, on='cell_id', how='left', validate='one_to_one')
    result['pseudocount'] = eps
    result['boundary_log2_ratio'] = np.log2((result.boundary_mean+eps)/(result.erode2_mean+eps))
    result['interface_log2_asymmetry'] = np.log2((result.interface_mean+eps)/(result.other_boundary_mean+eps))
    result['donor_log2_ratio'] = np.log2((result.donor_mean+eps)/(result.whole_mean+eps))
    result['erosion_log2_loss'] = np.log2((result.whole_mean+eps)/(result.erode2_mean+eps))
    result['basic_signal_evidence'] = (result.interface_sum >= 5) & (result.donor_log2_ratio >= 1) & (result.interface_log2_asymmetry >= np.log2(1.5)) & (result.boundary_log2_ratio >= np.log2(1.5))
    return result, interfaces, whole, inner2


def score_marker(frame, cell_info, k=50, minimum=30):
    """Same-FOV non-motif controls matched on log area and 50-pixel crowding.

    The control itself is excluded from its own reference neighborhood.
    Calibration uses geometry-valid cells, not just those with favorable scores.
    """
    frame = frame.merge(cell_info[['cell_id', 'mask_area', 'crowding_50px', 'geometry_valid', 'in_motif']], on='cell_id', validate='one_to_one')
    frame['marker_eligible'] = frame.geometry_valid & np.isfinite(frame[METRICS]).all(axis=1)
    refs = frame.loc[frame.marker_eligible & ~frame.in_motif]
    frame['reference_n'] = 0
    for q in QUANTILES:
        frame[f'flag_q{int(q*100)}'] = pd.Series(pd.NA, index=frame.index, dtype='boolean')
    if len(refs) < minimum:
        return frame
    coords = np.log1p(frame[['mask_area', 'crowding_50px']].to_numpy(float))
    ref_coords = coords[refs.index]
    scale = np.std(ref_coords, axis=0)
    scale[scale == 0] = 1
    nn = NearestNeighbors(n_neighbors=min(k+1, len(refs))).fit(ref_coords/scale)
    neighbors = nn.kneighbors(coords/scale, return_distance=False)
    ref_indices = refs.index.to_numpy()
    matches = []
    for i, ns in enumerate(neighbors):
        selected = ref_indices[ns]
        selected = selected[selected != i][:k]
        matches.append(selected)
    reference_n = np.array([len(ids) for ids in matches])
    width = int(reference_n.max())
    index_matrix = np.zeros((len(frame), width), dtype=int)
    padding = np.ones((len(frame), width), dtype=bool)
    for i, ids in enumerate(matches):
        index_matrix[i, :len(ids)] = ids
        padding[i, :len(ids)] = False
    available = frame.marker_eligible.to_numpy() & (reference_n >= minimum)
    frame['reference_n'] = np.where(available, reference_n, 0)
    for metric in METRICS:
        values = frame[metric].to_numpy()
        matched = values[index_matrix].copy()
        matched[padding] = np.nan
        less = np.sum(matched < values[:, None], axis=1)
        equal = np.sum(matched == values[:, None], axis=1)
        percentiles = (less + .5*equal)/reference_n
        percentiles[~available] = np.nan
        frame[metric+'_reference_percentile'] = percentiles
        thresholds = np.nanquantile(matched, QUANTILES, axis=1)
        for j, q in enumerate(QUANTILES):
            thresholds[j, ~available] = np.nan
            frame[f'{metric}_q{int(q*100)}'] = thresholds[j]
    frame['joint_reference_percentile'] = frame[[x+'_reference_percentile' for x in METRICS]].min(axis=1, skipna=False)
    available = frame.marker_eligible & frame.reference_n.ge(minimum)
    for q in QUANTILES:
        exceeds = np.logical_and.reduce([frame[m].gt(frame[f'{m}_q{int(q*100)}']).to_numpy() for m in METRICS])
        frame.loc[available, f'flag_q{int(q*100)}'] = (frame.basic_signal_evidence & exceeds)[available]
    return frame


def cell_geometry(g, frame):
    n, mask = g['n'], g['labels']
    ids = frame.cell_id.to_numpy(int)
    present = ids < n
    area = np.bincount(mask.ravel(), minlength=n)
    present[present] &= area[ids[present]] > 0
    rows, cols = np.indices(mask.shape)
    rowmean, colmean = mean_by_label(rows, mask, n), mean_by_label(cols, mask, n)
    result = frame[['cell_id', 'in_motif', 'motif_occurrences', 'Group', 'patient number', 'fov', 'centroid-0', 'centroid-1']].copy().reset_index(drop=True)
    for name, values in [('mask_area', area), ('interior1_pixels', np.bincount(g['erode1'].ravel(), minlength=n)), ('interior2_pixels', np.bincount(g['erode2'].ravel(), minlength=n))]:
        result[name] = [int(values[i]) if i < n else 0 for i in ids]
    result['mask_present'] = present
    result['centroid_error_px'] = [float(np.hypot(rowmean[i]-r, colmean[i]-c)) if ok else np.nan for i, r, c, ok in zip(ids, frame['centroid-0'], frame['centroid-1'], present)]
    edge_ids = np.unique(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]))
    result['edge_clipped'] = np.isin(ids, edge_ids)
    result['inadequate_interior'] = result.interior2_pixels < 5
    result['geometry_valid'] = result.mask_present & result.centroid_error_px.le(.75) & ~result.edge_clipped & ~result.inadequate_interior & result.mask_area.ge(20)
    result['crowding_50px'] = cKDTree(frame[['centroid-0', 'centroid-1']].to_numpy()).query_ball_point(frame[['centroid-0', 'centroid-1']].to_numpy(), 50, return_length=True)-1
    return result


def nuclear_qc(image, mask, ids):
    """Screen for multiple separated bright DNA components; not a doublet diagnosis."""
    smooth = ndi.gaussian_filter(image.astype(float), 1)
    positive = smooth[(mask > 0) & (smooth > 0)]
    threshold = np.quantile(positive, .75) if len(positive) else np.inf
    objects = ndi.find_objects(mask)
    counts = []
    for i in ids:
        sl = objects[i-1] if 0 < i <= len(objects) else None
        if sl is None:
            counts.append(np.nan)
            continue
        own = mask[sl] == i
        components, _ = ndi.label((smooth[sl] >= threshold) & own)
        areas = np.bincount(components.ravel())[1:]
        large = areas[areas >= 5]
        counts.append(int(np.count_nonzero(large >= .15*large.max())) if len(large) else 0)
    return np.asarray(counts)


def process_fov(frame, args, destination):
    patient, fov = frame.iloc[0][['patient number', 'fov']]
    focal = frame[frame.pred_mapped.eq('unidentified')].copy()
    if focal.empty:
        return
    archive_path = args.data_dir/'raw_data'/f'{fov}.zip'
    provenance = {'version': VERSION, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'patient': int(patient), 'fov': fov,
                  'archive': str(archive_path), 'archive_size': archive_path.stat().st_size if archive_path.exists() else None,
                  'archive_mtime_ns': archive_path.stat().st_mtime_ns if archive_path.exists() else None,
                  'cohort_hash': hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()}
    meta = destination/'complete.json'
    if meta.exists() and json.loads(meta.read_text()) == provenance and not args.force:
        print(f'{fov}: using completed measurements', flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        missing = focal.copy(); missing['fov_failure'] = 'missing_archive'
        missing.to_csv(destination/'cells.csv.gz', index=False)
        meta.write_text(json.dumps(provenance, indent=2)); return
    started = time.time()
    with ZipFile(archive_path) as archive:
        name = next((n for n in archive.namelist() if Path(n).name in ['segmentation_labels.tiff', 'segmentation_labels.tif']), None)
        if name is None:
            missing = focal.copy(); missing['fov_failure'] = 'missing_mask'
            missing.to_csv(destination/'cells.csv.gz', index=False)
            meta.write_text(json.dumps(provenance, indent=2)); return
        labels = read_image(archive, name).astype(np.int32)
        g = geometry(labels, focal.cell_id.to_numpy(int))
        info_all = cell_geometry(g, frame)
        info = info_all[info_all.cell_id.isin(focal.cell_id)].reset_index(drop=True)
        predictions = []
        interface_path = destination/'interfaces.csv.gz'
        if interface_path.exists(): interface_path.unlink()
        for j, marker in enumerate(FEATURES):
            filename = f'TIFs/{marker}.tif'
            if filename not in archive.namelist():
                absent = pd.DataFrame({'cell_id': focal.cell_id.to_numpy(), 'marker': marker, 'marker_failure': 'missing_channel'})
                predictions.append(absent); continue
            image = read_image(archive, filename)
            if image.shape != labels.shape:
                absent = pd.DataFrame({'cell_id': focal.cell_id.to_numpy(), 'marker': marker, 'marker_failure': 'dimension_mismatch'})
                predictions.append(absent); continue
            measured, interfaces, whole, inner = extract_marker(g, image, focal.cell_id.to_numpy(int))
            measured = score_marker(measured, info)
            measured['marker'] = marker
            measured['compartment'] = 'nuclear' if marker in NUCLEAR else 'cytoplasmic' if marker in CYTOPLASMIC else 'other' if marker in OTHER else 'membrane_associated'
            measured['marker_failure'] = ''
            interfaces['marker'] = marker
            donor_types = frame.set_index('cell_id').pred_mapped
            interfaces['donor_annotation'] = interfaces.donor_cell_id.map(donor_types).fillna('not_in_spatial_table')
            interfaces.to_csv(interface_path, index=False, mode='at', header=not interface_path.exists(), compression='gzip')
            measured['donor_annotation'] = measured.donor_cell_id.map(donor_types)
            if marker in TYPES:
                positive_ids = frame.loc[frame.pred_mapped.isin(TYPES[marker]) & frame.cell_id.lt(g['n']), 'cell_id'].to_numpy(int)
                positive_ids = positive_ids[info_all.set_index('cell_id').loc[positive_ids, 'geometry_valid'].to_numpy(bool)]
                measured['lineage_reference_n'] = len(positive_ids)
                if len(positive_ids) >= 10:
                    # Whole and interior each calibrated to the matching compartment.
                    for label, values, observed in [('whole', whole, 'whole_mean'), ('interior', inner, 'erode2_mean')]:
                        threshold = max(float(np.nanquantile(values[positive_ids], .10)), measured.pseudocount.iloc[0]*5)
                        measured[f'lineage_{label}_threshold'] = threshold
                        measured[f'lineage_{label}_positive'] = measured[observed].ge(threshold)
            if marker == 'dsDNA':
                info['bright_dna_components'] = nuclear_qc(image, labels, focal.cell_id.to_numpy(int))
                info['multiple_dna_components'] = info.bright_dna_components.ge(2)
            predictions.append(measured)
            if (j+1) % 12 == 0: print(f'{fov}: {j+1}/36 channels', flush=True)
        metrics = pd.concat(predictions, ignore_index=True)
        metrics.to_csv(destination/'markers.csv.gz', index=False)
        info.to_csv(destination/'cells.csv.gz', index=False)
    meta.write_text(json.dumps(provenance, indent=2))
    print(f'{fov}: {len(info)} unidentified cells, {time.time()-started:.1f}s', flush=True)


def summarize_cells(markers, info):
    out = info.set_index('cell_id').copy()
    diagnostic = markers[~markers.marker.eq('dsDNA')]
    valid = diagnostic.flag_q95.notna()
    out['scored_markers'] = diagnostic[valid].groupby('cell_id').size().reindex(out.index, fill_value=0)
    out['missing_or_unscored_markers'] = 35-out.scored_markers
    for q in [90, 95, 99]:
        col = f'flag_q{q}'
        flags = diagnostic[col].astype('boolean').fillna(False).astype(bool)
        out[f'flagged_markers_q{q}'] = diagnostic[flags].groupby('cell_id').marker.agg(';'.join).reindex(out.index, fill_value='')
        out[f'any_flag_q{q}'] = out[f'flagged_markers_q{q}'].ne('')
        out[f'status_q{q}'] = np.where(out[f'any_flag_q{q}'], 'spillover_consistent', np.where(out.scored_markers.eq(35), 'low_evidence', 'unresolved'))
    # Multi-component DNA is an independent segmentation concern, not proof of spillover.
    if 'multiple_dna_components' in out:
        for q in [90, 95, 99]:
            out.loc[out.multiple_dna_components.fillna(False) & ~out[f'any_flag_q{q}'], f'status_q{q}'] = 'unresolved'
    patterns = {'T_lineage': [('CD3', 'CD45')], 'B_lineage': [('CD20', 'CD45')],
                'myeloid': [('CD68', 'CD163'), ('CD68', 'CD206')],
                'NK_compatible': [('CD56', 'CD45')], 'tumor_compatible': [('SOX10', 'MelanA')]}
    for compartment in ['whole', 'interior']:
        col = f'lineage_{compartment}_positive'
        if col not in markers: continue
        positive = markers.pivot(index='cell_id', columns='marker', values=col).reindex(out.index)
        support = pd.DataFrame(index=out.index)
        for name, alternatives in patterns.items():
            checks = [positive.reindex(columns=list(pair)).eq(True).all(axis=1) for pair in alternatives]
            support[name] = np.logical_or.reduce(checks)
        # CD56 alone is not NK-specific; T-lineage support takes precedence.
        support['NK_compatible'] &= ~support.T_lineage
        out[f'{compartment}_lineage_patterns'] = support.apply(lambda r: ';'.join(r.index[r]), axis=1)
        out[f'{compartment}_mixed_lineage'] = support.sum(axis=1) >= 2
    if 'interior_lineage_patterns' in out:
        out['possible_reannotation_candidate'] = out.interior_lineage_patterns.ne('') & out.geometry_valid
        out['mixed_profile_resolves_with_boundary_exclusion'] = out.whole_mixed_lineage & ~out.interior_mixed_lineage
    return out.reset_index()


def bootstrap_summary(cells, iterations=2000, seed=42069):
    rng = np.random.default_rng(seed)
    patients = cells.groupby(['Group', 'patient number', 'in_motif']).agg(
        cells=('cell_id', 'size'), scored_cells=('scored_markers', lambda s: int((s > 0).sum())),
        spillover_consistent=('status_q95', lambda s: int(s.eq('spillover_consistent').sum())),
        low_evidence=('status_q95', lambda s: int(s.eq('low_evidence').sum())),
        unresolved=('status_q95', lambda s: int(s.eq('unresolved').sum()))).reset_index()
    patients['flagged_fraction_all_cells'] = patients.spillover_consistent/patients.cells
    rows = []
    for name in ['pooled', 'NN', 'NP']:
        sub = patients if name == 'pooled' else patients[patients.Group.eq(name)]
        for motif in [True, False]:
            a = sub.loc[sub.in_motif.eq(motif), 'flagged_fraction_all_cells'].to_numpy()
            if not len(a): continue
            draws = rng.choice(a, size=(iterations, len(a)), replace=True).mean(axis=1)
            rows.append(dict(group=name, comparison='motif' if motif else 'control', patients=len(a), estimate=a.mean(), ci_low=np.quantile(draws, .025), ci_high=np.quantile(draws, .975)))
        paired = sub.pivot(index='patient number', columns='in_motif', values='flagged_fraction_all_cells').reindex(columns=[False, True]).dropna()
        delta = (paired[True]-paired[False]).to_numpy()
        if len(delta):
            draws = rng.choice(delta, size=(iterations, len(delta)), replace=True).mean(axis=1)
            rows.append(dict(group=name, comparison='paired_motif_minus_control', patients=len(delta), estimate=delta.mean(), ci_low=np.quantile(draws, .025), ci_high=np.quantile(draws, .975)))
    return patients, pd.DataFrame(rows)


def make_plots(cells, summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    palette = ['#b34f65', '#6eaa9b', '#afb7c0']
    for ax, group in zip(axes, ['pooled', 'NN', 'NP']):
        sub = cells if group == 'pooled' else cells[cells.Group.eq(group)]
        counts = pd.crosstab(sub.in_motif, sub.status_q95).reindex(index=[False, True], columns=['spillover_consistent', 'low_evidence', 'unresolved'], fill_value=0)
        prop = counts.div(counts.sum(axis=1), axis=0)*100
        bottom = np.zeros(2)
        for color, status in zip(palette, counts.columns):
            ax.bar([0, 1], prop[status], bottom=bottom, color=color, label=status.replace('_', ' '))
            bottom += prop[status].fillna(0).to_numpy()
        ax.set_xticks([0, 1], [f'Controls\nn={counts.loc[False].sum():,}', f'Motif\nn={counts.loc[True].sum():,}'])
        ax.set_title(group); ax.set_ylim(0, 100); ax.set_ylabel('Unique unidentified cells (%)')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    fig.suptitle('Computational spillover-risk assessment · 95th-percentile screening')
    fig.tight_layout(); fig.savefig(output/'risk_summary.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, group in zip(axes, ['pooled', 'NN', 'NP']):
        sub = cells if group == 'pooled' else cells[cells.Group.eq(group)]
        for i, (motif, color) in enumerate([(False, '#758b9b'), (True, '#9672bb')]):
            p = sub.groupby(['patient number', 'in_motif']).status_q95.apply(lambda s: s.eq('spillover_consistent').mean()).reset_index(name='fraction')
            y = p.loc[p.in_motif.eq(motif), 'fraction'].to_numpy()*100
            ax.scatter(np.full(len(y), i)+np.linspace(-.12, .12, len(y)), y, color=color, alpha=.6, s=18)
            row = summary[(summary.group == group) & (summary.comparison == ('motif' if motif else 'control'))]
            if len(row):
                r = row.iloc[0]
                ax.errorbar(i, r.estimate*100, yerr=np.array([[r.estimate-r.ci_low], [r.ci_high-r.estimate]])*100, fmt='D', color='black', capsize=4)
        ax.set_xticks([0, 1], ['Controls', 'Motif']); ax.set_title(group)
    axes[0].set_ylabel('Flagged cells per patient (%)')
    fig.suptitle('Patient distributions · diamonds: equal-patient means and bootstrap 95% intervals')
    fig.tight_layout(); fig.savefig(output/'patient_uncertainty.png', dpi=180); plt.close(fig)
    sensitivity = []
    for q in [90, 95, 99]:
        for (group, motif), sub in cells.groupby(['Group', 'in_motif']):
            sensitivity.append(dict(quantile=q, Group=group, in_motif=motif, cells=len(sub), flagged=int(sub[f'any_flag_q{q}'].sum()), unresolved=int(sub[f'status_q{q}'].eq('unresolved').sum())))
    pd.DataFrame(sensitivity).to_csv(output/'threshold_sensitivity.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (group, motif), sub in pd.DataFrame(sensitivity).groupby(['Group', 'in_motif']):
        ax.plot(sub['quantile'], 100*sub.flagged/sub.cells, marker='o', label=f'{group}: '+('motif' if motif else 'controls'))
    ax.set(xlabel='Reference percentile', ylabel='Flagged unique cells (%)', title='Threshold sensitivity'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output/'threshold_sensitivity.png', dpi=180); plt.close(fig)


def make_review(cells, args):
    output = args.output/'blinded_review'
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(42069)
    valid = cells[cells.mask_present.astype('boolean').fillna(False) & cells.centroid_error_px.le(.75)]
    motif = valid[valid.in_motif].sample(n=min(50, int(valid.in_motif.sum())), random_state=42069)
    chosen = []; used = set()
    for _, row in motif.iterrows():
        controls = valid[~valid.in_motif & valid.fov.eq(row.fov) & valid['patient number'].eq(row['patient number'])]
        controls = controls[~controls.index.isin(used)]
        if len(controls):
            idx = int(rng.choice(controls.index)); used.add(idx); chosen.append(idx)
    controls = valid.loc[chosen]
    random_set = pd.concat([motif.assign(sampling='random_motif'), controls.assign(sampling='matched_control')])
    extra = valid[valid.in_motif & valid.any_flag_q95 & ~valid.index.isin(random_set.index)]
    extra = extra.sample(n=min(25, len(extra)), random_state=42070).assign(sampling='flagged_extra')
    selected = pd.concat([random_set, extra]).sample(frac=1, random_state=42071).reset_index(drop=True)
    selected['review_id'] = [f'R{i:03d}' for i in range(1, len(selected)+1)]
    selected.to_csv(output/'unblinding_key_DO_NOT_OPEN_DURING_REVIEW.csv', index=False)
    reviews = []
    for fov, sub in selected.groupby('fov'):
        with ZipFile(args.data_dir/'raw_data'/f'{fov}.zip') as archive:
            name = next(n for n in archive.namelist() if Path(n).name in ['segmentation_labels.tiff', 'segmentation_labels.tif'])
            mask = read_image(archive, name)
            # Same channel layout for every crop; adaptive channels would reveal the score.
            channels = ['dsDNA', 'CD3', 'CD4', 'CD45RO', 'CD8', 'CD56', 'CD20', 'CD68', 'CD163', 'CD45', 'SOX10', 'MelanA']
            images = {m: read_image(archive, f'TIFs/{m}.tif') for m in channels if f'TIFs/{m}.tif' in archive.namelist()}
            limits = {m: max(float(np.quantile(im[im > 0], .995)), 1) if (im > 0).any() else 1 for m, im in images.items()}
            for _, row in sub.iterrows():
                y, x = int(round(row['centroid-0'])), int(round(row['centroid-1']))
                sl = (slice(max(0, y-48), min(mask.shape[0], y+49)), slice(max(0, x-48), min(mask.shape[1], x+49)))
                local = mask[sl]
                outline = (local > 0) & (erode_labels(local) == 0)
                target = local == row.cell_id
                target_outline = target & ~ndi.binary_erosion(target)
                fig, axes = plt.subplots(3, 5, figsize=(12, 7.5))
                for ax, marker in zip(axes.ravel(), channels):
                    if marker in images:
                        ax.imshow(images[marker][sl], cmap='gray', vmin=0, vmax=limits[marker])
                    else: ax.text(.5, .5, 'Channel missing', ha='center')
                    ax.set_title(marker, fontsize=9)
                composite = np.stack([np.clip(images[m][sl]/limits[m], 0, 1) if m in images else np.zeros(local.shape) for m in ['CD3', 'CD56', 'dsDNA']], axis=-1)
                axes.ravel()[12].imshow(composite); axes.ravel()[12].set_title('CD3 / CD56 / DNA', fontsize=9)
                axes.ravel()[13].imshow(images['dsDNA'][sl], cmap='gray', vmin=0, vmax=limits['dsDNA'])
                axes.ravel()[13].set_title('Neighborhood + masks', fontsize=9)
                overlays = []
                for ax in axes.ravel()[:14]:
                    overlay = np.zeros(local.shape+(4,)); overlay[outline] = [0, 1, 1, .6]; overlay[target_outline] = [1, 1, 0, 1]
                    overlays.append(ax.imshow(overlay)); ax.axis('off')
                axes.ravel()[14].axis('off')
                axes.ravel()[14].text(0, .9, 'Yellow: focal cell\nCyan: other masks\n\nPixel coordinates only\nNo physical scale inferred', fontsize=9, va='top')
                fig.suptitle(row.review_id); fig.tight_layout()
                fig.savefig(output/f'{row.review_id}.png', dpi=130)
                for overlay in overlays: overlay.set_visible(False)
                fig.savefig(output/f'{row.review_id}_channels.png', dpi=130); plt.close(fig)
                reviews.append(dict(review_id=row.review_id, annotation_plausibility='', spillover='', segmentation_failure='', uncertainty='', notes=''))
    pd.DataFrame(reviews).sort_values('review_id').to_csv(output/'review_form.csv', index=False)
    (output/'INSTRUCTIONS.md').write_text('Open index.html to browse blinded cases and toggle segmentation outlines. Use review_form.csv to record judgments. Do not open the unblinding key until labels are complete.\n\nFor annotation_plausibility enter plausible / implausible / uncertain; for spillover and segmentation_failure enter yes / no / uncertain; for uncertainty enter low / medium / high. Record reasons in notes. Existing unidentified annotation applies to all focal cells.\n\nEach case has an outlined PNG and a _channels.png without outlines. Channel scaling is fixed within each FOV and channel; raw counts remain in the analysis tables. Inspect the channels without outlines when boundaries obscure signal. Analyze random cases separately from flagged_extra using the key after review. No manual conclusions have yet been recorded.\n')
    options = ''.join(f'<option>{i}</option>' for i in sorted(selected.review_id))
    (output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Blinded cell review</title>
<style>body{font:16px system-ui;margin:20px}header{position:sticky;top:0;background:white;padding:15px;border-bottom:1px solid #ccc}img{max-width:100%}button,select{font:inherit;margin:0 8px}</style>
<header><button onclick="move(-1)">Previous</button><select id="case" onchange="show()">'''+options+'''</select><button onclick="move(1)">Next</button><label><input type="checkbox" id="outlines" checked onchange="show()">Show segmentation outlines</label> · <a href="review_form.csv">Blank review form</a><p>Review annotation plausibility, spillover, segmentation and uncertainty. Groups and computational scores are hidden. This viewer does not save judgments; enter them in the CSV.</p></header><img id="image" alt="Blinded marker channels">
<script>function show(){document.getElementById('image').src=document.getElementById('case').value+(document.getElementById('outlines').checked?'':'_channels')+'.png'}function move(n){let s=document.getElementById('case');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+n));show()}show()</script>''')
    make_all_channel_review(selected, args)
    print(f'Prepared {len(selected)} blinded review crops', flush=True)


def make_all_channel_review(selected, args):
    """Fixed 36-channel atlas avoids revealing which marker triggered a flag."""
    output = args.output/'blinded_review'
    nearest = Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST
    for fov, frame in selected.groupby('fov'):
        with ZipFile(args.data_dir/'raw_data'/f'{fov}.zip') as archive:
            mask_name = next(n for n in archive.namelist() if Path(n).name in ['segmentation_labels.tiff', 'segmentation_labels.tif'])
            labels = read_image(archive, mask_name)
            panels = {row.review_id: [Image.new('RGB', (1230, 1380), '#111111') for _ in range(2)] for _, row in frame.iterrows()}
            for marker_index, marker in enumerate(FEATURES):
                filename = f'TIFs/{marker}.tif'
                image = read_image(archive, filename) if filename in archive.namelist() else None
                limit = max(float(np.quantile(image[image > 0], .995)), 1) if image is not None and (image > 0).any() else 1
                for _, row in frame.iterrows():
                    y, x = int(round(row['centroid-0'])), int(round(row['centroid-1']))
                    sl = (slice(max(0, y-48), min(labels.shape[0], y+49)), slice(max(0, x-48), min(labels.shape[1], x+49)))
                    local = labels[sl]
                    gray = (255*np.clip(image[sl]/limit, 0, 1)).astype(np.uint8) if image is not None else np.zeros(local.shape, np.uint8)
                    rgb = np.repeat(gray[:, :, None], 3, axis=2)
                    outlined = rgb.copy()
                    boundaries = (local > 0) & (erode_labels(local) == 0)
                    target = local == row.cell_id
                    outlined[boundaries] = (.4*outlined[boundaries]+.6*np.array([0, 255, 255])).astype(np.uint8)
                    outlined[target & ~ndi.binary_erosion(target)] = [255, 255, 0]
                    tx, ty = (marker_index % 6)*205, 40+(marker_index//6)*223
                    for panel, pixels in zip(panels[row.review_id], [rgb, outlined]):
                        panel.paste(Image.fromarray(pixels).resize((pixels.shape[1]*2, pixels.shape[0]*2), nearest), (tx+4, ty+22))
                        draw = ImageDraw.Draw(panel)
                        draw.text((tx+4, ty+4), marker+(' (missing)' if image is None else ''), fill='white')
            for review_id, pair in panels.items():
                for image, suffix in zip(pair, ['_all_channels', '_all_channels_masks']):
                    ImageDraw.Draw(image).text((8, 10), review_id+' | all 36 channels | raw display, 2x pixels', fill='white')
                    image.save(output/f'{review_id}{suffix}.png')
    options = ''.join(f'<option>{i}</option>' for i in sorted(selected.review_id))
    (output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Blinded cell review</title>
<style>body{font:16px system-ui;margin:20px}header{position:sticky;top:0;background:white;padding:15px;border-bottom:1px solid #ccc}img{max-width:100%}button,select{font:inherit;margin:0 8px}</style>
<header><button onclick="move(-1)">Previous</button><select id="case" onchange="show()">'''+options+'''</select><button onclick="move(1)">Next</button><select id="view" onchange="show()"><option value="overview">Lineage overview</option><option value="all">All 36 channels</option></select><label><input type="checkbox" id="outlines" checked onchange="show()">Show segmentation outlines</label> · <a href="review_form.csv">Blank review form</a><p>Groups and scores are hidden. Toggle outlines to inspect raw signal; the all-channel view includes markers absent from the overview. Record judgments in the CSV; this viewer does not save them.</p></header><img id="image" alt="Blinded marker channels">
<script>function show(){let o=document.getElementById('outlines').checked;let suffix=document.getElementById('view').value==='all'?(o?'_all_channels_masks':'_all_channels'):(o?'':'_channels');document.getElementById('image').src=document.getElementById('case').value+suffix+'.png'}function move(n){let s=document.getElementById('case');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+n));show()}show()</script>''')
    with (output/'INSTRUCTIONS.md').open('a') as handle:
        handle.write('\nThe viewer also offers all 36 measured channels in a fixed layout for every case, with or without outlines. This prevents the channel selection from disclosing computational flags.\n')


def aggregate(args, cohort):
    frames = []; total_marker_rows = 0
    combined_path = args.output/'per_cell_per_marker.csv.gz'
    if combined_path.exists(): combined_path.unlink()
    for directory in sorted((args.output/'fovs').iterdir()):
        info_path = directory/'cells.csv.gz'
        if not (directory/'complete.json').exists(): continue
        info = pd.read_csv(info_path)
        marker_path = directory/'markers.csv.gz'
        if not marker_path.exists() or 'fov_failure' in info:
            info['scored_markers'] = 0
            for q in [90, 95, 99]: info[f'status_q{q}'] = 'unresolved'; info[f'any_flag_q{q}'] = False
            frames.append(info); continue
        markers = pd.read_csv(marker_path, low_memory=False)
        for q in [90, 95, 99]: markers[f'flag_q{q}'] = markers[f'flag_q{q}'].astype('boolean')
        summarized = summarize_cells(markers, info)
        frames.append(summarized)
        for col in ['patient number', 'fov', 'Group']:
            markers[col] = info[col].iloc[0]
        markers.to_csv(combined_path, index=False, mode='at', header=not combined_path.exists(), compression='gzip')
        total_marker_rows += len(markers)
    cells = pd.concat(frames, ignore_index=True)
    # Join descriptive original classifier probabilities by the historic integer-coordinate key.
    path = args.data_dir/'cleaned_expression_with_both_classification_prob_spatial_27_09_23.csv'
    cols = ['patient number', 'fov', 'centroid-0', 'centroid-1', 'pred', 'pred_prob', 'score']
    protein = pd.read_csv(path, usecols=cols).rename(columns={'pred': 'protein_table_pred', 'score': 'protein_table_score'})
    join = ['patient number', 'fov', 'centroid-0', 'centroid-1']
    temp = cells[KEY+['centroid-0', 'centroid-1']].copy()
    temp[['centroid-0', 'centroid-1']] = np.trunc(temp[['centroid-0', 'centroid-1']])
    prob = temp.merge(protein, on=join, how='left', validate='many_to_one')
    cells = cells.merge(prob[KEY+['protein_table_pred', 'pred_prob', 'protein_table_score']], on=KEY, how='left', validate='one_to_one')
    cells.to_csv(args.output/'per_cell_summary.csv', index=False)
    patients, summary = bootstrap_summary(cells)
    patients.to_csv(args.output/'patient_summary.csv', index=False)
    summary.to_csv(args.output/'patient_bootstrap.csv', index=False)
    make_plots(cells, summary, args.output)
    coverage = cells.groupby(['Group', 'in_motif']).agg(cells=('cell_id', 'size'),
        matched_masks=('mask_present', 'sum'), edge_clipped=('edge_clipped', 'sum'), inadequate_interior=('inadequate_interior', 'sum'),
        geometry_valid=('geometry_valid', 'sum'), multiple_dna_components=('multiple_dna_components', 'sum'),
        all_markers_scored=('scored_markers', lambda s: int(s.eq(35).sum())))
    coverage.to_csv(args.output/'coverage.csv')
    pd.crosstab([cells.Group, cells.in_motif], cells.status_q95).to_csv(args.output/'status_counts.csv')
    config = dict(version=VERSION, motif=20748339660, unique_motif_cells=int(cells.in_motif.sum()), motif_occurrences=int(cells.loc[cells.in_motif, 'motif_occurrences'].sum()),
                  unidentified_cells=len(cells), marker_rows=total_marker_rows, features=FEATURES,
                  quantiles=QUANTILES, seed=42069, bootstrap_iterations=2000,
                  donor_radius_px=3, erosion_px=[1, 2], same_fov_reference_neighbors=50,
                  minimum_reference_cells=30, centroid_tolerance_px=.75, min_area_px=20,
                  min_interior_px=5, min_interface_px=3, min_other_boundary_px=3,
                  minimum_interface_sum=5, minimum_donor_fold=2, minimum_localization_fold=1.5,
                  partial_run=args.max_fovs is not None, manually_reviewed=False)
    (args.output/'run_metadata.json').write_text(json.dumps(config, indent=2))
    write_report(cells, summary, coverage, args.output)
    if not args.skip_review: make_review(cells, args)


def write_report(cells, summary, coverage, output):
    statuses = pd.crosstab(cells.in_motif, cells.status_q95).reindex(index=[False, True], columns=['spillover_consistent', 'low_evidence', 'unresolved'], fill_value=0)
    overview = statuses.copy()
    overview['total'] = overview.sum(axis=1)
    for col in statuses:
        overview[col+'_percent'] = 100*overview[col]/overview.total
    overview.index = ['Other unidentified cells', 'Motif-associated unidentified cells']
    overview.to_csv(output/'population_summary.csv')
    marker_parts = []; donor_parts = []
    use = ['patient number', 'fov', 'Group', 'in_motif', 'marker', 'compartment', 'flag_q95', 'donor_annotation']
    for chunk in pd.read_csv(output/'per_cell_per_marker.csv.gz', usecols=use, chunksize=100000, low_memory=False):
        chunk['flag_q95'] = chunk.flag_q95.astype('boolean')
        part = chunk.groupby(['Group', 'in_motif', 'marker', 'compartment'], dropna=False).agg(
            measured_cells=('marker', 'size'), scored_cells=('flag_q95', 'count'), flagged_cells=('flag_q95', 'sum')).reset_index()
        marker_parts.append(part)
        donor_parts.append(chunk[chunk.flag_q95.fillna(False)].groupby(['Group', 'in_motif', 'marker', 'donor_annotation'], dropna=False).size().rename('flagged_cells').reset_index())
    marker_summary = pd.concat(marker_parts).groupby(['Group', 'in_motif', 'marker', 'compartment'], dropna=False)[['measured_cells', 'scored_cells', 'flagged_cells']].sum().reset_index()
    marker_summary['flagged_fraction_scored'] = marker_summary.flagged_cells/marker_summary.scored_cells.replace(0, np.nan)
    marker_summary.to_csv(output/'marker_risk_summary.csv', index=False)
    donors = pd.concat(donor_parts).groupby(['Group', 'in_motif', 'marker', 'donor_annotation'], dropna=False).flagged_cells.sum().reset_index()
    donors.to_csv(output/'marker_donor_summary.csv', index=False)
    pooled = marker_summary[marker_summary.marker.ne('dsDNA')].groupby(['in_motif', 'marker'])[['scored_cells', 'flagged_cells']].sum().reset_index()
    pooled['fraction'] = pooled.flagged_cells/pooled.scored_cells.replace(0, np.nan)
    rates = pooled.pivot(index='marker', columns='in_motif', values='fraction').reindex(columns=[False, True])
    top = rates.sort_values(True, ascending=False).head(12).index
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(top))
    ax.bar(x-.19, rates.loc[top, False]*100, .38, label='Other unidentified cells', color='#9cabb6')
    ax.bar(x+.19, rates.loc[top, True]*100, .38, label='Motif-associated unidentified cells', color='#9672bb')
    ax.set_xticks(x, top, rotation=45, ha='right'); ax.set_ylabel('Flagged / scored cells (%)')
    ax.set_title('Markers with the highest motif-cell flag rates (95th-percentile screen)')
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(output/'marker_risk.png', dpi=180); plt.close(fig)
    # Additional uncertainty estimate restricted to outcome-independent complete cases.
    complete = cells.scored_markers.eq(35) & ~cells.multiple_dna_components.astype('boolean').fillna(True)
    complete_cells = cells.loc[complete].copy()
    complete_patients, complete_summary = bootstrap_summary(complete_cells)
    complete_patients.to_csv(output/'patient_summary_complete_cases.csv', index=False)
    complete_summary.to_csv(output/'patient_bootstrap_complete_cases.csv', index=False)
    lineage = cells.groupby(['Group', 'in_motif']).agg(
        cells=('cell_id', 'size'), mixed_whole=('whole_mixed_lineage', lambda s: int(s.eq(True).sum())),
        mixed_interior=('interior_mixed_lineage', lambda s: int(s.eq(True).sum())),
        possible_reannotation=('possible_reannotation_candidate', lambda s: int(s.eq(True).sum())),
        mixed_resolves=('mixed_profile_resolves_with_boundary_exclusion', lambda s: int(s.eq(True).sum())))
    lineage.to_csv(output/'lineage_compatibility_summary.csv')
    failure = cells.get('fov_failure', pd.Series('', index=cells.index)).fillna('')
    missing = cells.loc[failure.ne(''), KEY+['Group', 'in_motif']].assign(fov_failure=failure[failure.ne('')])
    missing.to_csv(output/'unmeasured_cells.csv', index=False)
    manifest = dict(implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    data_files={p.name: {'size_bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                                for p in [output/'per_cell_per_marker.csv.gz', output/'per_cell_summary.csv']})
    (output/'report_provenance.json').write_text(json.dumps(manifest, indent=2))
    lines = ['# Computational spillover-risk assessment', '',
        'Motif 20748339660; unique NN/NP unidentified cells; no protein-cluster selection. This analysis does not validate annotation accuracy or rule out spillover. No manual review has been performed.', '',
        'Population summary:', '', '```', overview.round(3).to_string(), '```', '',
        f'The flagged proportion is {overview.loc["Motif-associated unidentified cells", "spillover_consistent_percent"]:.2f}% among motif-associated cells and {overview.loc["Other unidentified cells", "spillover_consistent_percent"]:.2f}% among controls. These are screening results, not proof that annotations were caused by spillover. Low evidence means the selected detector did not flag the cell; it does not establish a correct annotation. See marker_risk_summary.csv for marker-specific counts and denominators.', '',
        'Flagged motif-associated CD56 donor annotations (potential sources, not proven contamination):', '', '```', donors[donors.in_motif & donors.marker.eq('CD56')].to_string(index=False), '```', '',
        '## Results', '', 'Status counts (95th-percentile rule):', '',
        '```', pd.crosstab([cells.Group, cells.in_motif], cells.status_q95).to_string(), '```', '',
        'Equal-patient estimates and 2,000 patient-bootstrap 95% intervals (fractions of all unidentified cells, including unresolved cases in the denominator):', '', summary.to_string(index=False), '',
        'Complete-case sensitivity (35 scored non-DNA markers and no multiple-DNA-component concern; eligibility does not depend on the spillover flag):', '', '```', complete_summary.to_string(index=False), '```', '',
        'Coverage:', '', coverage.to_string(), '',
        '## Interpretation and methods', '',
        'Spillover-consistent flags require all three metrics above their matched control percentile, a donor at least 2x brighter than the focal cell, at least 1.5x interface asymmetry and boundary/interior concentration, and at least five raw counts across at least three interface pixels. dsDNA is measured but excluded from the 35-marker aggregate risk flag. Nuclear-marker flags are reported separately by compartment and need not represent lateral membrane spillover. No label is changed.', '',
        'Each marker has its own same-FOV reference: up to 50 non-motif unidentified cells nearest in standardized log(1+mask area) and log(1+50-pixel crowding); self is excluded, at least 30 references required. Reference percentiles are exploratory, not p-values, and flags across 35 markers are not multiple-testing-corrected. Threshold sensitivity uses 90th, 95th, and 99th percentiles. All metrics and thresholds are retained.', '',
        'The brightest geometry-valid donor is selected independently for each marker; all candidate interfaces are stored in fovs/*/interfaces.csv.gz. A 3-pixel mask-distance disk defines proximity, independently of motif graph edges. Neighbor-facing regions may overlap across donors but are deduplicated within each donor pair. Entire-mask means include only labeled pixels. Background label 0 is never used as a cell. Erosion uses a 3x3 square, repeated twice; distances are pixels, with no assumed physical calibration.', '',
        'Low evidence requires all 35 non-DNA markers evaluable and no spillover flag. Missing channels, missing donors, small interiors, clipped cells, centroid errors, insufficient controls, or multiple bright DNA components prevent a low-evidence conclusion. A positive marker flag still counts as evidence even when other markers are unavailable. Multi-component DNA is a segmentation-risk screen, not a doublet diagnosis.', '',
        'Lineage-pattern screens use pairs of markers above compartment-specific 10th-percentile values from at least ten geometry-valid annotated reference cells in the same FOV, with a positive intensity floor. They are label-derived compatibility screens, not independently validated classifiers. Membrane erosion and loss of real lineage signal can create apparent profile resolution. Unidentified cells can remain ambiguous for biological or technical reasons. Original pred_prob values are joined descriptively using the historic truncated-coordinate key; they may come from a different classification stage.', '',
        'Patient-bootstrap effects use paired motif-minus-control differences among patients with both groups, while individual group means use all represented patients. Pooled estimates weight patients equally. Sparse motif samples can produce unstable patient fractions. Within-FOV matching controls technical and morphological differences but does not remove all confounding by tissue context.', '',
        '## Files and reproduction', '',
        '`per_cell_per_marker.csv.gz`: strongest-donor marker measurements, matched thresholds, percentiles, flags and marker QC. `fovs/*/interfaces.csv.gz`: all donor-interface measurements. `per_cell_summary.csv`: unique-cell QC, flags and lineage screens. `coverage.csv`, `status_counts.csv`, `threshold_sensitivity.csv`, `patient_summary.csv`, `patient_bootstrap.csv`: numeric report tables. PNGs show status, patient uncertainty and threshold sensitivity. `blinded_review/` contains PNGs, a blank review form, and a separate unblinding key. No review outcomes are inferred.', '',
        'Run from repository root: `MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=2 python analysis/marker_ownership/unidentified_spillover.py`. Per-FOV checkpoints verify version, spatial cohort hash, and archive size/mtime before reuse; use --force to recompute. --max-fovs provides an explicitly partial run; use a separate --output directory for trials. --skip-review omits crop rendering. Original inputs are read-only.', '',
        '## Basis', '',
        '[Bussi et al., CellTune, Nature Methods (2026)](https://www.nature.com/articles/s41592-026-03162-2), Fig. 5k–m and Compute features / Landmark gating. The present screening thresholds, matched-control design, and bootstrap are our proposed adaptations, not a validated CellTune spillover detector.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    pictures = []
    for filename in ['risk_summary.png', 'marker_risk.png', 'patient_uncertainty.png', 'threshold_sensitivity.png']:
        encoded = base64.b64encode((output/filename).read_bytes()).decode()
        pictures.append(f'<img alt="{html.escape(filename)}" src="data:image/png;base64,{encoded}">')
    report = '''<!doctype html><meta charset="utf-8"><title>Unidentified-cell spillover audit</title>
<style>body{font:16px/1.5 system-ui;margin:40px auto;max-width:1150px;padding:0 20px;color:#24313b}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}img{max-width:100%;margin:20px 0}pre{white-space:pre-wrap;font-size:13px;background:#f5f7f9;padding:20px}h1{font-size:29px}</style>
<h1>Computational spillover-risk assessment</h1><p>Motif 20748339660 · unidentified cells · NN and NP</p>
<p><strong>Spillover has not been ruled out.</strong> Flags indicate a spatial pattern consistent with neighboring-cell signal. They do not establish causation or annotation accuracy. No manual review has been performed.</p>'''
    report += overview.round(3).to_html()+''.join(pictures)
    report += '<h2>Patient-bootstrap estimates</h2><p>Equal-patient means; unresolved cases included in the denominator.</p>'+summary.round(4).to_html(index=False)
    report += '<h2>Complete-case sensitivity</h2>'+complete_summary.round(4).to_html(index=False)
    report += '<h2>Methods, coverage and limitations</h2><pre>'+html.escape('\n'.join(lines))+'</pre>'
    (output/'report.html').write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/groups/assafza_group/assafza/Datasets/Melanoma'))
    parser.add_argument('--hits', type=Path, default=Path('/groups/assafza_group/assafza/Omri/melanoma_motif_instances/20748339660_hits_wide.csv'))
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/outputs/unidentified_spillover_20748339660')
    parser.add_argument('--max-fovs', type=int)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip-review', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'fovs').mkdir(exist_ok=True)
    cohort = load_cohort(args.data_dir, args.hits)
    print(f'{len(cohort)} spatial cells; {cohort.loc[cohort.pred_mapped.eq("unidentified"), "in_motif"].sum()} unique motif unidentified cells', flush=True)
    groups = list(cohort.groupby(['patient number', 'fov'], sort=True))
    if args.max_fovs: groups = groups[:args.max_fovs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process_fov, frame, args, args.output/'fovs'/f'{int(patient)}_{fov}') for (patient, fov), frame in groups]
        for job in as_completed(jobs): job.result()
    aggregate(args, cohort)
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
