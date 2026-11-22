"""Rank real motif cells by marker location, without changing cohort outcomes."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from analysis.marker_ownership import figure_marker_ownership as base

OUT=base.OUT/'central_marker_examples'


def localization(image,own,threshold):
    """Positive excess signal radius relative to the mask's uniform RMS radius."""
    yy,xx=np.where(own)
    if not len(yy):return {'measurement_failure':'missing_mask'}
    inner=ndi.binary_erosion(own,structure=np.ones((3,3),bool),iterations=2)
    values=image[own].astype(float);excess=np.maximum(values-threshold,0)
    radius=np.hypot(yy-yy.mean(),xx-xx.mean());rms=float(np.sqrt(np.mean(radius**2)))
    inside=inner[own];total=float(excess.sum())
    result=dict(mask_pixels=int(own.sum()),interior_pixels=int(inner.sum()),background_threshold=threshold,
        positive_pixels=int((values>threshold).sum()),interior_positive_pixels=int(((values>threshold)&inside).sum()),
        excess_signal_sum=total,interior_positive_fraction=float((image[inner]>threshold).mean()) if inner.any() else np.nan,
        central_signal_present=bool(((values>threshold)&inside).sum()>=3),measurement_failure='')
    if total>0 and rms>0:
        result.update(signal_mean_radius_px=float(np.sum(excess*radius)/total),
            relative_signal_radius=float(np.sum(excess*radius)/total/rms),
            signal_centroid_offset_px=float(np.hypot(np.sum(excess*(yy-yy.mean()))/total,np.sum(excess*(xx-xx.mean()))/total)),
            interior_excess_fraction=float(excess[inside].sum()/total))
    else:result.update(relative_signal_radius=np.nan,signal_mean_radius_px=np.nan,signal_centroid_offset_px=np.nan,interior_excess_fraction=np.nan)
    return result


def measure_fov(frame):
    fov=frame.iloc[0].fov;rows=[]
    with ZipFile(base.study.DATA/'raw_data'/f'{fov}.zip') as archive:
        mask=base.imaging.read_image(archive,'TIFs/segmentation_labels.tiff')
        for marker,group in frame.groupby('marker'):
            image=base.imaging.read_image(archive,f'TIFs/{marker}.tif')
            for r in group.to_dict('records'):
                yy,xx=np.where(mask==r['cell_id'])
                if not len(yy):result={'measurement_failure':'missing_mask'}
                else:
                    sl=(slice(max(0,yy.min()-3),min(mask.shape[0],yy.max()+4)),slice(max(0,xx.min()-3),min(mask.shape[1],xx.max()+4)))
                    result=localization(image[sl],mask[sl]==r['cell_id'],float(r['pixel_positive_threshold']))
                rows.append({**r,**result})
    return rows


def measure():
    OUT.mkdir(exist_ok=True)
    cols=base.study.KEY+['marker','role','pixel_positive_threshold','whole_mean','erode2_mean','donor_cell_id',
        'donor_annotation','ownership_risk_q95','interface_log2_asymmetry','selective_interface_log2_loss']
    d=pd.read_csv(base.OUT/'per_cell_per_marker.csv.gz',usecols=cols,low_memory=False)
    d=d[d.role.eq('candidate')]
    members=pd.read_csv(base.OUT/'motif_membership.csv')
    d=d.merge(members,on=base.study.KEY,validate='many_to_one')
    d=d[d.apply(lambda r:r.marker in base.study.CORE[r.candidate_kind],axis=1)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(measure_fov,g):fov for fov,g in d.groupby('fov')}
        for future in as_completed(futures):
            rows.extend(future.result());print(f'{futures[future]} measured',flush=True)
    markers=pd.DataFrame(rows);markers.to_csv(OUT/'all_candidate_marker_centrality.csv',index=False)
    geometry=pd.read_csv(base.OUT/'cell_geometry.csv.gz',usecols=base.study.KEY+['geometry_valid','bright_dna_components'])
    cells=[]
    for key,g in markers.groupby(base.study.KEY):
        cells.append(dict(zip(base.study.KEY,key))|dict(motif=g.motif.iloc[0],candidate_kind=g.candidate_kind.iloc[0],
            central_markers=int(g.central_signal_present.sum()),detected_markers=int(g.positive_pixels.ge(3).sum()),
            worst_relative_radius=float(g.relative_signal_radius.max()) if g.relative_signal_radius.notna().all() else np.inf,
            mean_relative_radius=float(g.relative_signal_radius.mean()),
            minimum_interior_fraction=float(g.interior_excess_fraction.min()),
            any_core_risk_flag=bool(g.ownership_risk_q95.eq(1).any())))
    cells=pd.DataFrame(cells).merge(geometry,on=base.study.KEY,validate='one_to_one')
    cells['display_geometry_eligible']=cells.geometry_valid&cells.bright_dna_components.eq(1)
    cells=cells.sort_values(['motif','display_geometry_eligible','any_core_risk_flag','central_markers','worst_relative_radius','minimum_interior_fraction'] ,
        ascending=[True,False,True,False,True,False])
    cells['motif_rank']=cells.groupby('motif').cumcount()+1
    cells.to_csv(OUT/'all_candidate_cell_ranking.csv',index=False)
    (OUT/'selection_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        population=167,core_markers=base.study.CORE,background='Existing FOV/channel unsegmented-background 99th percentile.',
        signal='max(raw pixel intensity - background threshold, 0)',
        centrality='Mean pixel radius weighted by positive excess signal, divided by the mask uniform RMS radius. Smaller is more central; zero total excess is undefined.',
        central_detection='At least 3 above-background pixels after 2-pixel square erosion, per marker.',
        selection_order='Valid geometry and one bright DNA component; no prior core-marker risk flag; more centrally detected core markers; lower worst-marker relative radius; greater minimum interior signal fraction.',
        interpretation='Deliberately selected for central staining; not random, not cohort validation, not evidence of a validated HEV/DC identity. Existing risk outcomes remain unchanged.'),indent=2))
    print(cells.groupby('motif').head(4).to_string(index=False),flush=True)


if __name__=='__main__':measure()
