"""Spatial marker ownership in four motifs; exploratory profiles, not reannotation.

Spatial measurements are inspired by CellSighter Fig. 3 and CellTune Fig. 5/
Supplementary S1/S4. Our thresholds are exploratory, not validated by those papers.
"""

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
from scipy.spatial import cKDTree

from analysis.marker_ownership import unidentified_spillover as old

ROOT=_REPO_ROOT
DATA=Path('/groups/assafza_group/assafza/Datasets/Melanoma')
HITS=Path('/groups/assafza_group/assafza/Omri/melanoma_motif_instances')
OUT=ROOT/'analysis/outputs/motif_marker_ownership'
KEY=['patient number','fov','cell_id']
CONFIG=[('I',4071751116,'cell3','HEV'),('II',15341317580,'cell2','HEV'),
        ('III',17763644908,'cell3','DC'),('IV',4880694732,'cell1','DC')]
CORE={'HEV':['CD31','MECA-79'],'DC':['CD11c','HLA-DR-DP-DQ']}
MARKERS=['CD31','MECA-79','CD11c','HLA-DR-DP-DQ','CD3','CD45','CD20','CD68','CD163','CD206','DC-SIGN','dsDNA','SMA']
POSITIVE={'CD31':['hevs'],'MECA-79':['hevs'],'CD11c':['DC'],'HLA-DR-DP-DQ':['DC'],
          'CD3':['cd4 t cell','memory%cd4 t cell','cd8 t cell','treg'],
          'CD45':['cd4 t cell','memory%cd4 t cell','cd8 t cell','treg','mac','DC','b cell','nk cell'],
          'CD20':['b cell'],'CD68':['mac'],'CD163':['mac'],'CD206':['mac'],'DC-SIGN':['DC']}
NEGATIVE={m:['cd4 t cell','memory%cd4 t cell','cd8 t cell','treg','tumor','stroma']
          for m in ['MECA-79','CD11c','HLA-DR-DP-DQ','DC-SIGN','CD68','CD163','CD206','CD20']}
NEGATIVE.update({'CD31':['tumor','stroma'],'CD3':['b cell','tumor','stroma','hevs'],
                 'CD45':['tumor','stroma','hevs']})
VERSION='ownership-1'


def population():
    spatial=pd.read_csv(DATA/'mapped_cell_types.csv')
    members=[]; occurrences=[]
    for roman,mid,col,kind in CONFIG:
        hits=pd.read_csv(HITS/f'{mid}_hits_wide.csv')
        counts=hits.groupby(['patient','FOV',col]).size().reset_index(name='occurrences')
        counts=counts.rename(columns={'patient':'patient number','FOV':'fov',col:'cell_id'})
        counts['motif']=roman;counts['motif_id']=mid;counts['candidate_kind']=kind
        members.append(counts)
        hits['motif']=roman;hits['candidate_kind']=kind;hits['motif_id']=mid;hits['focal_column']=col
        occurrences.append(hits)
    members=pd.concat(members,ignore_index=True).merge(spatial[KEY+['Group','pred_mapped']],on=KEY,validate='many_to_one')
    assert members.pred_mapped.eq('unidentified').all()
    assert not members.duplicated(['motif']+KEY).any()
    return spatial,members,pd.concat(occurrences,ignore_index=True)


def geometry_info(mask,frame):
    n=int(mask.max())+1;count=np.bincount(mask.ravel(),minlength=n)
    yy,xx=np.indices(mask.shape);sy=old.sums(yy,mask,n);sx=old.sums(xx,mask,n)
    ids=frame.cell_id.to_numpy(int);present=(ids<n)
    area=np.zeros(len(ids),int);area[present]=count[ids[present]];present&=area>0
    error=np.full(len(ids),np.nan)
    valid_ids=ids[present]
    error[present]=np.hypot(sy[valid_ids]/count[valid_ids]-frame.loc[present,'centroid-0'],
                            sx[valid_ids]/count[valid_ids]-frame.loc[present,'centroid-1'])
    eroded=old.erode_labels(old.erode_labels(mask));inner=np.bincount(eroded.ravel(),minlength=n)
    inner_n=np.zeros(len(ids),int);inner_n[present]=inner[valid_ids]
    edge_ids=np.unique(np.r_[mask[0],mask[-1],mask[:,0],mask[:,-1]])
    info=frame.copy().reset_index(drop=True)
    info['mask_area']=area;info['mask_present']=present;info['centroid_error_px']=error
    info['edge_clipped']=np.isin(ids,edge_ids);info['interior2_pixels']=inner_n
    info['geometry_valid']=present&(error<=.75)&~info.edge_clipped&(area>=20)&(inner_n>=5)
    info['crowding_50px']=cKDTree(info[['centroid-0','centroid-1']]).query_ball_point(
        info[['centroid-0','centroid-1']],50,return_length=True)-1
    return info


def matched_controls(info,target_ids,seed):
    rng=np.random.default_rng(seed)
    controls=info[info.pred_mapped.eq('unidentified')&~info.cell_id.isin(target_ids)&info.geometry_valid]
    links=[]
    scale=np.log1p(info[['mask_area','crowding_50px']]).std().replace(0,1).to_numpy()
    control_coords=np.log1p(controls[['mask_area','crowding_50px']].to_numpy())/scale
    for row in info[info.cell_id.isin(target_ids)].itertuples():
        point=np.log1p([row.mask_area,row.crowding_50px])/scale
        dist=np.sum((control_coords-point)**2,axis=1)
        # Random ties, with a fixed seed; no image-based selection.
        order=np.lexsort((rng.random(len(controls)),dist))[:50]
        for j in order:links.append(dict(target_cell_id=row.cell_id,control_cell_id=int(controls.iloc[j].cell_id),match_distance=float(np.sqrt(dist[j]))))
    return pd.DataFrame(links,columns=['target_cell_id','control_cell_id','match_distance'])


def boundary_measurements(image,mask,cell_id,donor_id,positive_threshold,eps,interface_radius=3,shams=12):
    """Compare removal of a donor-facing boundary with equal-area boundary removals.

    Sham removals are angularly contiguous in the remaining boundary, selected at
    fixed angles. Insufficient non-facing area makes the comparison unmeasured.
    """
    own=mask==int(cell_id)
    if not own.any():return {'boundary_failure':'missing_mask'}
    eroded=ndi.binary_erosion(own,structure=np.ones((3,3),bool),iterations=2);boundary=own&~eroded
    y,x=np.where(boundary);v=image[boundary].astype(float)
    interface=np.zeros(len(v),bool)
    has_donor=donor_id is not None and np.isfinite(donor_id) and int(donor_id)>0
    if has_donor:
        donor=mask==int(donor_id)
        structure=np.fromfunction(lambda i,j:(i-interface_radius)**2+(j-interface_radius)**2<=interface_radius**2,
                                  (2*interface_radius+1,2*interface_radius+1))
        near=ndi.binary_dilation(donor,structure=structure)
        interface=near[y,x]
    away=~interface;k=int(interface.sum());n=len(v);whole_sum=float(image[own].sum());area=int(own.sum())
    positive=v>positive_threshold
    result=dict(boundary_pixels=n,removed_interface_pixels=k,away_boundary_pixels=int(away.sum()),
        pixel_positive_threshold=float(positive_threshold),boundary_positive_fraction=float(positive.mean()),
        away_positive_pixels=int(positive[away].sum()),
        away_positive_fraction=float(positive[away].mean()) if away.any() else np.nan,
        away_boundary_mean=float(v[away].mean()) if away.any() else np.nan,
        retained_cell_mean=(whole_sum-float(v[interface].sum()))/(area-k) if area>k else np.nan,
        interface_positive_fraction=float(positive[interface].mean()) if k else np.nan,
        interface_signal_sum=float(v[interface].sum()),
        boundary_failure='',no_proximity_donor=not has_donor,
        sham_boundary_retained_mean=np.nan,sham_cell_retained_mean=np.nan,
        selective_interface_log2_loss=np.nan,sham_repetitions=0)
    if k==0:
        result.update(sham_boundary_retained_mean=float(v.mean()),sham_cell_retained_mean=whole_sum/area,
                      selective_interface_log2_loss=0.,sham_repetitions=shams)
        return result
    if k<3 or away.sum()<max(k+3,6):
        result['boundary_failure']='insufficient_interface_or_sham_area';return result
    cy,cx=np.where(own);angles=np.arctan2(y-cy.mean(),x-cx.mean())
    allowed=np.flatnonzero(away);sham_b=[];sham_cell=[]
    for angle in np.linspace(-np.pi,np.pi,shams,endpoint=False):
        distance=np.abs(np.angle(np.exp(1j*(angles[allowed]-angle))))
        selected=allowed[np.argsort(distance,kind='stable')[:k]]
        removed=float(v[selected].sum())
        sham_b.append((float(v.sum())-removed)/(n-k))
        sham_cell.append((whole_sum-removed)/(area-k))
    result.update(sham_boundary_retained_mean=float(np.median(sham_b)),
                  sham_cell_retained_mean=float(np.median(sham_cell)),sham_repetitions=shams,
                  selective_interface_log2_loss=float(np.log2((np.median(sham_b)+eps)/(result['away_boundary_mean']+eps))))
    return result


def process_fov(frame,target_rows,force=False):
    patient=int(frame.iloc[0]['patient number']);fov=frame.iloc[0].fov
    dest=OUT/'fovs'/f'{patient}_{fov}';dest.mkdir(parents=True,exist_ok=True)
    path=DATA/'raw_data'/f'{fov}.zip'
    provenance=dict(version=VERSION,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archive=str(path),archive_size=path.stat().st_size if path.exists() else None,
        archive_mtime_ns=path.stat().st_mtime_ns if path.exists() else None,
        target_ids=sorted(target_rows.cell_id.unique().astype(int).tolist()),markers=MARKERS)
    if not force and (dest/'complete.json').exists() and json.loads((dest/'complete.json').read_text())==provenance:
        return f'{fov}: cached'
    if not path.exists():
        (dest/'failure.json').write_text(json.dumps(dict(reason='missing_archive',**provenance)));return f'{fov}: missing archive'
    with ZipFile(path) as archive:
        mask=old.read_image(archive,'TIFs/segmentation_labels.tiff').astype(np.int32)
        info=geometry_info(mask,frame)
        target_ids=target_rows.cell_id.unique().astype(int)
        links=matched_controls(info,target_ids,42069+patient)
        links['patient number']=patient;links['fov']=fov
        links.to_csv(dest/'matched_controls.csv',index=False)
        # Provisional annotated reference pools, sampled before image measurement.
        # The all-annotated and local-maximum subsets are both retained for review.
        rng=np.random.default_rng(52069+patient)
        ref_ids=[]
        for types in POSITIVE.values():
            pool=info[info.geometry_valid&info.pred_mapped.isin(types)]
            if len(pool)>100:pool=pool.iloc[rng.choice(len(pool),100,replace=False)]
            ref_ids.extend(pool.cell_id.tolist())
        selected_ids=sorted(set(target_ids)|set(links.control_cell_id)|set(ref_ids))
        selected=info[info.cell_id.isin(selected_ids)].copy()
        selected['role']=np.where(selected.cell_id.isin(target_ids),'candidate',np.where(
            selected.pred_mapped.eq('unidentified'),'matched_unidentified_control','annotated_reference'))
        g=old.geometry(mask,np.array(selected_ids))
        rows=[]
        for marker in MARKERS:
            name=f'TIFs/{marker}.tif'
            if name not in archive.namelist():
                rows.append(pd.DataFrame({'cell_id':selected_ids,'marker':marker,'measurement_failure':'missing_channel'}));continue
            image=old.read_image(archive,name)
            if image.shape!=mask.shape:
                rows.append(pd.DataFrame({'cell_id':selected_ids,'marker':marker,'measurement_failure':'shape_mismatch'}));continue
            if marker=='dsDNA':
                selected['bright_dna_components']=old.nuclear_qc(image,mask,np.array(selected.cell_id))
            measured,interfaces,whole,inner=old.extract_marker(g,image,np.array(selected_ids))
            # Background is an empirical display-independent detection reference,
            # not an assertion of antibody specificity or a calibrated positivity call.
            bg=image[mask==0]
            pixel_threshold=float(np.quantile(bg,.99)) if len(bg)>=100 else 0.
            negative_ids=info.loc[info.geometry_valid&info.pred_mapped.isin(NEGATIVE.get(marker,[])),'cell_id'].to_numpy(int)
            negative_mean_threshold=float(np.quantile(whole[negative_ids],.95)) if len(negative_ids)>=10 else np.nan
            measured['pixel_positive_threshold']=pixel_threshold
            measured['negative_reference_n']=len(negative_ids)
            measured['negative_reference_mean_q95']=negative_mean_threshold
            if marker in POSITIVE:
                extra=[]
                # Work on tight crops; preserve true boundary geometry by padding.
                boxes=ndi.find_objects(mask)
                for row in measured.itertuples():
                    box=boxes[int(row.cell_id)-1] if 0<int(row.cell_id)<=len(boxes) else None
                    if box is None:extra.append({'cell_id':row.cell_id,'boundary_failure':'missing_mask'});continue
                    # Include the donor's pixels within the requested proximity radius.
                    sl=tuple(slice(max(0,b.start-5),min(mask.shape[i],b.stop+5)) for i,b in enumerate(box))
                    values=boundary_measurements(image[sl],mask[sl],row.cell_id,row.donor_cell_id,
                                                  pixel_threshold,row.pseudocount)
                    values['cell_id']=row.cell_id;extra.append(values)
                measured=measured.merge(pd.DataFrame(extra),on='cell_id',validate='one_to_one',suffixes=('','_boundary'))
            measured['marker']=marker;measured['measurement_failure']=''
            measured['donor_annotation']=measured.donor_cell_id.map(info.set_index('cell_id').pred_mapped)
            rows.append(measured)
        selected.to_csv(dest/'cell_geometry.csv',index=False)
        table=pd.concat(rows,ignore_index=True).merge(selected[['cell_id','role','pred_mapped','geometry_valid']],on='cell_id',validate='many_to_one')
        table['patient number']=patient;table['fov']=fov
        table.to_csv(dest/'measurements.csv.gz',index=False)
    (dest/'complete.json').write_text(json.dumps(provenance,indent=2))
    return f'{fov}: {len(target_ids)} candidates, {links.control_cell_id.nunique()} controls'


def calibrate(frame):
    frame=frame.copy();frame['reference_local_max']=False
    for q in [90,95,99]:frame[f'ownership_risk_q{q}']=np.nan
    for marker,idx in frame.groupby('marker').groups.items():
        d=frame.loc[idx]
        positive=d.role.eq('annotated_reference')&d.pred_mapped.isin(POSITIVE.get(marker,[]))&d.geometry_valid
        local=positive&d.whole_mean.gt(d.donor_mean.fillna(0))&d.whole_mean.gt(d.pseudocount*5)
        frame.loc[idx,'reference_local_max']=local.to_numpy()
        refs=d[local]
        # No fallback to pooling absolute intensities across FOVs.
        thresholds={'whole_reference_threshold':np.nan,'away_reference_threshold':np.nan,
                    'coverage_reference_threshold':np.nan,'retained_reference_threshold':np.nan}
        if len(refs)>=5:
            neg=float(d.negative_reference_mean_q95.iloc[0])
            thresholds['whole_reference_threshold']=max(float(refs.whole_mean.quantile(.10)),neg if np.isfinite(neg) else 0,
                                                       float(d.pseudocount.iloc[0]*5))
            thresholds['away_reference_threshold']=float(refs.away_boundary_mean.quantile(.10))
            thresholds['coverage_reference_threshold']=max(.05,float(refs.away_positive_fraction.quantile(.10)))
            thresholds['retained_reference_threshold']=float(refs.retained_cell_mean.quantile(.10))
        for name,value in thresholds.items():frame.loc[idx,name]=value
        frame.loc[idx,'positive_reference_n']=int(positive.sum())
        frame.loc[idx,'local_max_reference_n']=len(refs)
        metrics=['interface_log2_asymmetry','donor_log2_ratio','selective_interface_log2_loss']
        controls=d[d.role.eq('matched_unidentified_control')&d.geometry_valid].dropna(subset=metrics)
        if len(controls)<30:continue
        coords=np.log1p(controls[['mask_area','crowding_50px']].to_numpy(float))
        scales=np.nanstd(coords,axis=0);scales[scales==0]=1
        tree=cKDTree(coords/scales)
        points=np.log1p(d[['mask_area','crowding_50px']].to_numpy(float))/scales
        _,nearest=tree.query(points,k=min(51,len(controls)))
        own=d.cell_id.to_numpy();control_ids=controls.cell_id.to_numpy()
        selected=[];counts=[]
        for i,ns in enumerate(nearest):
            valid=ns[control_ids[ns]!=own[i]][:50];counts.append(len(valid))
            selected.append(valid)
        width=max(counts);matrix=np.zeros((len(d),width),int);padding=np.ones(matrix.shape,bool)
        for i,ns in enumerate(selected):matrix[i,:len(ns)]=ns;padding[i,:len(ns)]=False
        values=controls[metrics].to_numpy(float)[matrix];values[padding]=np.nan
        thresholds=np.nanquantile(values,[.90,.95,.99],axis=1)
        observed=d[metrics].to_numpy(float)
        available=np.isfinite(observed).all(axis=1)&(np.array(counts)>=30)
        no_interface=d.removed_interface_pixels.eq(0).to_numpy()&(np.array(counts)>=30)
        frame.loc[idx,'risk_reference_n']=counts
        basic=(d.interface_signal_sum.ge(5)&d.interface_log2_asymmetry.ge(np.log2(1.5))&
               d.donor_log2_ratio.ge(1)&d.selective_interface_log2_loss.ge(np.log2(1.5))).to_numpy()
        for j,q in enumerate([90,95,99]):
            for k,m in enumerate(metrics):frame.loc[idx,f'{m}_q{q}']=thresholds[j,:,k]
            risk=basic&(observed>thresholds[j]).all(axis=1)
            frame.loc[idx,f'ownership_risk_q{q}']=np.where(available|no_interface,risk.astype(float),np.nan)
    frame['whole_marker_supported']=frame.geometry_valid&frame.local_max_reference_n.ge(5)&frame.whole_mean.ge(frame.whole_reference_threshold)
    frame['away_marker_supported']=frame.whole_marker_supported&frame.away_boundary_mean.ge(frame.away_reference_threshold)&frame.away_positive_fraction.ge(frame.coverage_reference_threshold)&frame.away_positive_pixels.ge(3)&frame.retained_cell_mean.ge(frame.retained_reference_threshold)
    return frame


def classify(measurements,geometry):
    rows=[]
    lookup=geometry.set_index('cell_id')
    for cid,table in measurements[measurements.role.isin(['candidate','matched_unidentified_control'])].groupby('cell_id'):
        cell=lookup.loc[cid]
        for kind,core in CORE.items():
            m=table.set_index('marker').reindex(core)
            competing=['CD45'] if kind=='HEV' else ['CD3','CD20','CD68','CD163','CD206']
            positive_other=[marker for marker in competing if marker in table.marker.values and bool(table.loc[table.marker.eq(marker),'away_marker_supported'].iloc[0])]
            for q in [90,95,99]:
                flag=m[f'ownership_risk_q{q}']
                missing=not cell.geometry_valid or cell.bright_dna_components!=1 or m.local_max_reference_n.lt(5).any() or m.whole_mean.isna().any()
                risk=flag.eq(True).any()
                risk_unmeasured=flag.isna().any()
                whole=bool(m.whole_marker_supported.fillna(False).all());away=bool(m.away_marker_supported.fillna(False).all())
                if missing:status='unresolved_quality_or_reference'
                elif risk:status='spillover_consistent'
                elif risk_unmeasured:status='unresolved_interface'
                elif whole and away and positive_other:status='unresolved_competing_lineage'
                elif whole and away:status='reference_like_profile_low_risk'
                else:status='proposed_profile_not_supported'
                rows.append(dict(cell_id=cid,candidate_kind=kind,threshold=q,status=status,
                    whole_profile_supported=whole,noninterface_profile_supported=away,
                    competing_intrinsic_markers=';'.join(positive_other),
                    risk_core_markers=';'.join(m.index[flag.eq(True)]),
                    role=cell.role,Group=cell.Group,**{'patient number':int(cell['patient number']),'fov':cell.fov}))
    return pd.DataFrame(rows)


def summarize(cells,members,links):
    chosen=members.merge(cells,on=KEY+['candidate_kind'],validate='one_to_many')
    chosen.to_csv(OUT/'candidate_cell_outcomes.csv',index=False)
    counts=chosen.groupby(['motif','candidate_kind','threshold','status']).size().rename('cells').reset_index()
    counts.to_csv(OUT/'status_counts.csv',index=False)
    matched=links.rename(columns={'control_cell_id':'cell_id'}).merge(cells,on=KEY,validate='many_to_many')
    target_lookup=members[KEY+['motif','candidate_kind']].rename(columns={'cell_id':'target_cell_id'})
    matched=matched.merge(target_lookup,on=['patient number','fov','target_cell_id','candidate_kind'],validate='many_to_many')
    matched['risk']=matched.status.eq('spillover_consistent')
    matched['supported']=matched.status.eq('reference_like_profile_low_risk')
    matches=matched.groupby(['motif','candidate_kind','threshold','patient number','fov','target_cell_id'])[['risk','supported']].mean().reset_index()
    matches.to_csv(OUT/'matched_control_outcome_fractions.csv',index=False)
    paired=chosen.merge(matches,left_on=['motif','candidate_kind','threshold']+KEY,
                        right_on=['motif','candidate_kind','threshold','patient number','fov','target_cell_id'],how='left')
    paired['risk_difference']=paired.status.eq('spillover_consistent').astype(float)-paired.risk
    paired['support_difference']=paired.status.eq('reference_like_profile_low_risk').astype(float)-paired.supported
    patient=paired.groupby(['motif','candidate_kind','threshold','patient number']).agg(
        candidates=('cell_id','size'),risk_fraction=('status',lambda x:float(x.eq('spillover_consistent').mean())),
        supported_fraction=('status',lambda x:float(x.eq('reference_like_profile_low_risk').mean())),
        risk_difference=('risk_difference','mean'),support_difference=('support_difference','mean')).reset_index()
    patient.to_csv(OUT/'patient_summary.csv',index=False)
    rng=np.random.default_rng(62069);bootstrap=[]
    for key,d in patient.groupby(['motif','candidate_kind','threshold']):
        for metric in ['risk_fraction','supported_fraction','risk_difference','support_difference']:
            values=d[metric].dropna().to_numpy();n=len(values)
            if not n:continue
            draws=values[rng.integers(0,n,(2000,n))].mean(axis=1)
            bootstrap.append(dict(zip(['motif','candidate_kind','threshold'],key),metric=metric,patients=n,
                                  patient_balanced_mean=float(values.mean()),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975))))
    pd.DataFrame(bootstrap).to_csv(OUT/'patient_bootstrap.csv',index=False)
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--force',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    spatial,members,occurrences=population();members.to_csv(OUT/'motif_membership.csv',index=False)
    occurrences.to_csv(OUT/'original_motif_occurrences.csv',index=False)
    groups=[]
    for (patient,fov),targets in members.groupby(['patient number','fov']):
        groups.append((spatial[spatial['patient number'].eq(patient)&spatial.fov.eq(fov)].reset_index(drop=True),targets))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(process_fov,frame,targets,args.force) for frame,targets in groups]
        for future in as_completed(futures):print(future.result(),flush=True)
    measured=[];geometries=[];links=[];outcomes=[]
    for frame,targets in groups:
        patient=int(frame.iloc[0]['patient number']);fov=frame.iloc[0].fov;dest=OUT/'fovs'/f'{patient}_{fov}'
        if not (dest/'complete.json').exists():continue
        d=pd.read_csv(dest/'measurements.csv.gz',low_memory=False)
        g=pd.read_csv(dest/'cell_geometry.csv');g['patient number']=patient;g['fov']=fov
        d=d.merge(g[['cell_id','mask_area','crowding_50px']],on='cell_id',validate='many_to_one')
        d=calibrate(d);d.to_csv(dest/'scored_measurements.csv.gz',index=False)
        measured.append(d);geometries.append(g);links.append(pd.read_csv(dest/'matched_controls.csv'))
        outcomes.append(classify(d,g));print(f'{fov}: calibrated',flush=True)
    all_measured=pd.concat(measured,ignore_index=True);all_geometry=pd.concat(geometries,ignore_index=True)
    all_outcomes=pd.concat(outcomes,ignore_index=True);all_links=pd.concat(links,ignore_index=True)
    all_measured.to_csv(OUT/'per_cell_per_marker.csv.gz',index=False)
    all_geometry.to_csv(OUT/'cell_geometry.csv.gz',index=False);all_links.to_csv(OUT/'matched_control_links.csv.gz',index=False)
    all_outcomes.to_csv(OUT/'all_cell_outcomes.csv.gz',index=False)
    selected=summarize(all_outcomes,members,all_links)
    # Preserve failures explicitly in the coverage table rather than silently dropping.
    coverage=members.merge(selected[selected.threshold.eq(95)][KEY+['candidate_kind','status']],on=KEY+['candidate_kind'],how='left',validate='many_to_one')
    coverage['status']=coverage.status.fillna('unresolved_missing_fov_measurements')
    coverage.to_csv(OUT/'coverage.csv',index=False)
    retained=[]
    for roman,mid,col,kind in CONFIG:
        hits=occurrences[occurrences.motif.eq(roman)]
        for q in [90,95,99]:
            status=selected[selected.motif.eq(roman)&selected.threshold.eq(q)].set_index(KEY).status
            observed=[status.get((int(r.patient),r.FOV,int(getattr(r,col))),'unresolved_missing_fov_measurements') for r in hits.itertuples()]
            retained.append(dict(motif=roman,threshold=q,original_occurrences=len(hits),
                occurrences_with_measured_no_core_flag=sum(x in ['reference_like_profile_low_risk','proposed_profile_not_supported'] for x in observed),
                occurrences_with_reference_like_profile=sum(x=='reference_like_profile_low_risk' for x in observed)))
    pd.DataFrame(retained).to_csv(OUT/'descriptive_occurrence_retention.csv',index=False)
    metadata=dict(version=VERSION,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unique_candidates=len(members[KEY].drop_duplicates()),motif_memberships=len(members),FOVs=len(groups),
        markers=MARKERS,primary_rules=CORE,seed=42069,
        interpretation='Exploratory reference-like profiles and spillover-consistent patterns; no reannotation, independent positivity validation, blinded audit verdict, or enrichment recalculation.',
        calibration='Within-FOV provisional locally maximal annotated references; matched non-motif unidentified controls; 90/95/99 percentile sensitivity. Not thresholds validated by CellSighter/CellTune.')
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print(selected[selected.threshold.eq(95)].groupby(['motif','status']).size().to_string(),flush=True)


if __name__=='__main__':main()
