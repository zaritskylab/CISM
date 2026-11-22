"""All motif-I/II unidentified–annotated-HEV pairs, measured from raw TIFFs."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from zipfile import ZipFile
import hashlib
import html
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import export_spillover_microscopy_panels as imaging
from analysis.marker_ownership import export_four_motif_microscopy as source

OUT=source.ROOT/'analysis/outputs/meca79_motif_I_II'
CURRENT=source.ROOT/'analysis/outputs/unidentified_motif_I_IV_examples/marker_focused/example_manifest.csv'
CONFIG=[('I',4071751116,'cell3','cell4'),('II',15341317580,'cell2','cell3')]


def pair_list():
    frames=[]
    for roman,mid,u,h in CONFIG:
        hits=pd.read_csv(source.SOURCE/f'{mid}_hits_wide.csv')
        pairs=hits[['patient','FOV',u,h]].rename(columns={u:'unidentified_cell_id',h:'hev_cell_id'})
        pairs=pairs.groupby(list(pairs.columns),dropna=False).size().reset_index(name='motif_occurrences')
        pairs['motif']=roman; pairs['motif_id']=mid
        frames.append(pairs)
    return pd.concat(frames,ignore_index=True)


def measurements(mask,raw,cid,cell):
    binary=mask==int(cid); y,x=np.where(binary)
    if not len(y): return {'mask_pixels':0},'missing_mask_label'
    eroded=ndi.binary_erosion(binary); boundary=binary&~eroded
    result=dict(mean_MECA79=float(raw[binary].mean()),mask_pixels=int(binary.sum()),
        boundary_1px_mean_MECA79=float(raw[boundary].mean()),
        interior_1px_mean_MECA79=float(raw[eroded].mean()) if eroded.any() else np.nan,
        fraction_nonzero_MECA79=float((raw[binary]>0).mean()),
        touches_image_edge=bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any()),
        centroid_error_px=float(np.hypot(y.mean()-cell['centroid-0'],x.mean()-cell['centroid-1'])))
    return result,'ok' if result['centroid_error_px']<.75 else 'centroid_mismatch'


def region(mask,ids,padding):
    y,x=np.where(np.isin(mask,ids))
    half=max((y.max()-y.min())/2,(x.max()-x.min())/2)+padding
    return imaging.bounds((y.min()+y.max())/2,(x.min()+x.max())/2,half,mask.shape)


def outlines(ax,mask,sl,u,h):
    for cid,color,style in [(u,'white','--'),(h,'#00e5ff','-')]:
        binary=mask[sl]==int(cid)
        if binary.any() and (~binary).any():
            ax.contour(binary.astype(float),levels=[.5],colors=[color],linewidths=.7,linestyles=style)


def draw(ax,case,column):
    sl=case['pair_region'] if column<2 else case['hev_region']
    marker='MECA-79' if column in [1,2] else 'CD31' if column==3 else 'dsDNA'
    if column==0:
        rgb=np.zeros(case['mask'][sl].shape+(3,))
        for m,color in [('MECA-79',(0,1,0)),('dsDNA',(.15,.3,1))]:
            rgb+=imaging.normalized(case['images'][m],case['limits'][m],sl)[:,:,None]*np.array(color)
        rgb=np.clip(rgb,0,1)
    else:
        rgb=imaging.normalized(case['images'][marker],case['limits'][marker],sl)
        rgb=rgb[:,:,None]*np.array((0,1,0) if marker!='dsDNA' else (1,1,1))
    ax.imshow(rgb,interpolation='nearest')
    outlines(ax,case['mask'],sl,case['u'],case['h'])
    imaging.scale_bar(ax,case['mask'][sl].shape,20 if column<2 else 10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=pair_list(); spatial=pd.read_csv(imaging.DATA/'mapped_cell_types.csv')
    current=pd.read_csv(CURRENT); current=current[current.roman.isin(['I','II'])]
    records=[]; cases=[]; archives={}
    for (patient,fov),batch in pairs.groupby(['patient','FOV'],sort=True):
        archive_path=imaging.DATA/'raw_data'/f'{fov}.zip'
        frame=spatial[spatial['patient number'].eq(patient)&spatial.fov.eq(fov)].set_index('cell_id')
        if not archive_path.exists():
            for _,pair in batch.iterrows():records.append({**pair.to_dict(),'quality_status':'missing_archive','archive':str(archive_path)})
            continue
        archives[str(archive_path)]={'size':archive_path.stat().st_size,'mtime_ns':archive_path.stat().st_mtime_ns}
        with ZipFile(archive_path) as z:
            try:
                mask=imaging.read_image(z,'TIFs/segmentation_labels.tiff')
                raw=imaging.read_image(z,'TIFs/MECA-79.tif')
            except KeyError:
                for _,pair in batch.iterrows():records.append({**pair.to_dict(),'quality_status':'missing_archive_member','archive':str(archive_path)})
                continue
            selected_here=current[current['patient number'].eq(patient)&current.fov.eq(fov)]
            images={'MECA-79':raw}
            if len(selected_here):
                images.update({m:imaging.read_image(z,f'TIFs/{m}.tif') for m in ['CD31','dsDNA']})
        for _,pair in batch.iterrows():
            rec={**pair.to_dict(),'archive':str(archive_path),'mask_member':'TIFs/segmentation_labels.tiff','marker_member':'TIFs/MECA-79.tif'}
            u,h=int(pair.unidentified_cell_id),int(pair.hev_cell_id)
            errors=[]
            if raw.shape!=mask.shape: errors.append('image_mask_shape_mismatch')
            for prefix,cid,expected in [('unidentified',u,'unidentified'),('hev',h,'hevs')]:
                if cid not in frame.index:
                    errors.append(prefix+'_missing_cell_table_row');continue
                cell=frame.loc[cid]; rec['Group']=cell.Group
                rec[prefix+'_annotation']=cell.pred_mapped
                if cell.pred_mapped!=expected:errors.append(prefix+'_annotation_mismatch')
                if raw.shape==mask.shape:
                    values,status=measurements(mask,raw,cid,cell)
                    rec.update({prefix+'_'+k:v for k,v in values.items()})
                    if status!='ok':errors.append(prefix+'_'+status)
            rec['quality_status']=';'.join(errors) if errors else 'ok'
            um=rec.get('unidentified_mean_MECA79',np.nan);hm=rec.get('hev_mean_MECA79',np.nan)
            rec['hev_minus_unidentified_mean']=hm-um
            rec['hev_over_unidentified_mean']=hm/um if um>0 else np.nan
            rec['ratio_status']='finite' if um>0 else 'undefined_zero_unidentified_mean' if um==0 else 'missing'
            match=selected_here[selected_here.roman.eq(pair.motif)&selected_here.cell_id.eq(u)]
            rec['current_figure_example']=''
            if len(match):
                selected=match.iloc[0];instance=json.loads(selected.selected_instance_json)
                expected_neighbor=int(instance['cell4' if pair.motif=='I' else 'cell3'])
                if expected_neighbor==h:
                    rec['current_figure_example']=selected.example_id
                    if not errors:
                        limits={m:max(float(np.quantile(im[im>0],.995)),1) if (im>0).any() else 1 for m,im in images.items()}
                        cases.append(dict(example_id=selected.example_id,patient=patient,fov=fov,u=u,h=h,
                            mask=mask,images=images,limits=limits,pair_region=region(mask,[u,h],18),
                            hev_region=region(mask,[h],12),u_mean=um,h_mean=hm))
            records.append(rec)
        print(f'{fov}: {len(batch)} unique pairs',flush=True)
    table=pd.DataFrame(records).sort_values(['motif','patient','FOV','unidentified_cell_id','hev_cell_id'])
    table.to_csv(OUT/'all_unidentified_HEV_pairs.csv',index=False)
    summaries=[]
    for key,batch in table.groupby(['motif','motif_id','patient','FOV','unidentified_cell_id']):
        summaries.append(dict(zip(['motif','motif_id','patient','FOV','unidentified_cell_id'],key),
            Group=batch.Group.iloc[0] if 'Group' in batch else '',
            unidentified_mean_MECA79=batch.unidentified_mean_MECA79.iloc[0],
            annotated_HEV_neighbor_ids=';'.join(batch.hev_cell_id.astype(str)),
            corresponding_HEV_means=';'.join(str(x) for x in batch.hev_mean_MECA79),
            unique_HEV_neighbors=len(batch),motif_occurrences=int(batch.motif_occurrences.sum()),
            quality_status=';'.join(batch.quality_status.unique())))
    pd.DataFrame(summaries).to_csv(OUT/'one_row_per_unidentified_cell_per_motif.csv',index=False)
    display_columns=['motif','Group','patient','FOV','unidentified_cell_id','hev_cell_id',
                     'unidentified_mean_MECA79','hev_mean_MECA79','hev_over_unidentified_mean','motif_occurrences','quality_status']
    display=table[display_columns].rename(columns={
        'unidentified_mean_MECA79':'Mean MECA-79: unidentified', 'hev_mean_MECA79':'Mean MECA-79: annotated HEV',
        'hev_over_unidentified_mean':'HEV / unidentified'})
    page='''<!doctype html><meta charset="utf-8"><title>MECA-79: all motif-I/II pairs</title>
<style>body{font:14px/1.5 system-ui;margin:25px}table{border-collapse:collapse}td,th{padding:6px 9px;border:1px solid #ddd;text-align:right}th{position:sticky;top:0;background:#eef5f5}img{max-width:1400px;width:100%}</style>
<h1>MECA-79 in all motif-I/II unidentified–annotated-HEV pairs</h1>
<p>Original image means over each entire segmentation mask. No denoising, background subtraction, or intensity transformation. Compare paired cells within their FOV. These are existing annotations, not validated HEV identities. Duplicate motif occurrences are collapsed to unique cell pairs; multiple HEV neighbors remain separate rows.</p>
<p><a href="all_unidentified_HEV_pairs.csv">Complete pair table (CSV)</a> · <a href="one_row_per_unidentified_cell_per_motif.csv">One row per unidentified cell per motif (CSV)</a> · <a href="neighbors_with_segmentation.pdf">Neighbor segmentation panels (PDF)</a></p>'''
    page+=display.to_html(index=False,float_format=lambda x:f'{x:.4f}',na_rep='missing / undefined',escape=True)
    page+='<h2>Current figure examples: both segmentations</h2><p>Dashed white: unidentified. Solid cyan: annotated HEV. Merge: MECA-79 green, DNA blue. Separate MECA-79/CD31: green; DNA: grayscale. Bars: 20 pixels in pair views; 10 pixels in HEV close-ups. Display limits fixed per FOV/channel at the positive-pixel 99.5th percentile.</p><img src="neighbors_with_segmentation.png" alt="Unidentified cells and HEV neighbors with their segmentations">'
    (OUT/'index.html').write_text(page)
    cases.sort(key=lambda c:({'I':0,'II':1}[c['example_id'].split('_')[0]],int(c['example_id'].split('_')[1])))
    assert len(cases)==4
    titles=['Pair context','MECA-79: pair','MECA-79: HEV zoom','CD31: HEV zoom','DNA: HEV zoom']
    fig,axes=plt.subplots(4,5,figsize=(11,10.5))
    for i,case in enumerate(cases):
        for j,ax in enumerate(axes[i]):
            draw(ax,case,j)
            if i==0:ax.set_title(titles[j],fontsize=9)
        pos_label=f'{case["example_id"]} · P{int(case["patient"])} · {case["fov"]}\nU {case["u"]}: {case["u_mean"]:.3f}\nHEV {case["h"]}: {case["h_mean"]:.3f}'
        axes[i,0].set_ylabel(pos_label,rotation=0,labelpad=60,fontsize=8,va='center')
    fig.suptitle('Unidentified cells and their annotated HEV neighbors',fontsize=14,y=.98)
    fig.text(.53,.035,'Dashed white: unidentified mask  |  Solid cyan: annotated HEV mask  |  Values: whole-mask mean MECA-79',ha='center',fontsize=8)
    fig.text(.53,.015,'Existing annotations; these views do not establish HEV identity or exclude spillover. Bars: 20 px (pair), 10 px (zoom).',ha='center',fontsize=8)
    fig.subplots_adjust(left=.17,right=.99,bottom=.07,top=.93,wspace=.08,hspace=.22)
    for ext in ['png','pdf']:fig.savefig(OUT/f'neighbors_with_segmentation.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    with PdfPages(OUT/'individual_neighbor_panels.pdf') as pdf:
        for case in cases:
            fig,axes=plt.subplots(1,5,figsize=(12,3.6))
            for j,ax in enumerate(axes):draw(ax,case,j);ax.set_title(titles[j],fontsize=9)
            fig.suptitle(f'{case["example_id"]}: unidentified {case["u"]} (white dashed), annotated HEV {case["h"]} (cyan)',fontsize=11)
            fig.tight_layout(rect=(0,.04,1,.9));pdf.savefig(fig,dpi=300);plt.close(fig)
    counts=pd.DataFrame([dict(motif=motif,unique_pairs=len(batch),
        unique_unidentified_cells=len(batch[['patient','FOV','unidentified_cell_id']].drop_duplicates()),
        motif_occurrences=int(batch.motif_occurrences.sum())) for motif,batch in table.groupby('motif')]).set_index('motif')
    counts.to_csv(OUT/'coverage.csv')
    (OUT/'provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archives=archives,source_motifs=CONFIG,measurement='mean original MECA-79 TIFF pixels inside each original labeled mask',
        quality_counts=table.quality_status.value_counts().to_dict(),image_examples=current.example_id.tolist()),indent=2))
    (OUT/'README.md').write_text('''# MECA-79 motif-I/II pair table and neighbor segmentations

All saved motif-I/II hits are included, without restricting by spillover status or protein cluster. Motif I: unidentified cell3, annotated HEV cell4. Motif II: unidentified cell2, annotated HEV cell3. Repeated hits are collapsed to unique (motif, patient, FOV, unidentified ID, HEV ID) pairs, retaining the occurrence count. A cell with two HEV neighbors has two pair rows; one_row_per_unidentified_cell_per_motif.csv records the two IDs and their corresponding means without averaging them. Cell IDs are local to FOV; use patient/FOV/ID keys across fields.

Mean expression is the arithmetic mean of original MECA-79 TIFF intensities within the entire segmentation mask. Boundary and interior means use one-pixel binary erosion and are descriptive; erosion sensitivity alone is not spillover evidence. Fractions nonzero are not positivity calls. HEV/unidentified ratios are undefined if the unidentified mean is zero. No pseudocount is added. Missing archives, masks and table matches are reported in quality_status; they are never interpreted as negative staining. Annotated identities, dimensions and mask centroids are checked. A mask touching the image edge is recorded separately. No new segmentation or annotation is performed. Raw image means are not calibrated molecular abundances.

Images show the four current motif-I/II figure examples, including the replacement I_1. Both masks are outlined, with complete pair and HEV bounding regions plus context. Green is MECA-79 or CD31, DNA is blue in the composite and grayscale separately; white dashed is unidentified and solid cyan is annotated HEV. Existing HEV labels are not independent ground truth. Display limits are linear and fixed per whole FOV/channel at the 99.5th percentile of positive pixels; no smoothing or background subtraction. Bars are pixels because physical calibration has not been established.

MECA-79 staining need not uniformly fill a segmented cell. Published HEV studies show surface-associated/luminal staining, sometimes basal and cytoplasmic signal in tumors. A low whole-cell mean must be interpreted with localization, membrane coverage, staining controls and segmentation alignment. Absence of staining does not by itself distinguish technical failure, incomplete masks, other endothelial states, or annotation error. Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC8976767/ .

Reproduce: MPLCONFIGDIR=/tmp/cism-mpl OPENBLAS_NUM_THREADS=1 python analysis/marker_ownership/meca79_motif_pairs.py
''')
    print(counts.to_string(),flush=True)
    print(table.quality_status.value_counts().to_string(),flush=True)
    print(f'Complete: {OUT}',flush=True)


if __name__=='__main__':main()
