"""Paper-style marker ownership figure, supplied motif art, and blinded review packet."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor,as_completed
import gc
import hashlib
import html
import json
import textwrap

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis.marker_ownership import motif_marker_ownership as study
from analysis.marker_ownership import export_spillover_microscopy_panels as imaging

OUT=study.OUT
ART=Path('/home/omriavit/.codex/attachments/8726b497-da47-40e0-ab9d-f200ff00db14/codex-clipboard-dd2cb973-e683-4658-a2d7-f2d499b294f7.png')
PREVIOUS=study.ROOT/'analysis/outputs/unidentified_motif_I_IV_examples/marker_focused/example_manifest.csv'
COLORS={'HEV':np.array((.1,1,.1)),'DC':np.array((1,.15,.1)),'DNA':np.array((.15,.3,1))}
STATUS_COLORS={'reference_like_profile_low_risk':'#278448','spillover_consistent':'#c43030',
               'proposed_profile_not_supported':'#8c8c8c','unresolved_quality_or_reference':'#d8a94d',
               'unresolved_interface':'#bd803b','unresolved_competing_lineage':'#9375ab'}
SHORT={'reference_like_profile_low_risk':'Reference-like; low risk','spillover_consistent':'Spillover-consistent',
       'proposed_profile_not_supported':'Proposed profile not supported','unresolved_quality_or_reference':'Unresolved: quality/reference',
       'unresolved_interface':'Unresolved: interface','unresolved_competing_lineage':'Unresolved: competing lineage'}
LABEL={'HLA-DR-DP-DQ':'HLA-II','MECA-79':'MECA-79 / PNAd','dsDNA':'DNA'}


def graphics():
    directory=OUT/'graphics';directory.mkdir(exist_ok=True)
    with Image.open(ART) as source:
        regions={'I':(200,40,659,460),'II':(80,460,659,820),'III':(80,820,650,1210),'IV':(80,1215,650,1636)}
        for roman,box in regions.items():
            crop=source.crop(box).convert('RGBA')
            # The supplied transparent sheet includes detached Roman numerals
            # outside the motif drawings. Keep the connected drawing, not those
            # cropped fragments of adjacent labels.
            pixels=np.array(crop)
            components,n=ndi.label(pixels[:,:,3]>0,structure=np.ones((3,3),bool))
            if n:
                areas=np.bincount(components.ravel());areas[0]=0
                pixels[components!=int(areas.argmax()),3]=0
                crop=Image.fromarray(pixels)
            background=Image.new('RGBA',crop.size,'white');background.alpha_composite(crop)
            background.convert('RGB').save(directory/f'motif_{roman}.png')
    (directory/'provenance.json').write_text(json.dumps(dict(source=str(ART),sha256=hashlib.sha256(ART.read_bytes()).hexdigest(),
        crops=regions,operation='Crop supplied artwork; retain the connected motif drawing to omit detached adjacent Roman-label fragments; composite transparency onto white. No AI generation or redrawing.'),indent=2))


def choose_review(outcomes,links):
    rng=np.random.default_rng(72069)
    candidates=outcomes[outcomes.threshold.eq(95)].copy()
    selected=[]
    for kind in ['HEV','DC']:
        pool=candidates[candidates.candidate_kind.eq(kind)]
        chosen=pool.iloc[rng.choice(len(pool),min(25,len(pool)),replace=False)].copy()
        chosen['review_stratum']='random_candidate';selected.append(chosen)
    selected=pd.concat(selected,ignore_index=True)
    controls=[];used=set()
    for _,row in selected.iterrows():
        pool=links[links.fov.eq(row.fov)&links.target_cell_id.eq(row.cell_id)]
        pool=pool.iloc[rng.permutation(len(pool))]
        for link in pool.itertuples():
            key=(row.fov,int(link.control_cell_id))
            if key in used:continue
            used.add(key)
            d=row.to_dict();d['cell_id']=int(link.control_cell_id);d['motif']='control';d['review_stratum']='FOV_matched_control'
            controls.append(d);break
    if len(controls)!=len(selected):raise ValueError('Could not construct all FOV-matched review controls')
    control_frame=pd.DataFrame(controls)
    needed=['patient number','fov','cell_id','candidate_kind','motif','review_stratum']
    review=pd.concat([selected[needed],control_frame[needed]],ignore_index=True)
    random_keys=set(map(tuple,review[study.KEY+['candidate_kind']].values.tolist()))
    flagged=candidates[candidates.status.eq('spillover_consistent')].copy()
    flagged=flagged[~flagged.apply(lambda r:tuple(r[k] for k in study.KEY+['candidate_kind']) in random_keys,axis=1)]
    if len(flagged)>25:flagged=flagged.iloc[rng.choice(len(flagged),25,replace=False)]
    flagged['review_stratum']='deliberately_flagged'
    review=pd.concat([review,flagged[needed]],ignore_index=True)
    review=review.iloc[rng.permutation(len(review))].reset_index(drop=True)
    review['review_id']=[f'R{i:03d}' for i in range(1,len(review)+1)]
    review.to_csv(OUT/'blinded_review_key.csv',index=False)
    form=review[['review_id']].copy()
    for field in ['proposed_identity_plausibility','spillover','segmentation','uncertainty','reviewer','notes']:form[field]=''
    form.to_csv(OUT/'blinded_review_form.csv',index=False)
    return review


def choose_reference_review():
    columns=study.KEY+['marker','role','pred_mapped','reference_local_max','whole_marker_supported']
    d=pd.read_csv(OUT/'per_cell_per_marker.csv.gz',usecols=columns)
    rng=np.random.default_rng(82069);selected=[]
    for kind,markers in study.CORE.items():
        annotation='hevs' if kind=='HEV' else 'DC'
        x=d[d.role.eq('annotated_reference')&d.marker.isin(markers)&d.pred_mapped.eq(annotation)]
        pool=x.groupby(study.KEY).agg(markers=('marker','nunique'),local=('reference_local_max','all'),supported=('whole_marker_supported','all')).reset_index()
        pool=pool[pool.markers.eq(2)&pool.local&pool.supported]
        pool=pool.iloc[rng.permutation(len(pool))].drop_duplicates('patient number').head(10).copy()
        pool['candidate_kind']=kind;pool['render_role']='reference';pool['main_example']=False
        pool['motif']='annotated reference';pool['status']='provisional_reference'
        pool['render_id']=[f'reference_{kind}_{int(r["patient number"])}_{r.fov}_{int(r.cell_id)}' for _,r in pool.iterrows()]
        selected.append(pool)
    result=pd.concat(selected,ignore_index=True)
    result['human_validation']='';result['notes']=''
    result.to_csv(OUT/'provisional_reference_review.csv',index=False)
    return result


def crop_region(mask,ids,padding=10):
    y,x=np.where(np.isin(mask,ids))
    if not len(y):raise ValueError('Missing requested mask')
    half=max(y.max()-y.min(),x.max()-x.min())/2+padding
    return imaging.bounds((y.min()+y.max())/2,(x.min()+x.max())/2,half,mask.shape)


def make_case(mask,images,limits,row,partners):
    cid=int(row['cell_id']);kind=row['candidate_kind']
    zoom=crop_region(mask,[cid],11)
    context=crop_region(mask,[cid]+partners,20)
    def cut(sl):return dict(mask=mask[sl].copy(),images={m:im[sl].copy() for m,im in images.items()})
    return dict(row=row,cell_id=cid,kind=kind,partners=partners,limits=limits,zoom=cut(zoom),context=cut(context))


def draw(ax,case,column,blind=False):
    kind=case['kind'];region=case['context'] if column==0 else case['zoom']
    marker=([None]+study.CORE[kind]+(['CD45'] if kind=='HEV' else ['CD3'])+['dsDNA'])[column]
    if marker is None:
        rgb=np.zeros(region['mask'].shape+(3,))
        for m,color in [(study.CORE[kind][1] if kind=='HEV' else study.CORE[kind][0],COLORS[kind]),('dsDNA',COLORS['DNA'])]:
            rgb+=np.clip(region['images'][m].astype(float)/case['limits'][m],0,1)[:,:,None]*color
        pixels=np.clip(rgb,0,1)
    else:
        color=COLORS[kind] if column in [1,2] else np.ones(3)
        pixels=np.clip(region['images'][marker].astype(float)/case['limits'][marker],0,1)[:,:,None]*color
    ax.imshow(pixels,interpolation='nearest')
    imaging.outline(ax,region['mask'],case['cell_id'],width=.55)
    for cid in case['partners']:
        binary=region['mask']==cid
        if binary.any() and (~binary).any():ax.contour(binary.astype(float),levels=[.5],colors=['#00e5ff'],linewidths=.5)
    h,w=region['mask'].shape;length=20 if column==0 else 10
    if length<w*.8:ax.plot([w*.07,w*.07+length],[h*.91,h*.91],color='white',lw=1.1)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    return 'Merge' if marker is None else LABEL.get(marker,marker)


def single_figure(case,blind=False):
    fig,axes=plt.subplots(1,5,figsize=(10,3))
    for j,ax in enumerate(axes):ax.set_title(draw(ax,case,j,blind),fontsize=9)
    row=case['row']
    if blind:
        title=f'Review {row["review_id"]}'
        subtitle='Assess staining ownership, segmentation, and plausible identity. No motif membership or computational scores are shown.'
    else:
        title=f'Motif {row.get("motif", "reference")} · P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]}'
        subtitle=SHORT.get(row.get('status',''),'Provisional annotated reference; visual validation required')
    fig.suptitle(title,fontsize=11,y=.98);fig.text(.5,.86,subtitle,ha='center',fontsize=8)
    fig.text(.5,.025,'Dashed white: focal mask. Solid cyan: motif partner (if shown). Bars: context 20 px; close-ups 10 px. DNA is context.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.015,right=.99,bottom=.13,top=.80,wspace=.08)
    return fig


def render_fov(frame,requested,selected):
    fov=frame.iloc[0].fov
    with ZipFile(study.DATA/'raw_data'/f'{fov}.zip') as archive:
        mask=imaging.read_image(archive,'TIFs/segmentation_labels.tiff')
        markers=list(dict.fromkeys([m for kind in requested.candidate_kind.unique() for m in study.CORE[kind]+(['CD45'] if kind=='HEV' else ['CD3'])+['dsDNA']]))
        images={m:imaging.read_image(archive,f'TIFs/{m}.tif') for m in markers}
    limits={m:max(float(np.quantile(im[im>0],.995)),1) if (im>0).any() else 1 for m,im in images.items()}
    cases=[]
    for row in requested.to_dict('records'):
        partners=[]
        if row.get('render_role')=='candidate':
            config=next(c for c in study.CONFIG if c[0]==row['motif'])
            hits=pd.read_csv(study.HITS/f'{config[1]}_hits_wide.csv')
            hit=hits[hits.FOV.eq(fov)&hits[config[2]].eq(row['cell_id'])].iloc[0]
            # Prefer the previously displayed instance for the eight existing examples.
            known=selected[selected.fov.eq(fov)&selected.cell_id.eq(row['cell_id'])&selected.roman.eq(row['motif'])]
            instance=json.loads(known.iloc[0].selected_instance_json) if len(known) else {f'cell{i}':int(hit[f'cell{i}']) for i in range(1,5)}
            columns=['cell4'] if row['motif']=='I' else ['cell3'] if row['motif']=='II' else ['cell4'] if row['motif']=='III' else ['cell2','cell3']
            partners=[int(instance[c]) for c in columns]
        # Blinded images show only focal outlines; labels/partner types are hidden.
        if row.get('render_role')=='review':partners=[]
        case=make_case(mask,images,limits,row,partners)
        target=OUT/'rendered'/f'{row["render_id"]}.png'
        fig=single_figure(case,row.get('render_role')=='review');fig.savefig(target,dpi=200);plt.close(fig)
        if row.get('main_example',False):cases.append(case)
    return cases


def main_figure(cases):
    fig=plt.figure(figsize=(7.2,6.6))
    for k,roman in enumerate(['I','II','III','IV']):
        pair=sorted([c for c in cases if c['row']['motif']==roman],key=lambda c:c['row']['example_id'])
        assert len(pair)==2
        left=.02+(k%2)*.50;bottom=.55 if k<2 else .10
        ax=fig.add_axes([left,bottom+.30,.15,.12]);ax.imshow(Image.open(OUT/'graphics'/f'motif_{roman}.png'));ax.axis('off')
        fig.text(left+.17,bottom+.40,f'{"ABCD"[k]}  Motif {roman}',fontsize=9,weight='bold')
        kind='HEV' if k<2 else 'DC'
        fig.text(left+.17,bottom+.367,'HEV-like assessment' if kind=='HEV' else 'DC/APC-like assessment',fontsize=6,color='#21853c' if kind=='HEV' else '#be3333')
        fig.text(left+.17,bottom+.343,'Merge: MECA-79 green; DNA blue' if kind=='HEV' else 'Merge: CD11c red; DNA blue',fontsize=4.8)
        grid=fig.add_gridspec(2,5,left=left,right=left+.46,bottom=bottom,top=bottom+.29,wspace=.065,hspace=.43)
        for i,case in enumerate(pair):
            for j in range(5):
                ax=fig.add_subplot(grid[i,j]);title=draw(ax,case,j)
                if i==0:ax.set_title('MECA-79' if title=='MECA-79 / PNAd' else title,fontsize=6,pad=2)
            pos=fig.axes[-5].get_position();row=case['row']
            fig.text(left,pos.y0-.012,f'P{int(row["patient number"])} · {row["fov"]} · cell {case["cell_id"]}',fontsize=4.9)
            fig.text(left,pos.y0-.029,SHORT[row['status']],fontsize=4.9,color=STATUS_COLORS[row['status']])
    fig.text(.02,.045,'Dashed white: unidentified mask. Cyan: annotated motif partner. Bars: merge 20 px; close-ups 10 px.',fontsize=5.5)
    fig.text(.02,.022,'Same eight examples as the previous revision; statuses are exploratory. Question marks retain the provisional identities.',fontsize=5.3,color='#555555')
    return fig


def summary_figure(candidates,bootstrap):
    data=candidates[candidates.threshold.eq(95)]
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    ax=axes[0,0];bottom=np.zeros(4)
    for status,color in STATUS_COLORS.items():
        counts=np.array([int(((data.motif==m)&(data.status==status)).sum()) for m in ['I','II','III','IV']])
        totals=np.array([int((data.motif==m).sum()) for m in ['I','II','III','IV']])
        ax.bar(['I','II','III','IV'],counts/totals*100,bottom=bottom,color=color,label=SHORT[status])
        bottom+=counts/totals*100
    ax.set_ylabel('Candidate cells (%)');ax.set_title('A  All cells retained in the denominator',loc='left',fontsize=11)
    ax.legend(fontsize=6,loc='upper left',bbox_to_anchor=(0,-.19),frameon=False)
    ax=axes[0,1]
    for i,motif in enumerate(['I','II','III','IV']):
        d=data[data.motif.eq(motif)]
        ax.bar(i-.18,d.whole_profile_supported.mean()*100,width=.34,color='#77a7ca',label='Whole-mask core profile' if i==0 else None)
        ax.bar(i+.18,d.noninterface_profile_supported.mean()*100,width=.34,color='#3c7561',label='Profile away from interface' if i==0 else None)
    ax.set_xticks(range(4),['I','II','III','IV']);ax.set_ylabel('Cells meeting provisional reference rules (%)')
    ax.set_title('B  Does the proposed profile persist?',loc='left',fontsize=11);ax.legend(fontsize=7,frameon=False)
    ax=axes[1,0]
    for j,metric in enumerate(['risk_difference','support_difference']):
        d=bootstrap[bootstrap.threshold.eq(95)&bootstrap.metric.eq(metric)].set_index('motif').reindex(['I','II','III','IV'])
        y=np.arange(4)+j*.15-.075
        ax.errorbar(d.patient_balanced_mean*100,y,xerr=np.vstack([(d.patient_balanced_mean-d.ci_low)*100,(d.ci_high-d.patient_balanced_mean)*100]),fmt='o',capsize=3,
            label='Spillover-consistent' if j==0 else 'Reference-like profile')
    ax.axvline(0,color='#888888',lw=.8);ax.set_yticks(range(4),['I','II','III','IV']);ax.invert_yaxis()
    ax.set_xlabel('Candidate minus FOV-matched controls (percentage points)')
    ax.set_title('C  Patient bootstrap, 95% intervals',loc='left',fontsize=11);ax.legend(fontsize=7,frameon=False)
    ax=axes[1,1]
    for m in ['I','II','III','IV']:
        values=[candidates[candidates.motif.eq(m)&candidates.threshold.eq(q)].status.eq('spillover_consistent').mean()*100 for q in [90,95,99]]
        ax.plot([90,95,99],values,'o-',label=f'Motif {m}')
    ax.set_xticks([90,95,99]);ax.set_xlabel('Exploratory matched-control percentile');ax.set_ylabel('Spillover-consistent candidates (%)')
    ax.set_title('D  Threshold sensitivity',loc='left',fontsize=11);ax.legend(fontsize=7,frameon=False)
    fig.suptitle('Population evidence across all 167 unidentified cells',fontsize=14,y=.98)
    fig.text(.5,.015,'Provisional annotated references and exploratory thresholds; these outcomes are not validated cell identities or an enrichment reanalysis.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.95),h_pad=5,w_pad=2)
    return fig


def validation_figure():
    mask=np.zeros((55,65),int);mask[15:40,12:36]=1;mask[15:40,36:53]=2
    own=mask==1;boundary=own&~ndi.binary_erosion(own,structure=np.ones((3,3),bool),iterations=2)
    y,x=np.where(boundary);angle=np.abs(np.arctan2(y-y.mean(),x-x.mean()))
    results=[];examples=[]
    for fraction in [.125,.25,.5,.75,1.]:
        locations=np.flatnonzero(boundary.ravel());selected=locations[np.argsort(angle)[:int(np.ceil(len(locations)*fraction))]]
        image=np.zeros(mask.shape);image.ravel()[selected]=1000/len(selected);image[mask==2]=50
        measured=study.boundary_measurements(image,mask,1,2,.5,.01)
        results.append(dict(membrane_fraction=fraction,whole_mean=float(image[own].mean()),**measured))
        if fraction in [.125,1.]:examples.append((fraction,image))
    pd.DataFrame(results).to_csv(OUT/'synthetic_membrane_coverage.csv',index=False)
    fig,axes=plt.subplots(1,4,figsize=(11,3.3))
    for ax,(fraction,image) in zip(axes[:2],examples):
        ax.imshow(image,cmap='magma',vmin=0,vmax=50,interpolation='nearest');imaging.outline(ax,mask,1)
        ax.set_title(f'{fraction*100:g}% membrane coverage',fontsize=9);ax.axis('off')
    d=pd.DataFrame(results)
    axes[2].plot(d.membrane_fraction*100,d.whole_mean,'o-');axes[2].set_xlabel('Membrane coverage (%)');axes[2].set_ylabel('Whole-cell mean')
    axes[3].plot(d.membrane_fraction*100,d.selective_interface_log2_loss,'o-');axes[3].set_xlabel('Membrane coverage (%)');axes[3].set_ylabel('Selective interface loss (log2)')
    fig.suptitle('Measurement validation: equal total signal, different spatial patterns',fontsize=12)
    fig.text(.5,.02,'Inspired by CellSighter Fig. 3G–H. This validates measurement behavior; it is not a replication of classifier accuracy.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.93))
    return fig


def pdf_from_images(paths,path,title):
    with PdfPages(path) as pdf:
        for p in paths:
            with Image.open(p) as im:pixels=np.array(im)
            fig=plt.figure(figsize=(11,4));ax=fig.add_axes([0,0,1,1]);ax.imshow(pixels);ax.axis('off')
            pdf.savefig(fig,dpi=200);plt.close(fig)


def main():
    (OUT/'rendered').mkdir(exist_ok=True)
    graphics()
    candidates=pd.read_csv(OUT/'candidate_cell_outcomes.csv')
    q95=candidates[candidates.threshold.eq(95)].copy()
    links=pd.read_csv(OUT/'matched_control_links.csv.gz');selected=pd.read_csv(PREVIOUS)
    review=choose_review(candidates,links)
    references=choose_reference_review()
    rows=[]
    for row in q95.to_dict('records'):
        row['render_role']='candidate';row['render_id']=f'candidate_{row["motif"]}_{int(row["patient number"])}_{row["fov"]}_{int(row["cell_id"])}'
        known=selected[selected.fov.eq(row['fov'])&selected.cell_id.eq(row['cell_id'])&selected.roman.eq(row['motif'])]
        row['main_example']=bool(len(known));row['example_id']=known.iloc[0].example_id if len(known) else ''
        rows.append(row)
    for row in review.to_dict('records'):
        row['render_role']='review';row['render_id']=row['review_id'];row['main_example']=False;rows.append(row)
    rows.extend(references.to_dict('records'))
    requests=pd.DataFrame(rows);requests.to_csv(OUT/'render_manifest.csv',index=False)
    spatial=pd.read_csv(study.DATA/'mapped_cell_types.csv');cases=[]
    tasks=[]
    for (patient,fov),d in requests.groupby(['patient number','fov']):
        frame=spatial[spatial['patient number'].eq(patient)&spatial.fov.eq(fov)]
        tasks.append((frame,d,selected))
    with ProcessPoolExecutor(max_workers=2) as executor:
        jobs={executor.submit(render_fov,*task):task[0].iloc[0].fov for task in tasks}
        for job in as_completed(jobs):cases.extend(job.result());print(f'{jobs[job]}: microscopy rendered',flush=True)
    assert len(cases)==8
    fig=main_figure(cases)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'motif_ownership_figure.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    figures=[main_figure(cases),summary_figure(candidates,pd.read_csv(OUT/'patient_bootstrap.csv')),validation_figure()]
    with PdfPages(OUT/'marker_ownership_results.pdf') as pdf:
        for i,fig in enumerate(figures):
            fig.savefig(OUT/f'results_page_{i+1}.png',dpi=280,facecolor='white');pdf.savefig(fig,dpi=300);plt.close(fig)
    atlas=[OUT/'rendered'/f'{r.render_id}.png' for r in requests[requests.render_role.eq('candidate')].itertuples()]
    pdf_from_images(atlas,OUT/'all_candidate_cells.pdf','All candidate cells')
    pdf_from_images([OUT/'rendered'/f'{r.review_id}.png' for r in review.itertuples()],OUT/'blinded_review.pdf','Blinded review')
    pdf_from_images([OUT/'rendered'/f'{r.render_id}.png' for r in references.itertuples()],OUT/'provisional_reference_review.pdf','Provisional annotated reference review')
    counts=q95.groupby(['motif','status']).size().unstack(fill_value=0)
    page='''<!doctype html><meta charset="utf-8"><title>Marker ownership in unidentified motif cells</title>
<style>body{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}td,th{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}</style>
<h1>Marker ownership in unidentified motif cells</h1><p>All 167 unique candidates across 56 FOVs. Proposed HEV-like and DC/APC-like profiles are assessed separately from spillover-consistent patterns. No labels are changed. Provisional annotated references and exploratory thresholds require visual validation.</p>
<p><a href="marker_ownership_results.pdf">Three-page results PDF</a> · <a href="motif_ownership_figure.svg">Editable main figure</a> · <a href="all_candidate_cells.pdf">All candidate images</a> · <a href="candidate_cell_outcomes.csv">Per-cell outcomes</a> · <a href="blinded_review.pdf">Blinded review images</a> · <a href="blinded_review_form.csv">Review form</a> · <a href="provisional_reference_review.pdf">Annotated reference review</a></p>
<p>Reviewers should not open blinded_review_key.csv before completing the form. Random candidates and matched controls must be analyzed separately from deliberately flagged additions. No review verdicts have been filled automatically.</p>'''
    page+=counts.to_html()+''.join(f'<img src="results_page_{i}.png" alt="Results page {i}">' for i in [1,2,3])
    page+='<h2>Interpretation</h2><p>Reference-like means that the measured marker profile meets provisional within-FOV reference thresholds and persists away from the suspected interface. It does not establish an HEV/DC identity. Profile-not-supported is not proof of spillover. Unresolved measurements remain in the denominator. Statistical motif enrichment has not been rerun; occurrence retention is descriptive.</p>'
    (OUT/'index.html').write_text(page)
    (OUT/'figure_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        graphic=str(ART),main_examples=selected[study.KEY+['example_id']].to_dict('records'),
        review_random_candidates=int(review.review_stratum.eq('random_candidate').sum()),
        review_matched_controls=int(review.review_stratum.eq('FOV_matched_control').sum()),
        deliberately_flagged=int(review.review_stratum.eq('deliberately_flagged').sum()),
        provisional_reference_examples=len(references)),indent=2))
    print(f'Complete: {OUT}',flush=True)


if __name__=='__main__':main()
