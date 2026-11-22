"""Continuous marker evidence and interpretation alongside the exploratory screen."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
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

from analysis.marker_ownership import motif_marker_ownership as study


def continuous_figure():
    columns=study.KEY+['marker','role','pred_mapped','whole_mean','away_boundary_mean','pseudocount']
    d=pd.read_csv(study.OUT/'per_cell_per_marker.csv.gz',usecols=columns)
    members=pd.read_csv(study.OUT/'motif_membership.csv')
    family=members.set_index(study.KEY).candidate_kind
    patient_rows=[];all_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,marker in zip(axes.flat,sum(study.CORE.values(),[])):
        kind=next(k for k,v in study.CORE.items() if marker in v);annotation='hevs' if kind=='HEV' else 'DC'
        x=d[d.marker.eq(marker)].copy()
        x=x[(~x.role.eq('annotated_reference')|x.pred_mapped.eq(annotation))]
        candidate_kind=x.set_index(study.KEY).index.map(family)
        x=x[~x.role.eq('candidate')|np.asarray(candidate_kind==kind)]
        references=x[x.role.eq('annotated_reference')]
        medians=references.groupby(['patient number','fov'])[['whole_mean','away_boundary_mean']].median().rename(
            columns={'whole_mean':'reference_whole_median','away_boundary_mean':'reference_away_median'})
        x=x.merge(medians,on=['patient number','fov'],how='left')
        # These distributions do not use the local-maximum or positivity cutoffs.
        x['whole_log2_reference_ratio']=np.log2((x.whole_mean+x.pseudocount)/(x.reference_whole_median+x.pseudocount))
        x['away_log2_reference_ratio']=np.log2((x.away_boundary_mean+x.pseudocount)/(x.reference_away_median+x.pseudocount))
        all_rows.append(x)
        patient=x.groupby(['patient number','role'])[['whole_log2_reference_ratio','away_log2_reference_ratio']].median().reset_index()
        patient['marker']=marker;patient_rows.append(patient)
        rng=np.random.default_rng(92069)
        for i,role in enumerate(['candidate','matched_unidentified_control','annotated_reference']):
            subset=patient[patient.role.eq(role)]
            for j,(metric,color,label) in enumerate([('whole_log2_reference_ratio','#4a83af','Whole mask'),('away_log2_reference_ratio','#42906c','Away boundary')]):
                y=subset[metric].dropna().to_numpy();offset=-.12 if j==0 else .12
                ax.scatter(i+offset+rng.uniform(-.045,.045,len(y)),y,s=11,color=color,alpha=.45,label=label if i==0 else None)
                if len(y):ax.plot([i+offset-.065,i+offset+.065],[np.median(y)]*2,color=color,lw=2)
        ax.axhline(0,color='#999999',ls='--',lw=.8)
        ax.set_xticks([0,1,2],['Candidates','Matched U','Annotated\nreferences'])
        ax.set_title(marker.replace('HLA-DR-DP-DQ','HLA-II'),loc='left',fontsize=11)
        ax.set_ylabel('log2(signal / same-FOV reference median)')
        ax.legend(fontsize=7,frameon=False)
    pd.concat(all_rows).to_csv(study.OUT/'continuous_marker_evidence.csv.gz',index=False)
    pd.concat(patient_rows).to_csv(study.OUT/'continuous_marker_patient_medians.csv',index=False)
    fig.suptitle('Marker evidence without the provisional profile thresholds',fontsize=14)
    fig.text(.5,.018,'Dots: patient medians; bars: median across patients. Separate normalization for whole-mask and away-boundary means. References are provisional.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.95),h_pad=2,w_pad=1.5)
    return fig


def interpretation_figure():
    fig=plt.figure(figsize=(8.5,11));y=.94
    def paragraph(text,size=9,weight='normal',gap=.020):
        nonlocal y
        lines=textwrap.fill(text,108 if size<=10 else 78)
        fig.text(.07,y,lines,fontsize=size,weight=weight,va='top',linespacing=1.5)
        y-=len(lines.splitlines())*.0175+gap
    paragraph('Interpretation, marker choices, and limits',15,'bold',.045)
    paragraph('All 167 unique unidentified cells in motifs I–IV were measured across 56 FOVs. The primary 95th-percentile screen finds 1 spillover-consistent case, 61 unresolved cases, and 105 below the strict provisional marker-profile rules. No cell passes every profile, ownership, and quality requirement. These are exploratory screen outcomes, not diagnoses of contamination or definitive cell identities.')
    paragraph('Only two candidates meet the primary whole-mask two-marker profile, and neither meets the non-interface rule. Relaxing the reference quantile to 5% and omitting the negative-reference cutoff increases the whole-profile count to 12, but none meets the complete non-interface rule. Full sensitivity results are supplied in profile_threshold_sensitivity.csv.')
    paragraph('The profile rules are stringent, particularly for DC/APC-like cells. Among the provisional annotated reference pools, 2,421/8,636 HEV references and 146/9,732 DC references meet the primary non-interface two-marker rule. This is an in-sample diagnostic, not independent accuracy validation. Thus, failing the rule does not establish that a cell is biologically not an HEV or DC. The continuous marker distributions are provided separately from these cutoffs.')
    paragraph('Channel justification: CellSighter Fig. 2E and its melanoma lymph-node Methods support CD31 plus PNAd (our MECA-79 stain) for the HEV hypothesis. CD11c and HLA-II test a DC/APC-like profile; neither uniquely establishes a DC subtype. CD45 checks an alternative immune profile in HEV-like candidates; CD3 checks an alternative T-cell profile in DC-like candidates. CD20 and macrophage-associated channels are assessed numerically. DNA is nuclear context; SMA is vascular/stromal context, not independent HEV identity evidence.')
    paragraph('Spatial-method alignment: CellSighter Fig. 3A–B measures real-image overlap; Fig. 3C–E and G–H test neighboring patches and membrane coverage. CellTune Fig. 5k–m and Supplementary S1/S4 motivate cell-versus-neighbor features, source-aware interpretation, and segmentation checks. We compare actual interface removal with equal-area sham boundary removal; erosion sensitivity alone is never a flag. We have not reproduced either classifier or claimed its validated performance for our screen.')
    paragraph('The supplied motif artwork retains the unidentified-cell question marks. The main figure keeps the same eight examples as the previous revision, with I_1 = P14/FOV132/cell7775. Provisional annotations remain unchanged. Scale bars are pixels; raw means are untransformed, while image display limits are fixed per FOV/channel. The separate review packet contains 50 random candidates and 50 matched unidentified controls. Existing annotated references are supplied for inspection. No manual verdicts have been fabricated.')
    paragraph('Next evidence needed: inspect the reference and candidate segmentations and complete the blinded review before claiming validated identities or dismissing spillover. Motif-occurrence retention is descriptive; statistical motif enrichment has not been rerun because no new labels have been validated.')
    fig.text(.07,.07,'CellSighter — Nature Communications (2023)',fontsize=9,url='https://www.nature.com/articles/s41467-023-40066-7',color='#2b5d8a')
    fig.text(.07,.045,'CellTune — Nature Methods (2026), including Supplementary Information',fontsize=9,url='https://www.nature.com/articles/s41592-026-03162-2#MOESM1',color='#2b5d8a')
    return fig


def main():
    fig=continuous_figure();fig.savefig(study.OUT/'continuous_marker_evidence.png',dpi=280);fig.savefig(study.OUT/'continuous_marker_evidence.pdf');plt.close(fig)
    notes=interpretation_figure();notes.savefig(study.OUT/'interpretation.png',dpi=180);plt.close(notes)
    paths=[study.OUT/'results_page_1.png',study.OUT/'results_page_2.png',study.OUT/'continuous_marker_evidence.png',
           study.OUT/'results_page_3.png',study.OUT/'interpretation.png']
    with PdfPages(study.OUT/'marker_ownership_report.pdf',metadata={'Title':'Motif-associated unidentified cells: marker ownership and interpretation'}) as pdf:
        for path in paths:
            with Image.open(path) as image: pixels=np.array(image)
            width,height=pixels.shape[1],pixels.shape[0]
            fig=plt.figure(figsize=(10,10*height/width));ax=fig.add_axes([0,0,1,1]);ax.imshow(pixels,interpolation='nearest');ax.axis('off')
            pdf.savefig(fig,dpi=300);plt.close(fig)
    page=study.OUT/'index.html';text=page.read_text()
    if 'Full five-page report' not in text:
        text=text.replace('<h1>Marker ownership in unidentified motif cells</h1>',
        '<h1>Marker ownership in unidentified motif cells</h1><p><strong>The current data do not establish the proposed HEV/DC identities or exclude spillover. These are strict exploratory screens; reference review and blinded validation are pending.</strong></p><p><a href="marker_ownership_report.pdf">Full five-page report with continuous marker evidence and interpretation</a> · <a href="profile_threshold_sensitivity.csv">Profile threshold sensitivity</a> · <a href="provisional_reference_diagnostics.csv">Reference-screen diagnostics</a></p>')
    if '<h2>Continuous marker evidence</h2>' not in text:
        text+='<h2>Continuous marker evidence</h2><img src="continuous_marker_evidence.png" alt="Patient-level marker evidence without profile thresholds"><h2>Interpretation and channel justification</h2><img src="interpretation.png" alt="Methods and interpretation">'
    page.write_text(text)
    (study.OUT/'report_provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_images={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},pages=len(paths),
        reference_diagnostic='In-sample provisional annotations; no independent sensitivity estimate.'),indent=2))
    print(f'Complete: {study.OUT}/marker_ownership_report.pdf',flush=True)


if __name__=='__main__':main()
