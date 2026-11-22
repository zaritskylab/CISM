"""Sensitivity of provisional marker-profile rules; no change to primary outcomes."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

import numpy as np
import pandas as pd
from analysis.marker_ownership import motif_marker_ownership as study


def main():
    columns=study.KEY+['marker','role','pred_mapped','geometry_valid','reference_local_max','whole_mean',
        'away_boundary_mean','away_positive_fraction','away_positive_pixels','retained_cell_mean',
        'negative_reference_mean_q95','pseudocount']
    d=pd.read_csv(study.OUT/'per_cell_per_marker.csv.gz',usecols=columns)
    d=d[d.marker.isin(sum(study.CORE.values(),[]))].copy()
    d['candidate_kind']=d.marker.map({m:kind for kind,markers in study.CORE.items() for m in markers})
    geometry=pd.read_csv(study.OUT/'cell_geometry.csv.gz',usecols=study.KEY+['bright_dna_components'])
    membership=pd.read_csv(study.OUT/'motif_membership.csv')[study.KEY+['motif','candidate_kind']]
    results=[];reference_results=[]
    for q in [.05,.10,.25]:
        for negative_cutoff in [False,True]:
            parts=[]
            for (patient,fov,marker),x in d.groupby(['patient number','fov','marker']):
                refs=x[x.reference_local_max]
                measured=x[x.role.isin(['candidate','annotated_reference'])].copy()
                whole=away=cov=retained=np.nan
                if len(refs)>=5:
                    neg=float(x.negative_reference_mean_q95.iloc[0])
                    whole=max(float(refs.whole_mean.quantile(q)),neg if negative_cutoff and np.isfinite(neg) else 0.,float(x.pseudocount.iloc[0]*5))
                    away=float(refs.away_boundary_mean.quantile(q))
                    cov=max(.05,float(refs.away_positive_fraction.quantile(q)))
                    retained=float(refs.retained_cell_mean.quantile(q))
                measured['whole_supported']=measured.geometry_valid&measured.whole_mean.ge(whole)
                measured['away_supported']=measured.whole_supported&measured.away_boundary_mean.ge(away)&measured.away_positive_fraction.ge(cov)&measured.retained_cell_mean.ge(retained)&measured.away_positive_pixels.ge(3)
                parts.append(measured[study.KEY+['candidate_kind','role','pred_mapped','marker','whole_supported','away_supported']])
            scored=pd.concat(parts)
            scored=scored.groupby(study.KEY+['candidate_kind','role','pred_mapped']).agg(
                markers=('marker','nunique'),whole_supported=('whole_supported','all'),away_supported=('away_supported','all')).reset_index()
            scored=scored[scored.markers.eq(2)].merge(geometry,on=study.KEY,validate='many_to_one')
            candidate=scored[scored.role.eq('candidate')].merge(membership,on=study.KEY+['candidate_kind'],validate='one_to_one')
            for motif,x in candidate.groupby('motif'):
                results.append(dict(reference_quantile=q,use_negative_q95_cutoff=negative_cutoff,motif=motif,
                    candidates=len(x),whole_core_profile=int(x.whole_supported.sum()),noninterface_core_profile=int(x.away_supported.sum()),
                    noninterface_profile_and_one_DNA=int((x.away_supported&x.bright_dna_components.eq(1)).sum())))
            for kind,annotation in [('HEV','hevs'),('DC','DC')]:
                x=scored[scored.role.eq('annotated_reference')&scored.candidate_kind.eq(kind)&scored.pred_mapped.eq(annotation)]
                reference_results.append(dict(reference_quantile=q,use_negative_q95_cutoff=negative_cutoff,candidate_kind=kind,
                    annotated_reference_cells=len(x),whole_core_profile=int(x.whole_supported.sum()),noninterface_core_profile=int(x.away_supported.sum()),
                    note='In-sample diagnostic; provisional labels, not independent accuracy validation'))
    result=pd.DataFrame(results);result.to_csv(study.OUT/'profile_threshold_sensitivity.csv',index=False)
    pd.DataFrame(reference_results).to_csv(study.OUT/'provisional_reference_diagnostics.csv',index=False)
    primary=pd.read_csv(study.OUT/'candidate_cell_outcomes.csv');primary=primary[primary.threshold.eq(95)]
    base=result[result.reference_quantile.eq(.1)&result.use_negative_q95_cutoff].set_index('motif')
    expected=primary.groupby('motif')[['whole_profile_supported','noninterface_profile_supported']].sum()
    assert np.array_equal(base.loc[expected.index,'whole_core_profile'].to_numpy(),expected.whole_profile_supported.to_numpy())
    assert np.array_equal(base.loc[expected.index,'noninterface_core_profile'].to_numpy(),expected.noninterface_profile_supported.to_numpy())
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()
