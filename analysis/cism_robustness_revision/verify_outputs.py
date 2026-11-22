"""Completion checks against all execution files and exported source tables."""
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analyze import OUT,patient_bootstrap,write_json

def main():
    expected={'nn_np_raw4':30,'nn_np_size3':1,'nn_np_size4':30,'nn_np_size5':1,'np_pn_size3':1,'np_pn_size4':1,'np_pn_size5':30}
    rs=[json.loads(p.read_text()) for p in (OUT/'executions').glob('*.json')]
    assert Counter(r['name'] for r in rs)==expected
    ncurves=0
    for r in rs:
        with (OUT/'prepared'/f"{r['name']}.pkl").open('rb') as f:spec=pickle.load(f)
        assert r['signature']==spec['signature']
        assert len(r['patients'])==len(set(r['patients']))==len(r['labels'])
        for path,size,mtime in spec['manifest']:
            st=Path(path).stat();assert (st.st_size,st.st_mtime_ns)==(size,mtime)
        expected_classes={'NN':20,'NP':18} if r['positive']=='NN' else {'NP':18,'PN':13}
        assert Counter(r['labels'])==expected_classes
        y=np.array(r['labels'])==r['positive']
        for c in r['curves']:
            probs=np.array(r['probabilities'][str(c['stringency'])])
            assert len(probs)==len(y) and np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
            assert c['rf_admissible']==(c['empty_folds']==0)
            if c['rf_admissible']: assert np.isclose(c['auc'],roc_auc_score(y,probs),atol=1e-12)
            else: assert c['auc'] is None
            assert c['n_motifs']==len(set(r['motif_ids'][str(c['stringency'])]))
            ncurves+=1
    hist=pd.read_csv(OUT/'historical_source_audit.csv')
    assert len(hist)==100 and hist.fingerprint_matches.all() and hist.export_matches.all() and (hist.n_fovs==76).all()
    metrics=pd.read_csv(OUT/'historical_run_metrics.csv')
    assert len(metrics)==200 and ((metrics.ci_low<=metrics.auc)&(metrics.auc<=metrics.ci_high)).all()
    predictions=pd.read_csv(OUT/'historical_patient_predictions.csv')
    assert len(predictions)==7600
    for (_,mode),g in predictions.groupby(['run_id','mode']):
        assert len(g)==38 and g.patient.nunique()==38
    main=pd.read_csv(OUT/'classifier_seed_metrics.csv')
    assert len(main)==90 and main.rf_admissible.all()
    for r in rs:
        if r['seed']==0 and r['name'] in ['nn_np_raw4','nn_np_size4','np_pn_size5']:
            th=.4 if r['positive']=='NP' else .46
            low,high=patient_bootstrap(np.array(r['labels'])==r['positive'],r['probabilities'][str(th)])
            row=main[(main.name==r['name'])&(main.seed==0)].iloc[0]
            np.testing.assert_allclose([low,high],[row.ci_low,row.ci_high])
    mapping=pd.read_csv(OUT/'tutorial_reference_mapping.csv')
    assert len(mapping)==19 and mapping.raw_motif_id.notna().all() and (mapping.raw_recovery_percent>90).sum()==18
    replay=json.loads((OUT/'reference_replay_audit.json').read_text())
    assert replay['selected_motif_sets_identical'] and replay['max_prediction_difference']==0
    parity=pd.read_csv(OUT/'native_selection_parity.csv')
    assert len(parity)==69
    assert parity[parity.name=='nn_np_size4'].same_order.all()
    native=[json.loads(p.read_text()) for p in (OUT/'native_executions').glob('*.json')]
    assert len(native)==30 and {r['seed'] for r in native}==set(range(30))
    native_selection=json.loads((OUT/'np_pn_size5_native_selection.json').read_text())
    assert native_selection['n_motifs']==210
    for r in native:
        assert r['n_motifs']==210 and len(r['patients'])==31 and len(set(r['patients']))==31
        y=np.array(r['labels'])=='NP'
        assert np.isclose(roc_auc_score(y,r['probabilities']),r['auc'])
        if r['seed']==0:
            np.testing.assert_allclose(patient_bootstrap(y,r['probabilities']),[r['ci_low'],r['ci_high']])
            assert np.isclose(r['auc'],.8183760683760684)
    for name in ['nn_np_size4','np_pn_size5']:
        sweep=json.loads((OUT/f'{name}_native_sweep.json').read_text())
        reference=json.loads((OUT/f'{name}_native_selection.json').read_text())
        fixed='0.4' if name=='np_pn_size5' else '0.46'
        assert sweep['fold_selections'][fixed]==reference['fold_selections']
        with (OUT/'prepared'/f'{name}.pkl').open('rb') as f:spec=pickle.load(f)
        y=spec['data']['labels']=='NN'
        for c in sweep['curves']:
            probs=sweep['probabilities'][str(c['stringency'])]
            assert c['empty_folds']==sum(p is None for p in probs)
            if c['rf_admissible']:assert np.isclose(c['auc'],roc_auc_score(y,probs))
            else:assert c['auc'] is None
    primary=pd.read_csv(OUT/'primary_classifier_summary.csv')
    final=primary[primary.name=='np_pn_size5_original_selector'].iloc[0]
    assert np.isclose(final.mean_auc,np.mean([r['auc'] for r in native]))
    assert final.n_motifs==210
    for artifact in ['RESULTS.md','suggested_edits.md','uncertainty_summary.pdf','primary_parameter_sensitivity.pdf']:
        assert (OUT/artifact).stat().st_size>1000
    receipt=dict(status='passed',new_first_pass_executions=len(rs),first_pass_forest_fits=sum(r['fits'] for r in rs),new_original_selector_np_pn_executions=30,original_selector_np_pn_fixed_fits=930,native_primary_sweeps=2,new_first_pass_threshold_results=ncurves,historical_executions=100,historical_source_fovs=7600,bootstrap_replicates_per_interval=10000,native_selector_folds=69,first_pass_nn_np_native_selection_identical=True,np_pn_primary_uses_native_selector=True,np_pn_historical_notebook_auc_recovered=True,exact_raw_reference_prediction_replay=True,google_documents_edited=False,gnn_runs=0)
    if (OUT/'crc/summary.json').exists():
        crc_summary=json.loads((OUT/'crc/summary.json').read_text())
        crc_metrics=pd.read_csv(OUT/'crc/seed_metrics.csv')
        crc_predictions=pd.read_csv(OUT/'crc/patient_predictions.csv')
        assert len(crc_metrics)==30 and set(crc_metrics.seed)==set(range(30))
        assert len(crc_predictions)==1050 and crc_predictions.groupby('seed').patient.nunique().eq(35).all()
        for seed,g in crc_predictions.groupby('seed'):
            assert g.true_class.value_counts().to_dict()=={'DII':18,'CLR':17}
            m=crc_metrics[crc_metrics.seed==seed].iloc[0]
            assert np.isclose(roc_auc_score(g.true_class=='CLR',g.p_CLR),m.auc)
            assert m.n_motifs==119
        assert np.isclose(crc_metrics.auc.mean(),crc_summary['mean_auc'])
        assert np.isclose(crc_metrics.auc.std(),crc_summary['sd_auc'])
        reference=crc_predictions[crc_predictions.seed==0]
        np.testing.assert_allclose(patient_bootstrap(reference.true_class=='CLR',reference.p_CLR),[crc_summary['reference']['ci_low'],crc_summary['reference']['ci_high']])
        assert crc_summary['historical_reference_max_prediction_difference']==0
        manifest=json.loads((OUT/'crc/source_manifest.json').read_text())
        for key in ['artifact','labels']:
            assert hashlib.sha256(Path(manifest[key]).read_bytes()).hexdigest()==manifest[key+'_sha256']
        receipt.update(crc_classifier_executions=30,crc_classifier_fits=1050,crc_reference_predictions_replayed_exactly=True,crc_source_hashes_match=True)
    if (OUT/'tnbc/summary.json').exists():
        tnbc_summary=json.loads((OUT/'tnbc/summary.json').read_text())
        tnbc_metrics=pd.read_csv(OUT/'tnbc/seed_metrics.csv')
        tnbc_predictions=pd.read_csv(OUT/'tnbc/patient_predictions.csv')
        assert len(tnbc_metrics)==60 and len(tnbc_predictions)==2220
        for setting,metrics in tnbc_metrics.groupby('setting'):
            assert len(metrics)==30 and set(metrics.seed)==set(range(30))
            summary=tnbc_summary[setting]
            assert np.isclose(metrics.auc.mean(),summary['mean_auc'])
            assert np.isclose(metrics.auc.std(),summary['sd_auc'])
            assert metrics.n_motifs.eq(33 if setting=='figure_8a' else 37).all()
            for seed,predictions in tnbc_predictions[tnbc_predictions.setting==setting].groupby('seed'):
                assert predictions.patient.nunique()==37 and len(predictions)==37
                assert predictions.true_class.value_counts().to_dict()=={'Long-term survival':30,'Short-term survival':7}
                assert np.isclose(roc_auc_score(predictions.true_class=='Long-term survival',predictions.p_long_term_survival),metrics[metrics.seed==seed].auc.iloc[0])
                if seed==0:np.testing.assert_allclose(patient_bootstrap(predictions.true_class=='Long-term survival',predictions.p_long_term_survival),[summary['reference']['ci_low'],summary['reference']['ci_high']])
        assert tnbc_summary['audit']['saved_reference_max_probability_difference']<1e-14
        manifest=json.loads((OUT/'tnbc/source_manifest.json').read_text())
        for key in ['artifact','labels']:
            assert hashlib.sha256(Path(manifest[key]).read_bytes()).hexdigest()==manifest[key+'_sha256']
        assert np.isclose(tnbc_summary['figure_8a']['stringency'],.45)
        receipt.update(tnbc_settings=2,tnbc_classifier_seeds_per_setting=30,tnbc_classifier_fits=2220,tnbc_figure8a_motifs=33,tnbc_saved_reference_replayed_to_floating_point_precision=True,tnbc_source_hashes_match=True)
    write_json(OUT/'completion_audit.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
