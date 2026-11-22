"""Repeat NP–PN classifier runs using the original CISM feature selector."""
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from analyze import OUT,patient_bootstrap,write_json

def run(seed):
    with (OUT/'prepared/np_pn_size5.pkl').open('rb') as f:spec=pickle.load(f)
    selected=json.loads((OUT/'np_pn_size5_native_selection.json').read_text())
    assert selected['n_motifs']==210
    data=spec['data'];index={int(mid):i for i,mid in enumerate(data['ids'])}
    patients=data['patients'];probs=[]
    for heldout,patient in enumerate(patients):
        ids=selected['fold_selections'][patient]
        assert ids
        columns=[index[mid] for mid in ids]
        x=data['features'][columns].T
        train=np.arange(len(patients))!=heldout
        clf=RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1)
        clf.fit(x[train],data['labels'][train])
        probs.append(float(clf.predict_proba(x[[heldout]])[0,list(clf.classes_).index('NN')]))
    y=data['labels']=='NN'
    low,high=patient_bootstrap(y,probs)
    result=dict(seed=seed,auc=roc_auc_score(y,probs),ci_low=low,ci_high=high,n_motifs=210,positive='NP',negative='PN',patients=patients,labels=[spec['classes'][p] for p in patients],probabilities=probs,selection_source='np_pn_size5_native_selection.json',data_signature=spec['signature'])
    write_json(OUT/'native_executions'/f'np_pn_seed_{seed:03d}.json',result)
    return result

def main():
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(30)))
    rows=pd.DataFrame([{k:r[k] for k in ['seed','auc','ci_low','ci_high','n_motifs']} for r in results])
    rows.to_csv(OUT/'native_np_pn_seed_metrics.csv',index=False)
    rows_pred=[]
    for r in results:
        for patient,label,prob in zip(r['patients'],r['labels'],r['probabilities']):rows_pred.append(dict(seed=r['seed'],patient=patient,label=label,positive='NP',probability=prob))
    pd.DataFrame(rows_pred).to_csv(OUT/'native_np_pn_patient_predictions.csv',index=False)
    summary=dict(n=30,mean_auc=rows.auc.mean(),sd_auc=rows.auc.std(),min_auc=rows.auc.min(),max_auc=rows.auc.max(),reference=rows[rows.seed==0].iloc[0].to_dict(),n_motifs=210)
    write_json(OUT/'native_np_pn_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
