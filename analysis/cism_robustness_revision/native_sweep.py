"""Native-selector sweeps for the two primary configurations; no documents edited."""
import ast
import json
import pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.compute as pc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from analyze import OUT,ROOT,THRESHOLDS,write_json

def main():
    source=ROOT/'cism/cism.py'
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TissueStateDiscriminativeMotifs')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_extract_features');fn.decorator_list=[]
    ns=dict(pd=pd,np=np,FeatureConfiguration=object,SoftDiscriminativeFC=type('Soft',(),{}),DiscriminativeMotifs=SimpleNamespace(PATIENT_CLASS='patient_class',PATIENT_PERCENTAGE_KEY='patient_percentage'))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    conf=SimpleNamespace(get_shared_percentage=lambda:.46,max_class_features=30,use_cells_type_composition=False,use_motifs=True,cells_type_composition_vec=[],switch_motif_with_cell_type_composition=[],switch_cell_type_composition_with_motif=[],switch_cell_type_composition_with_motif_map={},cells_type_composition_hash_remove=[],motifs_ids_remove=[])
    for name in ['nn_np_size4','np_pn_size5']:
        with (OUT/'prepared'/f'{name}.pkl').open('rb') as f:spec=pickle.load(f)
        data=spec['data'];idx={int(mid):i for i,mid in enumerate(data['ids'])}
        framecache=OUT/'prepared'/f'{name}_native_rows.parquet'
        if framecache.exists():frame=pd.read_parquet(framecache)
        else:
            frames=[]
            print('Reading native sweep rows',name,flush=True)
            for path,_,_ in spec['manifest']:
                t=pacsv.read_csv(path,read_options=pacsv.ReadOptions(use_threads=False),convert_options=pacsv.ConvertOptions(include_columns=['ID','Freq','Patient','p_value','Mean_Freq','nunique_colors']))
                t=t.filter(pc.and_(pc.less(t['p_value'],.05),pc.is_valid(t['Mean_Freq'])))
                frames.append(t.to_pandas())
            frame=pd.concat(frames,ignore_index=True)
            frame['patient']='Melanoma'+frame.Patient.astype(int).astype(str)
            frame['patient_class']=frame.patient.map(spec['classes'])
            frame=frame[frame.ID.isin(data['ids'])&frame.patient_class.notna()].copy()
            frame.to_parquet(framecache,index=False)
        curves={th:dict(probabilities=[],union=set(),fold_selections={}) for th in THRESHOLDS}
        ref=json.loads((OUT/f'{name}_native_selection.json').read_text())
        for heldout,p in enumerate(data['patients']):
            train=np.arange(len(data['patients']))!=heldout
            percentage=np.zeros(len(data['ids']))
            for label,other in [('NN','NP'),('NP','NN')]:
                local=train&(data['labels']==label);opposite=train&(data['labels']==other)
                freq=data['presence'][:,local].sum(axis=1)/local.sum()
                eligible=~data['presence'][:,opposite].any(axis=1)
                percentage=np.maximum(percentage,np.where(eligible,freq,0))
            pct=dict(zip(data['ids'],percentage))
            subset=frame[(frame.patient!=p)&frame.ID.isin(data['ids'][percentage>min(THRESHOLDS)])&(frame.nunique_colors>=1)].copy()
            subset['patient_percentage']=subset.ID.map(pct)
            fitcache={}
            for th in THRESHOLDS:
                conf.get_shared_percentage=lambda:th
                _,ids,_=ns['_extract_features'](subset,conf)
                if th==spec['threshold']:assert ids==ref['fold_selections'][p],(name,p)
                key=tuple(ids);c=curves[th]
                c['union'].update(ids);c['fold_selections'][p]=list(map(int,ids))
                if not ids:prob=None
                elif key in fitcache:prob=fitcache[key]
                else:
                    x=data['features'][[idx[mid] for mid in ids]].T
                    clf=RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=1)
                    clf.fit(x[train],data['labels'][train])
                    prob=float(clf.predict_proba(x[[heldout]])[0,list(clf.classes_).index('NN')]);fitcache[key]=prob
                c['probabilities'].append(prob)
        results=[];allprobs={};allfolds={}
        for th,c in curves.items():
            empty=sum(p is None for p in c['probabilities'])
            score=None if empty else float(roc_auc_score(data['labels']=='NN',c['probabilities']))
            results.append(dict(name=name,stringency=th,auc=score,empty_folds=empty,n_motifs=len(c['union']),rf_admissible=not empty))
            allprobs[str(th)]=c['probabilities'];allfolds[str(th)]=c['fold_selections']
        write_json(OUT/f'{name}_native_sweep.json',dict(curves=results,probabilities=allprobs,fold_selections=allfolds,patients=data['patients'],positive=spec['positive'],negative=spec['negative'],reference_selection_matches=True))
        pd.DataFrame(results).to_csv(OUT/f'{name}_native_sweep.csv',index=False)
        print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=='__main__':main()
