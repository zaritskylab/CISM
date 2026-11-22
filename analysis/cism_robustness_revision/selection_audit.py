"""Compare core selections with original CISM methods on the actual source rows."""
import ast
import json
import pickle
from types import SimpleNamespace
import typing
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.compute as pc
from analyze import OUT,ROOT,core,write_json

def main():
    path=ROOT/'cism/cism.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TissueStateDiscriminativeMotifs')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_extract_discriminative','_extract_features']]
    for m in methods:m.decorator_list=[]
    ns=dict(pd=pd,np=np,typing=typing,FeatureConfiguration=object,SoftDiscriminativeFC=type('Soft',(),{}),DiscriminativeMotifs=SimpleNamespace(PATIENT_CLASS='patient_class',PATIENT_COUNT_KEY='patient_count',PATIENT_PERCENTAGE_KEY='patient_percentage'))
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),ns)
    conf=SimpleNamespace(get_shared_percentage=lambda:.46,max_class_features=30,use_cells_type_composition=False,use_motifs=True,cells_type_composition_vec=[],switch_motif_with_cell_type_composition=[],switch_cell_type_composition_with_motif=[],switch_cell_type_composition_with_motif_map={},cells_type_composition_hash_remove=[],motifs_ids_remove=[])
    rows=[]
    for name in ['nn_np_size4','np_pn_size5']:
        with (OUT/'prepared'/f'{name}.pkl').open('rb') as f:spec=pickle.load(f)
        conf.get_shared_percentage=lambda:spec['threshold']
        frames=[]
        print('Reading for native selection audit:',name,flush=True)
        for path,_,_ in spec['manifest']:
            t=pacsv.read_csv(path,read_options=pacsv.ReadOptions(use_threads=False),convert_options=pacsv.ConvertOptions(include_columns=['ID','Freq','Count','Patient','FOV','p_value','Mean_Freq','nunique_colors']))
            t=t.filter(pc.and_(pc.less(t['p_value'],.05),pc.is_valid(t['Mean_Freq'])))
            frames.append(t.to_pandas())
        frame=pd.concat(frames,ignore_index=True)
        frame['Patient_uId']='Melanoma'+frame.Patient.astype(int).astype(str)
        frame['patient_class']=frame.Patient_uId.map(spec['classes'])
        frame=frame[frame.patient_class.notna()]
        data=spec['data'];union=set();fold_selections={}
        # Remove only IDs that cannot pass this threshold in any fold. Keep
        # every patient so the native method's prevalence denominators agree.
        prevalence=data['presence'].sum(axis=1)
        class_counts=np.stack([data['presence'][:,data['labels']==label].sum(axis=1) for label in ['NN','NP']],axis=1)
        denominators=np.array([(data['labels']==label).sum()-1 for label in ['NN','NP']])
        potential=(class_counts.min(axis=1)<=1)&((class_counts/denominators).max(axis=1)>spec['threshold'])
        frame=frame[frame.ID.isin(data['ids'][potential])].copy()
        assert set(frame.Patient_uId)==set(spec['classes'])
        for i,p in enumerate(data['patients']):
            train=frame[frame.Patient_uId!=p].copy()
            original,_=ns['_extract_discriminative'](None,train,'patient_class','ID',[],1,1,.05)
            _,native,_=ns['_extract_features'](original,conf)
            fold_selections[p]=[int(mid) for mid in native]
            actual=[int(data['ids'][ix]) for ix in core.select_fold(data,i,spec['threshold'],30)]
            union.update(native)
            rows.append(dict(name=name,heldout=p,native_count=len(native),core_count=len(actual),same_set=set(native)==set(actual),same_order=native==actual,native_only=sorted(set(native)-set(actual)),core_only=sorted(set(actual)-set(native))))
        print(name,'native union count',len(union),'same sets',sum(r['same_set'] for r in rows if r['name']==name),flush=True)
        write_json(OUT/f'{name}_native_selection.json',dict(motif_ids=sorted(map(int,union)),n_motifs=len(union),fold_selections=fold_selections,row_order='sorted source cache filename; original pandas default frequency sort',source='current cism/cism.py _extract_discriminative and _extract_features'))
    pd.DataFrame(rows).to_csv(OUT/'native_selection_parity.csv',index=False)

if __name__=='__main__':main()
