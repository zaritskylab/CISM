"""Compare the tutorial's 19-motif reference by labelled graph isomorphism."""
import base64
import json
import pickle
import sys
from pathlib import Path
import networkx as nx
import pandas as pd
from analyze import OUT, ROOT, core, write_json
sys.path.insert(0,str(ROOT))

def main():
    source=ROOT/'melanoma_tutorials/melanoma_tutorials/runtime'
    reference=pd.read_csv(source/'split_fov_scale_experiment/fold_union_stringency_046_stability/native_fold_union_selected_motifs.csv')
    with (source/'serialized/Melanoma_cism_ready.pkl').open('rb') as f: artifact=pickle.load(f)
    frame=artifact['cism'].motifs_dataset
    names=artifact['common_cells_type']
    def graph(encoded):
        g=pickle.loads(base64.b64decode(encoded)).to_undirected()
        for node in g:
            g.nodes[node]['cell']=core.normalize_cell(names[int(g.nodes[node]['type'])])
        return g
    native={int(mid):graph(frame[frame.ID==mid].iloc[0].motif) for mid in reference.ID}
    recovery=pd.read_csv(OUT/'motif_recovery.csv').set_index('motif_id')
    raw={}
    exportroot=Path('/groups/assafza_group/assafza/Omri/golan_output/output/Melanoma/Motifs')
    for path in sorted(exportroot.glob('discriminative_motifs*.csv')):
        for _,row in pd.read_csv(path).iterrows():
            mid=int(row.motif_id)
            if mid in raw: continue
            g=nx.Graph()
            for n in row.nodes.split(', '): g.add_node(n,cell=core.normalize_cell(n.rsplit('#',1)[0]))
            for e in row.edges.split('; '):
                a,b=e.split(' - ');g.add_edge(a,b)
            raw[mid]=g
    rows=[]
    for _,row in reference.iterrows():
        hits=[mid for mid,g in raw.items() if nx.is_isomorphic(native[int(row.ID)],g,node_match=nx.algorithms.isomorphism.categorical_node_match('cell',None))]
        assert len(hits)<=1
        rows.append(dict(tutorial_motif_id=int(row.ID),label=row.label,raw_motif_id=hits[0] if hits else None,raw_recovery_percent=int(recovery.loc[hits[0],'recovery_percent']) if hits else 0))
    pd.DataFrame(rows).to_csv(OUT/'tutorial_reference_mapping.csv',index=False)
    write_json(OUT/'tutorial_reference_mapping_summary.json',dict(n_reference=len(rows),matched_to_raw=sum(r['raw_motif_id'] is not None for r in rows),recovered_over_90_percent=sum(r['raw_recovery_percent']>90 for r in rows),reference='Tutorial native_fold_union_selected_motifs.csv; not independently verified as the manuscript S4/S5 motif list'))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
