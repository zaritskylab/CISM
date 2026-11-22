"""Small geometries with known directed counts and FOV-specific baselines."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('enrichment', Path(__file__).resolve().parents[1]/'analysis/marker_ownership/unidentified_neighbor_enrichment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_counts_self_exclusion_and_patient_isolation():
    cells = pd.DataFrame({
        'patient number':[1,1,1,2,2], 'fov':['FOV1']*5,
        'cell_id':[1,2,3,1,2], 'centroid-0':[0,1,0,0,1],
        'centroid-1':[0,0,1,0,0],
        'pred_mapped':['unidentified','b cell','cd8 t cell','unidentified','mac'],
        'cluster_15':[0,np.nan,np.nan,1,np.nan]})
    obs, exp, support = m.neighbor_matrices(cells,100)
    assert obs.shape == (15,16)
    assert obs.loc[0,'b cell'] == obs.loc[0,'cd8 t cell'] == 1
    assert obs.loc[1,'mac'] == 1
    assert obs['unidentified'].sum() == 0
    np.testing.assert_allclose(obs,exp)
    assert support.neighbor_count.tolist() == [2,1]
    empty, expected, support = m.neighbor_matrices(cells,0.5)
    assert empty.to_numpy().sum() == expected.to_numpy().sum() == 0
    assert support.neighbor_count.sum() == 0
