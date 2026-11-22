"""Synthetic localization tests: donor brightness alone is not spillover."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('spillover', Path(__file__).resolve().parents[1]/'analysis/marker_ownership/unidentified_spillover.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def adjacent():
    labels = np.zeros((30, 40), np.int32)
    labels[5:25, 5:20] = 1
    labels[5:25, 20:35] = 2
    return labels


def test_spillover_localization_and_interface_deduplication():
    labels = adjacent()
    image = np.zeros(labels.shape)
    image[labels == 1] = 1
    image[labels == 2] = 100
    image[5:25, 18:20] = 30
    geometry = m.geometry(labels, [1])
    row, interfaces, _, _ = m.extract_marker(geometry, image, [1])
    r = row.iloc[0]
    assert len(interfaces) == 1 and r.donor_cell_id == 2
    assert interfaces.interface_pixels.iloc[0] <= np.count_nonzero(geometry['boundary'] == 1)
    assert r.interface_log2_asymmetry > 1
    assert r.boundary_log2_ratio > 1
    assert r.donor_log2_ratio > 1
    assert r.basic_signal_evidence


def test_real_membrane_ring_and_bright_neighbor_do_not_suffice():
    labels = adjacent()
    geometry = m.geometry(labels, [1])
    image = np.ones(labels.shape)
    image[geometry['boundary'] == 1] = 30
    image[labels == 2] = 100
    row, _, _, _ = m.extract_marker(geometry, image, [1])
    assert row.iloc[0].boundary_log2_ratio > 1
    assert abs(row.iloc[0].interface_log2_asymmetry) < 1e-12
    assert not row.iloc[0].basic_signal_evidence
    image[labels == 1] = 2
    row, _, _, _ = m.extract_marker(geometry, image, [1])
    assert row.iloc[0].donor_log2_ratio > 1
    assert not row.iloc[0].basic_signal_evidence


def test_small_edge_missing_and_misaligned_cells_are_not_valid():
    labels = adjacent()
    labels[0:3, 0:3] = 3
    frame = pd.DataFrame({'cell_id': [1, 3, 99], 'in_motif': [True, False, False],
        'motif_occurrences': [2, 0, 0], 'Group': ['NN']*3, 'patient number': [1]*3,
        'fov': ['FOV1']*3, 'centroid-0': [14.5, 1, 50], 'centroid-1': [12, 1, 50]})
    info = m.cell_geometry(m.geometry(labels, [1, 3, 99]), frame)
    assert info.geometry_valid.tolist() == [True, False, False]
    assert info.edge_clipped.tolist() == [False, True, False]
    assert info.mask_present.tolist() == [True, True, False]
    frame.loc[0, 'centroid-0'] += 2
    assert not m.cell_geometry(m.geometry(labels, [1]), frame).geometry_valid.iloc[0]


def test_nuclear_components_screen_merged_mask():
    labels = np.zeros((35, 35), np.int32)
    labels[3:32, 3:32] = 1
    image = np.zeros(labels.shape)
    image[9:14, 9:14] = 100
    image[22:27, 22:27] = 100
    assert m.nuclear_qc(image, labels, [1])[0] == 2


def test_reference_scoring_excludes_self_and_requires_joint_evidence():
    n = 62
    frame = pd.DataFrame({'cell_id': range(1, n+1),
        'boundary_log2_ratio': np.linspace(-.1, .1, n),
        'interface_log2_asymmetry': np.linspace(-.1, .1, n),
        'donor_log2_ratio': np.linspace(-.1, .1, n), 'basic_signal_evidence': False})
    info = pd.DataFrame({'cell_id': range(1, n+1), 'mask_area': 100,
                         'crowding_50px': 5, 'geometry_valid': True, 'in_motif': False})
    info.loc[n-1, 'in_motif'] = True
    frame.loc[n-1, m.METRICS] = 5
    frame.loc[n-1, 'basic_signal_evidence'] = True
    result = m.score_marker(frame, info)
    assert result.iloc[-1].flag_q95
    assert not result.iloc[:-1].flag_q95.any()
    # With every control included, max control's donor threshold cannot contain itself.
    result = m.score_marker(frame, info, k=100)
    assert result.iloc[0].reference_n == 60
    assert result.iloc[-1].reference_n == 61
    result = m.score_marker(frame.iloc[:10].copy(), info.iloc[:10].copy())
    assert result.flag_q95.isna().all()


def test_cohort_deduplicates_motif_hits(tmp_path):
    spatial = pd.DataFrame({'patient number': [1, 1], 'fov': ['FOV1']*2,
        'cell_id': [5, 6], 'Group': ['NN', 'NN'], 'pred_mapped': ['unidentified']*2})
    spatial.to_csv(tmp_path/'mapped_cell_types.csv', index=False)
    hits = pd.DataFrame({'patient': [1, 1], 'FOV': ['FOV1']*2, 'cell4': [5, 5]})
    hits.to_csv(tmp_path/'hits.csv', index=False)
    result = m.load_cohort(tmp_path, tmp_path/'hits.csv')
    assert len(result) == 2 and result.in_motif.sum() == 1
    assert result.motif_occurrences.tolist() == [2, 0]


def test_bootstrap_uses_patients_and_paired_support():
    cells = pd.DataFrame({'Group': ['NN']*5, 'patient number': [1, 1, 1, 2, 2],
        'in_motif': [True, False, False, True, False], 'cell_id': range(5),
        'scored_markers': [35]*5, 'status_q95': ['spillover_consistent', 'low_evidence', 'low_evidence', 'low_evidence', 'spillover_consistent']})
    patients, summary = m.bootstrap_summary(cells, iterations=50)
    motif = summary[(summary.group == 'NN') & (summary.comparison == 'motif')].iloc[0]
    assert motif.patients == 2 and motif.estimate == .5
    paired = summary[(summary.group == 'NN') & (summary.comparison == 'paired_motif_minus_control')].iloc[0]
    assert paired.estimate == 0


def test_marker_shape_and_negative_values_rejected():
    import unittest
    g = m.geometry(adjacent(), [1])
    case = unittest.TestCase()
    with case.assertRaises(ValueError): m.extract_marker(g, np.ones((2, 3)), [1])
    with case.assertRaises(ValueError): m.extract_marker(g, -np.ones((30, 40)), [1])


def test_no_donor_is_missing_evidence_not_negative_evidence():
    labels = adjacent()
    labels[labels == 2] = 0
    row, interfaces, _, _ = m.extract_marker(m.geometry(labels, [1]), np.ones(labels.shape), [1])
    assert interfaces.empty
    assert np.isnan(row.iloc[0].donor_log2_ratio)
    assert not row.iloc[0].basic_signal_evidence
    info = pd.DataFrame({'cell_id': [1], 'mask_area': [300], 'crowding_50px': [0],
                         'geometry_valid': [True], 'in_motif': [True]})
    scored = m.score_marker(row, info)
    assert scored.flag_q95.isna().all()


def test_small_gap_donor_is_found_without_delaunay_edges():
    labels = adjacent()
    labels[5:25, 20:22] = 0
    g = m.geometry(labels, [1], radius=3)
    assert g['pair_donor'].tolist() == [2]
    assert g['interface_pixels'][0] == 20


def test_status_needs_complete_coverage_and_retains_positive_partial_evidence():
    info = pd.DataFrame({'cell_id': [1, 2, 3], 'in_motif': [True, False, False],
                         'geometry_valid': [True]*3, 'multiple_dna_components': [False]*3})
    records = []
    for cell in [1, 2, 3]:
        for marker in m.FEATURES:
            flag = True if cell == 3 and marker == 'CD3' else pd.NA if cell != 1 and marker == 'CD4' else False
            records.append({'cell_id': cell, 'marker': marker, 'flag_q90': flag, 'flag_q95': flag, 'flag_q99': flag})
    out = m.summarize_cells(pd.DataFrame(records), info)
    assert out.status_q95.tolist() == ['low_evidence', 'unresolved', 'spillover_consistent']
    assert out.scored_markers.tolist() == [35, 34, 34]


def test_lineage_patterns_are_compatibility_screens_and_dna_qc_is_independent():
    info = pd.DataFrame({'cell_id': [1, 2], 'in_motif': [True, False],
                         'geometry_valid': [True, True], 'multiple_dna_components': [False, True]})
    records = []
    for cell in [1, 2]:
        for marker in m.FEATURES:
            records.append({'cell_id': cell, 'marker': marker, 'flag_q90': False,
                'flag_q95': False, 'flag_q99': False,
                'lineage_whole_positive': marker in ['CD3', 'CD45', 'CD20'],
                'lineage_interior_positive': marker in ['CD3', 'CD45']})
    result = m.summarize_cells(pd.DataFrame(records), info)
    assert result.whole_mixed_lineage.all()
    assert not result.interior_mixed_lineage.any()
    assert result.mixed_profile_resolves_with_boundary_exclusion.all()
    assert result.possible_reannotation_candidate.all()
    assert result.status_q95.tolist() == ['low_evidence', 'unresolved']
    assert 'pred_mapped' not in result  # No new annotation is emitted.


if __name__ == '__main__':
    # Standard-library runner for the host's scientific Python without pytest.
    import tempfile
    import unittest
    def run_temporary_cohort():
        with tempfile.TemporaryDirectory() as directory:
            test_cohort_deduplicates_motif_hits(Path(directory))
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in list(globals().items())
                               if name.startswith('test_') and name != 'test_cohort_deduplicates_motif_hits')
    suite.addTest(unittest.FunctionTestCase(run_temporary_cohort))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
