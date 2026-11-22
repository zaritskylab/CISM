"""Scientific invariants for CRC permutation selection and evaluation."""
import importlib.util
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import unittest

spec = importlib.util.spec_from_file_location(
    'crc_permutation', Path(__file__).resolve().parents[1] / 'analysis/permutation_controls/crc_label_permutation.py')
crc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crc)


def example():
    labels = pd.Series(['CLR'] * 3 + ['DII'] * 3, index=[f'P{i}' for i in range(6)])
    rows = []
    for patient in labels.index:
        for motif, freq in [(1 if labels[patient] == 'CLR' else 2, .2), (3, .1)]:
            rows.append(dict(Patient_uId=patient, ID=motif, FOV='F0', Count=10,
                             Freq=freq, p_value=.01, nunique_colors=1))
    return pd.DataFrame(rows), labels


def test_held_out_label_cannot_influence_feature_selection():
    data, labels = example()
    a = crc.training_candidates(data, labels, 'P0')
    changed = labels.copy()
    changed['P0'] = 'DII'
    b = crc.training_candidates(data, changed, 'P0')
    pd.testing.assert_frame_equal(a, b)
    assert set(crc.select_ids(a, .55, 30)) == {1, 2}
    assert 3 not in a.ID.values  # Shared motif must never be discriminative.


def test_strict_stringency_and_historical_row_cap():
    table = pd.DataFrame({'ID': [1, 1, 2], 'Freq': [.9, .8, .7],
                          'patient_class': ['CLR'] * 3, 'prevalence': [.5] * 3})
    assert crc.select_ids(table, .5, 2) == []
    assert crc.select_ids(table, .49, 2) == [1]  # Cap rows before unique IDs.


def test_feature_frequency_aggregates_fovs_and_zero_fills():
    data, labels = example()
    extra = data.iloc[[0]].copy()
    extra.FOV = 'F1'
    extra.Count = 20
    extra.Freq = .1
    data = pd.concat([data, extra], ignore_index=True)
    _, matrix = crc.prepare_data(data, labels.index)
    assert np.isclose(matrix.loc['P0', 1], 30 / (50 + 200))
    assert matrix.loc['P3', 1] == 0


def test_zero_feature_folds_are_retained_at_chance():
    data, labels = example()
    significant, matrix = crc.prepare_data(data, labels.index)
    config = dict(max_class_features=30, forest_seed=0, trees=3)
    scores, predictions = crc.evaluate(significant, matrix, labels, [1.0], config, 1)
    assert len(predictions) == 6
    assert predictions.no_features.all()
    assert np.all(predictions.p_CLR == .5)
    assert scores.iloc[0].auc == .5
    assert scores.iloc[0].no_feature_folds == 6


def test_permutation_is_at_patient_level_and_preserves_class_counts():
    data, labels = example()
    shuffled = pd.Series(np.random.RandomState(4).permutation(labels), index=labels.index)
    assert shuffled.value_counts().sort_index().equals(labels.value_counts().sort_index())
    assigned = data.Patient_uId.map(shuffled)
    assert pd.DataFrame({'patient': data.Patient_uId, 'label': assigned}).groupby('patient').label.nunique().eq(1).all()


def test_aggregation_rejects_missing_shuffles():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'original').mkdir()
        (root / 'original/summary.json').write_text(json.dumps({'fingerprint': 'same'}))
        try:
            crc.aggregate(SimpleNamespace(output_root=root, seed_start=0, expected_shuffles=2))
        except ValueError as exc:
            assert 'Missing 2 shuffles' in str(exc)
        else:
            raise AssertionError('Incomplete arrays must not be silently aggregated')


def test_historical_parallel_rng_matches_serial_notebook_sequence():
    data, labels = example()
    significant, matrix = crc.prepare_data(data, labels.index)
    thresholds = [.2, .55, 1.]
    config = dict(max_class_features=30, forest_seed=0, trees=7, rng_mode='historical')
    _, actual = crc.evaluate(significant, matrix, labels, thresholds, config, 2)
    rng = np.random.RandomState(0)
    expected = []
    for threshold in thresholds:
        for patient in labels.index:
            features = crc.select_ids(crc.training_candidates(significant, labels, patient), threshold, 30)
            order = [p for group in pd.unique(labels) for p in significant.Patient_uId.unique()
                     if p != patient and labels[p] == group]
            probability = .5
            if features:
                model = crc.RandomForestClassifier(n_estimators=7, random_state=rng, n_jobs=1)
                model.fit(matrix.loc[order, features].to_numpy(), labels.loc[order].to_numpy())
                probability = model.predict_proba(matrix.loc[[patient], features].to_numpy())[0,
                                            np.flatnonzero(model.classes_ == 'CLR')[0]]
            expected.append(probability)
    np.testing.assert_array_equal(actual.p_CLR, expected)


def test_optimized_and_fixed_p_values_use_separate_matched_baselines():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = {'fingerprint': 'same', 'optimal_auc': .8, 'optimal_stringency': .55,
                    'fixed_auc': .8, 'fixed_stringency': .55}
        (root / 'original').mkdir()
        (root / 'original/summary.json').write_text(json.dumps(original))
        for seed, optimized, fixed in [(0, .7, .6), (1, .9, .7)]:
            directory = root / f'shuffled/seed_{seed}'
            directory.mkdir(parents=True)
            summary = dict(original, seed=seed, optimal_auc=optimized, fixed_auc=fixed, label_hash=str(seed))
            (directory / 'summary.json').write_text(json.dumps(summary))
        crc.aggregate(SimpleNamespace(output_root=root, seed_start=0, expected_shuffles=2))
        result = pd.read_csv(root / 'aggregate/empirical_p_values.csv').set_index('metric')
        assert np.isclose(result.loc['optimal_auc', 'empirical_p'], 2 / 3)
        assert np.isclose(result.loc['fixed_auc', 'empirical_p'], 1 / 3)
        assert (root / 'aggregate/crc_permutation_histograms.pdf').exists()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
