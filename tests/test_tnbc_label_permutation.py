"""Validate the TNBC cohort adapter and class encoding on scientific inputs."""
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from analysis.permutation_controls import tnbc_label_permutation as tnbc


class TNBCInputsTest(unittest.TestCase):
    def test_saved_labels_match_survival_cutoff_and_search(self):
        artifact_path = tnbc.ROOT / 'tutorials/runtime/serialized/TNBC_cism_ready.pkl'
        labels_path = tnbc.ROOT / 'pet_data/TNBC/patient_class.csv'
        if not artifact_path.is_file() or not labels_path.is_file():
            self.skipTest('External TNBC artifact and patient labels are required')
        artifact, labels, grid = tnbc.inputs(artifact_path, labels_path)
        self.assertEqual(labels.value_counts().to_dict(),
                         {'Long-term survival': 30, 'Short-term survival': 7})
        self.assertEqual(len(grid), 3)
        self.assertIn(artifact['best_feature_conf'].shared_percentage, grid)
        self.assertEqual(artifact['survival_rate_cutoff'], 1000)
        self.assertEqual(artifact['best_feature_conf'].max_class_features, 30)

    def test_encoded_positive_auc_equals_original_positive_auc(self):
        labels = np.array(['Short-term survival', 'Long-term survival'] * 3)
        p_short = np.array([.2, .1, .8, .6, .7, .3])
        self.assertAlmostEqual(roc_auc_score(labels == 'Short-term survival', p_short),
                               roc_auc_score(labels == 'Long-term survival', 1 - p_short))
        self.assertEqual(tnbc.ENCODING['Long-term survival'], 'CLR')
        self.assertEqual(tnbc.ENCODING['Short-term survival'], 'DII')


if __name__ == '__main__':
    unittest.main()
