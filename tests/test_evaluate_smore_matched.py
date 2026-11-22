"""Focused checks for the matched SMORE post-run evaluator."""

import unittest

import pandas as pd

from analysis.smore_comparison.evaluate_smore_matched import (
    canonical_consensus,
    evaluate,
    select_train_features,
    validate_manifest,
)


class TestSmoreMatchedEvaluation(unittest.TestCase):
    def test_reverse_canonical_consensus(self):
        self.assertEqual(canonical_consensus("10-3-10-5"), canonical_consensus("5-10-3-10"))
        with self.assertRaises(ValueError):
            canonical_consensus("10-3-5")

    def test_manifest_requires_full_cohort(self):
        manifest = pd.DataFrame({"patient": [1], "class": ["NN"], "fov": ["FOV1"]})
        with self.assertRaisesRegex(ValueError, "76"):
            validate_manifest(manifest)

    def test_training_only_selection_and_failure_policies(self):
        patients = pd.DataFrame({"patient": range(1, 39),
                                 "class": ["NN"] * 20 + ["NP"] * 18})
        features = pd.DataFrame({"patient": range(1, 39),
                                 "feature": ["A-B-C-D"] * 20 + ["W-X-Y-Z"] * 18,
                                 "frequency": [0.01] * 38})
        labels = patients.set_index("patient")["class"]
        selected = select_train_features(features[features.patient != 1],
                                         labels.drop(1), 0.2, 30)
        self.assertEqual(set(selected), {"A-B-C-D", "W-X-Y-Z"})
        summary, folds = evaluate(features, patients, (0.2, 1.1, 1.2), 30, 5)
        self.assertEqual(summary.stringency.tolist(), [0.2, 1.1])
        self.assertEqual(summary.failed_folds.tolist(), [0, 38])
        self.assertEqual(summary.auc_reportable.tolist(), [True, False])
        self.assertEqual(summary.auc_equal.iloc[0], 1.0)
        self.assertEqual(summary.auc_equal.iloc[1], 0.5)
        self.assertEqual(summary.auc_forced_false.iloc[1], 0.0)
        self.assertEqual(len(folds), 76)

    def test_absent_held_out_motif_gets_neutral_score(self):
        patients = pd.DataFrame({"patient": range(1, 39),
                                 "class": ["NN"] * 20 + ["NP"] * 18})
        features = pd.DataFrame({"patient": list(range(1, 20)) + list(range(21, 39)),
                                 "feature": ["A-B-C-D"] * 19 + ["W-X-Y-Z"] * 18,
                                 "frequency": [0.01] * 37})
        summary, folds = evaluate(features, patients, (0.35,), 30, 38)
        missing = folds.set_index("patient").loc[20]
        self.assertFalse(missing.failed)
        self.assertTrue(missing.no_test_motifs)
        self.assertEqual(missing.probability_NN_absent_neutral, 0.5)
        self.assertEqual(summary.no_test_motifs.iloc[0], 1)
        self.assertEqual(summary.no_motifs_to_classify.iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
