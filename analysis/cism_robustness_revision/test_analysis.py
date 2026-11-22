"""Run with python -m unittest discover -s analysis/cism_robustness_revision."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analyze import core, patient_bootstrap

class AnalysisTests(unittest.TestCase):
    def test_bootstrap_matches_explicit_patient_resampling_with_ties(self):
        y = np.array([1, 1, 1, 0, 0], bool)
        scores = np.array([.5, .8, .3, .5, .2])
        seed, draws = 13, 500
        rng = np.random.default_rng(seed)
        wa = rng.multinomial(3, [1/3]*3, size=draws)
        wb = rng.multinomial(2, [.5]*2, size=draws)
        exact = []
        for a, b in zip(wa, wb):
            boot_scores = np.r_[np.repeat(scores[y], a), np.repeat(scores[~y], b)]
            exact.append(roc_auc_score([1]*3+[0]*2, boot_scores))
        np.testing.assert_allclose(patient_bootstrap(y, scores, draws, seed), np.quantile(exact, [.025,.975]))

    def test_bootstrap_extremes(self):
        self.assertEqual(patient_bootstrap([1,1,0,0], [1,1,0,0]), [1,1])
        self.assertEqual(patient_bootstrap([1,1,0,0], [.5]*4), [.5,.5])
        with self.assertRaises(ValueError): patient_bootstrap([1,1], [.1,.2])

    def test_training_exclusivity_and_strict_prevalence(self):
        classes = dict(zip('abcdef', ['NN']*3+['NP']*3))
        rows = []
        for mid, patients in [(10,'ab'),(11,'abc'),(12,'abcd'),(13,'def')]:
            for p in patients:
                rows.append(dict(ID=mid, patient=p, FOV=p, Count=1, Freq=float(mid), nunique_colors=2, motif='unused'))
        data = core.prepare_data(pd.DataFrame(rows), classes, [.46, 2/3])
        def selected(heldout, threshold):
            return set(data['ids'][list(core.select_fold(data, data['patients'].index(heldout), threshold, 30))])
        self.assertIn(12, selected('d', .46))
        self.assertNotIn(12, selected('e', .46))
        self.assertNotIn(10, selected('f', 2/3))

    def test_historical_pair_normalization_and_graph_deduplication(self):
        from additional_uncertainty import graph_counts
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'Patient_1_FOV1.txt').write_text('1 2 0 0\n2 1 0 0\n2 3 0 1\n')
            counts,edges,manifest=graph_counts(Path(directory),{'Melanoma1':'NN'},2)
        np.testing.assert_array_equal(counts['Melanoma1'],[2,1])
        np.testing.assert_array_equal(edges['Melanoma1'],[[2,1],[1,0]])
        self.assertEqual(manifest[0]['edges'],2)
        self.assertEqual(edges['Melanoma1'][0,1]/edges['Melanoma1'].sum(),.25)

    def test_multiclass_bootstrap_matches_explicit_resampling(self):
        from additional_uncertainty import multiclass_bootstrap
        from sklearn.metrics import accuracy_score,balanced_accuracy_score
        labels=np.array(['NN','NN','NP','NP','PN','PN'])
        probabilities=np.array([[.6,.2,.2],[.3,.4,.3],[.1,.7,.2],[.2,.3,.5],[.1,.3,.6],[.4,.2,.4]])
        classes=np.array(['NN','NP','PN']);predicted=classes[probabilities.argmax(axis=1)]
        draws,seed=300,21;rng=np.random.default_rng(seed)
        weights=np.zeros((draws,6),int)
        for label in classes:weights[:,labels==label]=rng.multinomial(2,[.5,.5],size=draws)
        values={'accuracy':[],'balanced_accuracy':[],'roc_auc_ovr_macro':[]}
        for w in weights:
            ids=np.repeat(np.arange(6),w)
            values['accuracy'].append(accuracy_score(labels[ids],predicted[ids]))
            values['balanced_accuracy'].append(balanced_accuracy_score(labels[ids],predicted[ids]))
            values['roc_auc_ovr_macro'].append(roc_auc_score(labels[ids],probabilities[ids],labels=list(classes),multi_class='ovr',average='macro'))
        result=multiclass_bootstrap(labels,probabilities,predicted,draws,seed)
        for metric,samples in values.items():np.testing.assert_allclose(result[metric],np.quantile(samples,[.025,.975]))

    def test_pairwise_patient_alignment_rejects_numeric_lexical_order_mix(self):
        from additional_uncertainty import verify_graph_feature_alignment
        patients=['CRC1','CRC10','CRC2']
        counts={p:np.array([1,1]) for p in patients}
        edges={p:np.array([[2,i],[i,0]]) for p,i in zip(patients,[1,3,8])}
        x=np.array([[edges[p][0,1]/edges[p].sum()] for p in patients])
        verify_graph_feature_alignment(patients,x,counts,edges,'pairwise')
        with self.assertRaises(AssertionError):
            verify_graph_feature_alignment(['CRC1','CRC2','CRC10'],x,counts,edges,'pairwise')

if __name__ == '__main__': unittest.main()
