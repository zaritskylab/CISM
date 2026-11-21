import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.autonotebook import tqdm

from baselines.utils import Columns, load_tissue_state
from cism.evaluate_aux import get_roc_auc_score


class CellTypeDistributionAnalysis:
    """Cell type distribution baseline: a random forest on each patient's cell type fractions, scored leave one patient
    out. Takes the same patient class csv and graphs (GraphReader.get_graphs) as PairwiseAnalysis."""

    def __init__(
        self,
        tissue_state_csv_path: str,
        tissue_state_to_string: dict,
        tissue_state_func=None,
    ):
        self._patient_class_df = load_tissue_state(
            tissue_state_csv_path=tissue_state_csv_path,
            tissue_state_to_string=tissue_state_to_string,
            tissue_state_func=tissue_state_func,
        )

    @staticmethod
    def get_cell_type_fractions(full_graph_df: pd.DataFrame, cells_type: dict) -> pd.DataFrame:
        """Patient_uId x cell type: each cell type's share of the patient's graph nodes, pooled over its FOVs."""
        counts = {}
        for patient_uid, graphs in full_graph_df.groupby("Patient_uId", sort=False)["graph"]:
            node_types = [int(cell_type) for graph in graphs for _, cell_type in graph.nodes(data="type")]
            counts[patient_uid] = np.bincount(node_types, minlength=len(cells_type))
        counts = pd.DataFrame.from_dict(
            counts, orient="index", columns=[cells_type[cell_type] for cell_type in range(len(cells_type))]
        )
        return counts.div(counts.sum(axis=1), axis=0)

    def analyze(
        self,
        full_graph_df: pd.DataFrame,
        cells_type: dict,
        labels: list,
        trials: int = 10,
    ) -> list:
        """Leave-one-patient-out ROC AUC of each trial over the patients of the two labels. As in
        PairwiseAnalysis.analyze, all of trial i's forests draw from one np.random.RandomState(i)."""
        fractions = self.get_cell_type_fractions(full_graph_df=full_graph_df, cells_type=cells_type)
        patient_class = self._patient_class_df[Columns.PATIENT_CLASS].reindex(fractions.index)
        labelled = patient_class.isin(labels)
        x, y = fractions[labelled], patient_class[labelled]

        results = []
        for trial in range(trials):
            random_state = np.random.RandomState(trial)
            probabilities = []
            for test_patient in tqdm(x.index):
                train = x.index != test_patient
                clf = RandomForestClassifier(random_state=random_state)
                clf.fit(x[train], y[train])
                probability = clf.predict_proba(x.loc[[test_patient]])[0]
                probabilities.append(probability[list(clf.classes_).index(labels[0])])
            results.append(get_roc_auc_score(true_y=y.tolist(), y_prob=probabilities, pos_label=labels[0]))
        return results
