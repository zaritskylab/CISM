import itertools
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from tqdm.autonotebook import tqdm
from cism.cism import AnalyzeMotifsResult
from pairwise.common import Columns
from pairwise.reader import GraphReader
from collections import Counter


class PairwiseAnalysis:
    def __init__(self,
                 tissue_state_csv_path: str,
                 tissue_state_to_string: str,
                 tissue_state_func=None):

        self._patient_class_df = PairwiseAnalysis._load_tissue_state(
            tissue_state_csv_path=tissue_state_csv_path,
            tissue_state_to_string=tissue_state_to_string,
            tissue_state_func=tissue_state_func
        )

    def get_pairwise_counter(self,
                             full_graph_df: pd.DataFrame,
                             patient_class: str,
                             cells_type: dict):
        patients_ids = self._patient_class_df[self._patient_class_df[Columns.PATIENT_CLASS].isin([patient_class])].index
        filtered_df = full_graph_df[full_graph_df['Patient_uId'].isin(patients_ids)].copy()
        filtered_df['pairwise_freq'] = filtered_df[Columns.PAIRWISE_COUNT].transform(lambda x: x / x.sum())
        return GraphReader.get_normalized_matrix(cells_type=cells_type, graph_df=filtered_df)

    def get_cell_type_count(self,
                            full_graph_df: pd.DataFrame,
                            patient_class: str,
                            cells_type: dict,
                            normalize: bool = True):
        cell_type_counts = Counter()
        patients_ids = self._patient_class_df[self._patient_class_df[Columns.PATIENT_CLASS].isin([patient_class])].index
        filtered_df = full_graph_df[full_graph_df['Patient_uId'].isin(patients_ids)].copy()
        for patient_uid in tqdm(filtered_df.Patient_uId.unique()):
            cell_types = []
            for idx, row in filtered_df[filtered_df.Patient_uId == patient_uid].iterrows():
                cell_types.extend([data['type'] for _, data in row['graph'].nodes(data=True)])
            cell_type_counts = cell_type_counts + Counter(cell_types)

        if normalize:
            total_cell_types = sum(cell_type_counts.values())

            normalized_counts = {cells_type[int(cell_type)]: count / total_cell_types for cell_type, count in
                                 cell_type_counts.items()}

            cell_type_counts = Counter(normalized_counts)
        else:
            fix_cell_type_labels = {cells_type[int(cell_type)]: count for cell_type, count in
                                    cell_type_counts.items()}
            cell_type_counts = Counter(fix_cell_type_labels)

        return cell_type_counts

    def get_cell_type_count_from_classes(self,
                                         full_graph_df: pd.DataFrame,
                                         classes: list,
                                         cells_type: dict,
                                         normalize: bool = True):
        result = None
        for clazz in classes:
            counter = self.get_cell_type_count(full_graph_df=full_graph_df,
                                               patient_class=clazz,
                                               cells_type=cells_type,
                                               normalize=normalize)

            df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
            df.columns = ['Cell Type', 'Frequency']
            df['patient_class'] = clazz
            result = pd.concat([df, result])

        return result

    def analyze(self,
                full_graph_df: pd.DataFrame,
                cells_type: dict,
                labels: list,
                trials: int = 10) -> list:
        feature_list = list(itertools.combinations(cells_type.keys(), 2))
        results = []
        local_patient_class_df = self._patient_class_df[self._patient_class_df[Columns.PATIENT_CLASS].isin(labels)]
        local_full_graph_df = full_graph_df[full_graph_df['Patient_uId'].isin(local_patient_class_df.index)]
        for trial in range(trials):
            random_state = np.random.RandomState(trial)
            analyze_results = PairwiseAnalysis._analyze(full_graph_df=local_full_graph_df,
                                                        feature_list=feature_list,
                                                        random_state=random_state,
                                                        labels=labels,
                                                        patient_class_df=local_patient_class_df)
            results.append(analyze_results.get_roc_auc_score())

        return results

    @staticmethod
    def _analyze(full_graph_df: pd.DataFrame,
                 feature_list: list,
                 random_state: np.random.RandomState,
                 labels: list,
                 patient_class_df: pd.DataFrame):

        results = []
        for test_patient in tqdm(full_graph_df.Patient_uId.unique()):
            # iterate across patients and build the feature vector
            x_train_dataset = None
            TP, TN, FP, FN = 0, 0, 0, 0

            for patient in full_graph_df.Patient_uId.unique():
                features_dict = {}

                # add the pairwise freq
                for (u, v) in feature_list:
                    total_count = 0
                    total_all_pairs_count = 0
                    # iterate over all the FOV
                    for idx, row in full_graph_df[full_graph_df.Patient_uId == patient].iterrows():
                        total_count += row[Columns.PAIRWISE_COUNT][u, v]
                        total_all_pairs_count += row[Columns.PAIRWISE_COUNT].sum()

                    total_freq = total_count / total_all_pairs_count
                    features_dict[(u, v)] = total_freq

                # add the class
                features_dict[Columns.PATIENT_CLASS] = patient_class_df.loc[patient, Columns.PATIENT_CLASS]

                if patient == test_patient:
                    x_validation_dataset = pd.DataFrame(features_dict, index=[patient])
                    continue

                # add sample to dataset
                x_train_dataset = pd.concat([x_train_dataset, pd.DataFrame(features_dict, index=[patient])])

            # train the model
            clf = RandomForestClassifier(random_state=random_state)
            clf.fit(x_train_dataset.drop(Columns.PATIENT_CLASS, axis=1),
                    x_train_dataset[Columns.PATIENT_CLASS])

            instance = x_validation_dataset.drop(Columns.PATIENT_CLASS, axis=1)

            y_pred_validation_dataset = clf.predict(instance)
            y_prob_validation_dataset = clf.predict_proba(instance)

            # model expandability
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(instance)

            i = 0
            for class_i in x_validation_dataset[Columns.PATIENT_CLASS]:
                pred_result = y_pred_validation_dataset[i]
                if (pred_result == class_i) and (class_i == labels[0]):
                    TP = + 1
                elif (pred_result == class_i) and (class_i == labels[1]):
                    TN = + 1
                elif pred_result == labels[1]:
                    FN = + 1
                elif pred_result == labels[0]:
                    FP = + 1
                i = + 1

            results.append((TP, TN, FN, FP,
                            len(feature_list),
                            y_prob_validation_dataset, class_i, pred_result, clf.classes_,
                            None,
                            (shap_values, instance)))

        return AnalyzeMotifsResult(analyze_results=results,
                                   patients_ids=full_graph_df.Patient_uId.unique(),
                                   labels=labels)

    @staticmethod
    def _load_tissue_state(tissue_state_csv_path: str,
                           tissue_state_to_string: dict[int, str],
                           tissue_state_func) -> pd.DataFrame:
        patient_class_df = pd.read_csv(tissue_state_csv_path, index_col=0, names=[Columns.PATIENT_CLASS_ID])

        if tissue_state_func:
            patient_class_df[Columns.PATIENT_CLASS] = (patient_class_df[Columns.PATIENT_CLASS_ID]
                                                                    .transform(tissue_state_func, axis=0))
        else:
            # we remove spaces since later on the group-by optimization depends on that.
            patient_class_df[Columns.PATIENT_CLASS] = patient_class_df[Columns.PATIENT_CLASS_ID].transform(
                lambda row: tissue_state_to_string[row].replace(" ", ""), axis=0)
        return patient_class_df

