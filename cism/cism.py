import typing
from typing import List, Dict
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap

from cism import helpers

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
import os
import enum
# import modin.pandas as pd
import pandas as pd
import networkx as nx
import numpy as np
from tqdm.autonotebook import tqdm


class CISM:
    def __init__(self, **kwargs):
        """Initiate CISM object. Context-dependant identification of spatial motifs (CISM) that captures recurrent
        patterns in cell type organization within the tissue to identify modular components at the spatial scale of a
        few cells.

        Keyword Args:
            fanmod_path (str): filesystem path to fanmod+ tool.
            fanmod_exe (str): the filename of fanmod+ executable.
            network_dataset_root_path (str): filesystem path to the network dataset root directory.
            fanmod_output_root_path (str): filesystem path to output fanmod+ artifacts in leda format
            fanmod_cache_root_path (str): filesystem path to output fanmod+ cache artifacts
            motif_size (int): the number of nodes in each motif
            iterations (int): the number of random networks to be generated for assessing motif significance

        Network file format - each file should represent single patient/area with the format:
        {left cell id} {right cell id} {left cell type} {right cell type}
        """
        self.fanmod_path = kwargs['fanmod_path']
        self.fanmod_exe = kwargs['fanmod_exe']
        self.network_dataset_root_path = kwargs['network_dataset_root_path']
        self.fanmod_output_root_path = kwargs['fanmod_output_root_path']
        self.fanmod_cache_root_path = kwargs['fanmod_cache_root_path']
        self.motif_size = kwargs['motif_size']
        self.iterations = kwargs['iterations']
        self.motifs_dataset = None

    def add_dataset(self, dataset_folder, dataset_type, dataset_name, **kwargs) -> None:
        """
        Adding dataset to CISM pipeline as input

        :param dataset_folder: the relative filesystem sub-folder of the dataset
        :param dataset_type: the dataset type
        :param dataset_name: the dataset name

        Keyword Args:
            n_jobs (int): number of jobs allows running in parallel.
            prefer (string): joblib Parallel prefer parameter.
            force_run_fanmod (bool): whether to override fanmod+ existing results.
            force_parse (bool): whether to override existing parse files of fanmod+ outputs.

        :return: None
        """
        n_jobs = kwargs.setdefault("n_jobs", 16)
        prefer = kwargs.setdefault("prefer", 'threads')
        force_run_fanmod = kwargs.setdefault("force_run_fanmod", False)
        force_parse = kwargs.setdefault("force_parse", False)
        p_value = kwargs.setdefault("p_value", 0.05)
        quantile_threshold = kwargs.setdefault("quantile_threshold", 1)

        motifs_dataset = None
        set_loky_pickler()

        for r, d, files in os.walk(self.network_dataset_root_path + dataset_folder):
            Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(self._analyze_dataset)(
                                                                        file=file,
                                                                        output_dir=self.fanmod_output_root_path + dataset_folder,
                                                                        cache_dir=self.fanmod_cache_root_path + dataset_folder,
                                                                        force_run_fanmod=force_run_fanmod,
                                                                        raw_data_folder=self.network_dataset_root_path + dataset_folder,
                                                                        force_parse=force_parse,
                                                                        enable_parse=False) for file in tqdm(files))

            result_list = Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(delayed(self._analyze_dataset)(
                                                                        file=file,
                                                                        output_dir=self.fanmod_output_root_path + dataset_folder,
                                                                        cache_dir=self.fanmod_cache_root_path + dataset_folder,
                                                                        force_run_fanmod=False,
                                                                        raw_data_folder=self.network_dataset_root_path + dataset_folder,
                                                                        force_parse=force_parse,
                                                                        enable_parse=True,
                                                                        p_value=p_value,
                                                                        quantile_threshold=quantile_threshold) for file in tqdm(files))

            motifs_dataset = pd.concat(result_list, ignore_index=True)
            motifs_dataset['FOV'] = motifs_dataset['FOV'].astype('category')
            result = None

            result_list = None
            motifs_dataset[dataset_type] = dataset_name
            motifs_dataset['Patient_uId'] = motifs_dataset[dataset_type] + motifs_dataset['Patient'].astype('str')
            motifs_dataset[dataset_type] = motifs_dataset[dataset_type].astype('category')
            motifs_dataset['nunique_colors'] = motifs_dataset.nunique_colors.astype('int8')
            motifs_dataset['Patient_uId'] = motifs_dataset['Patient_uId'].astype('category')
            motifs_dataset['FOV'] = motifs_dataset['FOV'].astype('category')

        self.motifs_dataset = motifs_dataset if self.motifs_dataset is None else pd.concat([self.motifs_dataset, motifs_dataset])
        self.motifs_dataset.reset_index(drop=True)

    def motif_dataset(self) -> pd.DataFrame:
        """
        returns the raw dataframe for manual exploration

        :return: motifs dataset
        """
        return self.motifs_dataset

    def get_patients_ids(self, classes: list) -> List[str]:
        if classes is None:
            classes = self.motifs_dataset[DiscriminativeMotifs.PATIENT_CLASS].unique()

        return (self.motifs_dataset[self.motifs_dataset[DiscriminativeMotifs.PATIENT_CLASS]
                .isin(classes)]['Patient_uId'].unique())

    def _analyze_dataset(self, **kwargs):
        return helpers.analyze_file(
            FANMOD_exe=self.fanmod_exe,
            FANMOD_path=self.fanmod_path,
            iterations=self.iterations,
            motif_size=self.motif_size,
            **kwargs)


class AnalyzeMotifsResult:
    def __init__(self, analyze_results: list, patients_ids: list, labels: list):
        self.results = pd.DataFrame(analyze_results,
                                    columns=['TP', 'TN', 'FN', 'FP', 'cFeatures', 'prob', 'class', 'pred_class', 'classes', 'contributions', 'shape_values'],
                                    index=patients_ids)
        self.labels = labels

    def get_metrics(self):
        from cism.evaluate_aux import get_metrics
        return get_metrics(df=self.results)

    def get_roc_auc_score(self):
        from cism.evaluate_aux import get_roc_auc_score
        prob_vec = [row['prob'][0][np.where(row['classes'] == self.labels[0])] for idx, row in self.results.iterrows()]
        return get_roc_auc_score(true_y=self.results['class'].tolist(),
                                 y_prob=prob_vec,
                                 pos_label=self.labels[0])

    def plot_precision_recall_curve(self) -> float:
        from cism.evaluate_aux import plot_precision_recall_curve
        prob_vec = [row['prob'][0][np.where(row['classes'] == self.labels[0])] for idx, row in self.results.iterrows()]
        return plot_precision_recall_curve(true_y=self.results['class'],
                                           y_prob=prob_vec,
                                           pos_label=self.labels[0])

    def plot_roc_curve(self) -> float:
        from cism.evaluate_aux import plot_roc_curve
        prob_vec = [row['prob'][0][np.where(row['classes'] == self.labels[0])] for idx, row in self.results.iterrows()]
        return plot_roc_curve(true_y=self.results['class'],
                              y_prob=prob_vec,
                              pos_label=self.labels[0])

    def to_csv(self, file_path: str):
        self.results.to_csv(file_path)

    def read_csv(self, file_path: str):
        import ast
        self.results = pd.read_csv(file_path)
        self.results = self.results.prob.transform(lambda x: ast.literal_eval(x.replace(' ', ',')))


class DiscriminativeFeatureKey(enum.Enum):
    STRUCTURE_AND_CELL_IDENTITIES = 'ID'
    CELL_IDENTITIES = 'colors_vec_hash'


class FeatureConfiguration:
    def __init__(self,
                 labels: list,
                 fuzzy_match: bool = False,
                 top_n_similar: int = 0,
                 fuzzy_match_exclude_original: bool = False,
                 cell_type_composition_patient_map: Dict[str, List] = None,
                 motifs_patient_map: Dict[str, List] = None):
        self.labels = labels
        self.fuzzy_match = fuzzy_match
        self.top_n_similar = top_n_similar
        self.fuzzy_match_exclude_original = fuzzy_match_exclude_original
        self.cell_type_composition_patient_map = cell_type_composition_patient_map
        self.motifs_patient_map = motifs_patient_map

        # currently, we support only binary classification,
        # but we can change that in the future
        if len(labels) != 2:
            raise Exception("currently, we support only binary classification")

class HardDiscriminativeFC(FeatureConfiguration):
    def __init__(self,
                 labels: list,
                 extract_by: DiscriminativeFeatureKey,
                 use_cells_type_composition: bool,
                 use_motifs: bool,
                 shared_percentage: float,
                 max_class_features: int,
                 fuzzy_match: bool = False,
                 top_n_similar: int = 0,
                 fuzzy_match_exclude_original: bool = False,
                 cell_type_composition_patient_map: Dict[str, List] = None,
                 motifs_patient_map: Dict[str, List] = None):
        super(HardDiscriminativeFC, self).__init__(labels,
                                                   fuzzy_match,
                                                   top_n_similar,
                                                   fuzzy_match_exclude_original,
                                                   cell_type_composition_patient_map,
                                                   motifs_patient_map)
        self.extract_by = extract_by
        self.use_cells_type_composition = use_cells_type_composition
        self.use_motifs = use_motifs
        self.motifs_ids = set()
        self.switch_cell_type_composition_with_motif = set()
        self.switch_cell_type_composition_with_motif_map = dict()
        self.switch_motif_with_cell_type_composition = set()
        self.motifs_ids_remove = set()
        self.cells_type_composition_vec = set()
        self.cells_type_composition_hash_remove = set()
        self.shared_percentage = shared_percentage
        self.max_class_features = max_class_features

    def include_motif(self, motif_ids: list):
        self.motifs_ids.add(motif_ids)

    def switch_cell_type_composition_hash_with_motif(self, cells_type_composition_hash: list):
        self.switch_cell_type_composition_with_motif.add(cells_type_composition_hash)

    def switch_cell_type_composition_hash_with_motif(self, hash, motif_ids: list):
        self.switch_cell_type_composition_with_motif_map[hash] = motif_ids

    def switch_motif_with_cell_type_composition(self, motif_ids: list):
        self.switch_motif_with_cell_type_composition.add(motif_ids)

    def exclude_motifs(self, motif_ids: list):
        self.motifs_ids_remove.add(motif_ids)

    def include_cells_type_composition(self, cells_type_composition_vec: list):
        self.cells_type_composition_vec.add(cells_type_composition_vec)

    def exclude_cells_type_composition(self, cells_type_composition_vec: list):
        self.cells_type_composition_hash_remove.add(cells_type_composition_vec)

    def get_shared_percentage(self):
        return self.shared_percentage


class SoftDiscriminativeFC(FeatureConfiguration):
    def __init__(self,
                 labels: list,
                 extract_by: DiscriminativeFeatureKey,
                 use_cells_type_composition: bool,
                 use_motifs: bool,
                 shared_percentage: float,
                 max_class_features: int,
                 fuzzy_match: bool = False,
                 top_n_similar: int = 0,
                 fuzzy_match_exclude_original: bool = False,
                 cell_type_composition_patient_map: Dict[str, List] = None,
                 motifs_patient_map: Dict[str, List] = None):
        super(SoftDiscriminativeFC, self).__init__(labels=labels,
                                                   fuzzy_match=fuzzy_match,
                                                   top_n_similar=top_n_similar,
                                                   fuzzy_match_exclude_original=fuzzy_match_exclude_original,
                                                   cell_type_composition_patient_map=cell_type_composition_patient_map,
                                                   motifs_patient_map=motifs_patient_map)

        if extract_by != DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES:
            raise Exception(f"SoftDiscriminativeFC not support discrimination by {extract_by.value}")

        self.extract_by = extract_by
        self.use_cells_type_composition = use_cells_type_composition
        self.use_motifs = use_motifs
        self.shared_percentage = shared_percentage
        self.max_class_features = max_class_features
        self.switch_cell_type_composition_with_motif = set()
        self.switch_cell_type_composition_with_motif_map = dict()
        self.switch_motif_with_cell_type_composition = set()
        self.motifs_ids_remove = set()
        self.cells_type_composition_vec = set()
        self.cells_type_composition_hash_remove = set()
        self.motifs_ids = set()

    def get_shared_percentage(self):
        return self.shared_percentage

    def include_motif(self, motif_ids: list):
        self.motifs_ids.add(motif_ids)

    def switch_cell_type_composition_hash_with_motif(self, cells_type_composition_hash: list):
        self.switch_cell_type_composition_with_motif.add(cells_type_composition_hash)

    def switch_cell_type_composition_hash_with_motif(self, hash, motif_ids: list):
        self.switch_cell_type_composition_with_motif_map[hash] = motif_ids

    def switch_motif_with_cell_type_composition(self, motif_ids: list):
        self.switch_motif_with_cell_type_composition.add(motif_ids)

    def exclude_motifs(self, motif_ids: list):
        self.motifs_ids_remove.add(motif_ids)

    def include_cells_type_composition(self, cells_type_composition_vec: list):
        self.cells_type_composition_vec.add(cells_type_composition_vec)

    def exclude_cells_type_composition(self, cells_type_composition_vec: list):
        self.cells_type_composition_hash_remove.add(cells_type_composition_vec)


class TopNFC(FeatureConfiguration):
    def __init__(self,
                 labels: list,
                 top_n: int,
                 cell_type_composition_patient_map: Dict[str, List] = None,
                 motifs_patient_map: Dict[str, List] = None):
        super(TopNFC, self).__init__(labels=labels,
                                     cell_type_composition_patient_map=cell_type_composition_patient_map,
                                     motifs_patient_map=motifs_patient_map)
        self.top_n = top_n


class InferenceFC(FeatureConfiguration):
    def __init__(self,
                 labels: list,
                 motifs_ids: list,
                 cell_type_composition_patient_map: Dict[str, List] = None,
                 motifs_patient_map: Dict[str, List] = None):
        super(InferenceFC, self).__init__(labels=labels,
                                          cell_type_composition_patient_map=cell_type_composition_patient_map,
                                          motifs_patient_map=motifs_patient_map)
        self.motifs_ids = motifs_ids


class DiscriminativeMotifs:
    PATIENT_CLASS = 'patient_class'
    PATIENT_CLASS_ID = 'patient_class_id'
    PATIENT_PERCENTAGE_KEY = 'patient_percentage'
    PATIENT_COUNT_KEY = 'patient_count'


class DiscoverResult:
    def __init__(self,
                 extract_by: DiscriminativeFeatureKey,
                 common_cells: dict,
                 discriminative_motifs: pd.DataFrame):
        self.discriminative_motifs = discriminative_motifs
        self.extract_by = extract_by
        self.common_cells = common_cells

    def get_discriminative_motifs(self) -> pd.DataFrame:
        return self.discriminative_motifs

    def plot_clustermap(self, percentage_th: float, class_to_color: dict, index_by: str):
        self._plot_clustermap(percentage_th=percentage_th,
                              class_to_color=class_to_color,
                              index_by=index_by)

    def plot_number_of_motifs_versus_discrimination_stringency_parameter(self, class_to_color: dict):
        self._plot_number_of_motifs_versus_shared_percentage(class_to_color)

    def plot_number_of_motifs_versus_shared_percentage(self, class_to_color: dict):
        self._plot_number_of_motifs_versus_shared_percentage(class_to_color)

    def _plot_clustermap(self, percentage_th: float, class_to_color: dict, index_by: str):
        from fastcluster import linkage as fast_linkage
        import seaborn as sns

        class_to_color_func: typing.Callable[[int], str] = lambda class_id: class_to_color[class_id]
        temp_a = (self.discriminative_motifs[self.discriminative_motifs[DiscriminativeMotifs.PATIENT_PERCENTAGE_KEY] >
                                             percentage_th])
        temp_a = temp_a[~temp_a[self.extract_by.value].duplicated(keep='first')]
        temp_a = temp_a.sort_values(DiscriminativeMotifs.PATIENT_CLASS)
        temp_a = temp_a.set_index(index_by)

        clustemap_data = pd.DataFrame([row.tolist() for row in temp_a.colors_vec],
                                      columns=list(self.common_cells.values()),
                                      index=temp_a.index)

        linkage_data = fast_linkage(clustemap_data, 'single', metric=lambda x, y: sum(
            ((x - y) ** 2).astype(np.float32) / (x + y + 0.000001).astype(np.float32)))

        fg = sns.clustermap(clustemap_data.transpose(), row_cluster=False, row_linkage=linkage_data, figsize=(20, 5),
                            yticklabels=True, xticklabels=True)

        get_color_func = lambda motif_id: class_to_color_func(temp_a.loc[motif_id][DiscriminativeMotifs.PATIENT_CLASS])

        x_labels = fg.ax_heatmap.get_xmajorticklabels()
        for x in x_labels:
            x.set_color(get_color_func(int(x.get_text())))

        self._plot_color_legend(class_to_color=class_to_color)

        plt.show()

    @staticmethod
    def _plot_color_legend(class_to_color: dict):
        from matplotlib.patches import Patch

        color_to_class = {y: x for x, y in class_to_color.items()}

        handles = [Patch(facecolor=color) for color, clazz in color_to_class.items()]
        plt.legend(handles, class_to_color, title='Group',
                   bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

    def _plot_number_of_motifs_versus_shared_percentage(self, class_to_color: dict):
        import seaborn as sns

        ax = sns.ecdfplot(data=self.discriminative_motifs.groupby('ID', observed=True).first(),
                          x=DiscriminativeMotifs.PATIENT_PERCENTAGE_KEY, hue=DiscriminativeMotifs.PATIENT_CLASS,
                          log_scale=(False, True), complementary=True, stat="count", palette=class_to_color)

        ax.set_xlabel('Discrimination stringency parameter')
        ax.set_ylabel('Number of discriminative motifs (log scale)')
        ax.set_xlim([0, 1])


class GetMotifsResult:
    def __init__(self, motif_space_features, discriminator, classes):
        self.motif_space_features = motif_space_features
        self.discriminator = discriminator
        self.classes = classes
        self.all_motif_features = None
        self.motif_to_cells_identity_hash = None
        self.cell_identity_to_motif_hash = None

        self._process_result()

    def _process_result(self):
        all_motif_features = []
        for idx, row in self.motif_space_features.iterrows():
            all_motif_features += row['features']
        self.all_motif_features = list(set(all_motif_features))

        self.motif_to_cells_identity_hash = {}
        for motif_id in all_motif_features:
            self.motif_to_cells_identity_hash[motif_id] = self.discriminator.cism.motifs_dataset[
                self.discriminator.cism.motifs_dataset.ID == motif_id].colors_vec_hash.iloc[0]

        self.cell_identity_to_motif_hash = {}
        for k, v in self.motif_to_cells_identity_hash.items():
            self.cell_identity_to_motif_hash[v] = self.cell_identity_to_motif_hash.get(v, []) + [k]

    def get_motifs_by_class(self):
        from collections import Counter

        group_a_motifs = []
        group_b_motifs = []

        discriminator = self.discriminator

        relevant_patients = discriminator.get_patients_class(self.classes)
        for motif_id in self.all_motif_features:
            patients_ids = discriminator.cism.motifs_dataset[
                (discriminator.cism.motifs_dataset.ID == motif_id) &
                (discriminator.cism.motifs_dataset['Patient_uId'].isin(relevant_patients.index))][
                'Patient_uId'].unique()
            groups_ids = map(lambda x: relevant_patients.loc[x]['patient_class'], patients_ids)
            counter = Counter(groups_ids)
            if max(counter, key=counter.get) == self.classes[0]:
                group_a_motifs.append(motif_id)
            else:
                group_b_motifs.append(motif_id)

        return {self.classes[0]: group_a_motifs, self.classes[1]: group_b_motifs}

    def get_motifs_mean_weight(self):
        discriminator = self.discriminator
        relevant_patients = discriminator.get_patients_class(self.classes)

        discriminator.cism.motifs_dataset['Total'] = discriminator.cism.motifs_dataset['Count'] / \
                                                     discriminator.cism.motifs_dataset['Freq']

        new_freq_table = discriminator.cism.motifs_dataset[
            (discriminator.cism.motifs_dataset.ID.isin(self.all_motif_features)) &
            (discriminator.cism.motifs_dataset['Patient_uId'].isin(relevant_patients.index))].groupby(['Patient', 'ID'],
                                                                                                      observed=True)[
            ['Count', 'Total']].sum().reset_index()

        new_freq_table['Freq'] = new_freq_table['Count'] / new_freq_table['Total']
        return new_freq_table.groupby('ID')['Freq'].mean()


class TissueStateDiscriminativeMotifs(DiscriminativeMotifs):
    def __init__(self,
                 cism: CISM,
                 tissue_state_csv_path: str,
                 tissue_state_to_string: Dict[int, List],
                 common_cells_type: Dict[int, str],
                 tissue_state_func=None):
        self.cism = cism
        self._add_cell_identity_feature(data=self.cism.motifs_dataset, common_cell_type=common_cells_type)
        self.patient_class_df = self._load_tissue_state(tissue_state_csv_path=tissue_state_csv_path,
                                                        tissue_state_to_string=tissue_state_to_string,
                                                        tissue_state_func=tissue_state_func)
        self.common_cells = common_cells_type

    def get_patients_class(self, classes: list=None) -> pd.DataFrame:
        if classes is None:
            classes = self.patient_class_df.patient_class.unique()
        exist_patients = self.cism.motifs_dataset.Patient_uId.unique()
        return self.patient_class_df[self.patient_class_df.patient_class.isin(classes) &
                                     self.patient_class_df.index.isin(exist_patients)]

    def discover(self, extract_by: DiscriminativeFeatureKey, classes: list=None):
        patient_class_dict = self.patient_class_df[DiscriminativeMotifs.PATIENT_CLASS].to_dict()
        self.cism.motifs_dataset[DiscriminativeMotifs.PATIENT_CLASS] = self.cism.motifs_dataset.Patient_uId.transform(
            lambda row: patient_class_dict[row]).astype('category')

        discriminative_motifs, _ = self._extract_discriminative(
            x_data=self.cism.motifs_dataset[
                self.cism.motifs_dataset[
                    DiscriminativeMotifs.PATIENT_CLASS].isin(classes)],
            discriminative_group_key=DiscriminativeMotifs.PATIENT_CLASS,
            discriminative_feature_key=extract_by.value,
            common_cell_type=list(self.common_cells.keys())
        )
        return DiscoverResult(discriminative_motifs=discriminative_motifs,
                              extract_by=extract_by,
                              common_cells=self.common_cells)

    def patient_class_permutation_test(self,
                                       feature_conf: FeatureConfiguration,
                                       rand_patient_class: bool = True,
                                       **kwargs):
        n_permutations = kwargs.setdefault("n_permutations", 1000)
        n_jobs = kwargs.setdefault("n_jobs", 1)
        prefer = kwargs.setdefault("prefer", 'processes')

        if not isinstance(feature_conf, HardDiscriminativeFC):
            raise Exception(f"unsupported feature configuration type {type(feature_conf)}")

        result_list: typing.List[pd.DataFrame] = Parallel(n_jobs=n_jobs, verbose=0, prefer=prefer)(
            delayed(self._patient_class_permutation_test)(feature_conf=feature_conf,
                                                          rand_patient_class=rand_patient_class)
            for trial_i in tqdm(range(n_permutations)))

        final_results_list = []
        for records in result_list:
            agg_features = []
            for idx, row in records.iterrows():
                agg_features.extend(row['features'])
            agg_features = list(agg_features)
            final_results_list.append(len(agg_features))

        return final_results_list

    def _patient_class_permutation_test(self,
                                        feature_conf: HardDiscriminativeFC,
                                        rand_patient_class: bool,
                                        n_jobs: int = 1,
                                        **kwargs) -> pd.DataFrame:
        random_state = kwargs.setdefault("random_state", np.random)

        unique_patients_ids = self.get_patients_class(feature_conf.labels).index.unique()

        local_patient_class = self.patient_class_df[
            self.patient_class_df[DiscriminativeMotifs.PATIENT_CLASS].isin(feature_conf.labels)].copy()

        if rand_patient_class:
            local_patient_class[DiscriminativeMotifs.PATIENT_CLASS] = random_state.permutation(
                local_patient_class[DiscriminativeMotifs.PATIENT_CLASS])

        def __get_motif_dataset_of_patient(motifs_dataset: pd.DataFrame,
                                           unique_patients_ids,
                                           trial_i: int,
                                           train: bool):
            if train:
                return motifs_dataset[motifs_dataset.Patient_uId.isin(unique_patients_ids) &
                                      (motifs_dataset.Patient_uId != unique_patients_ids[trial_i]) &
                                      (motifs_dataset.p_value < 0.05)].copy()
            else:
                return motifs_dataset[motifs_dataset.Patient_uId.isin(unique_patients_ids) &
                                      (motifs_dataset.Patient_uId == unique_patients_ids[trial_i]) &
                                      (motifs_dataset.p_value < 0.05)].copy()

        raw_get_features_result = Parallel(n_jobs=n_jobs, verbose=0, prefer='threads')(
            delayed(self._get_features)(
                                    x_data=__get_motif_dataset_of_patient(self.cism.motifs_dataset,
                                                                              unique_patients_ids,
                                                                              trial_i,
                                                                              True),
                                    x_test=__get_motif_dataset_of_patient(self.cism.motifs_dataset,
                                                                          unique_patients_ids,
                                                                          trial_i,
                                                                          False),
                                    test_patient_uid=unique_patients_ids[trial_i],
                                    feature_conf=feature_conf,
                                    patient_class_df=local_patient_class) for trial_i in tqdm(range(len(unique_patients_ids))))

        records = [{'test_patient_id': test_patient_id,
                    'features': features} for test_patient_id, features, _ in raw_get_features_result]

        records = pd.DataFrame.from_records(records)

        # count the number of features per patient
        records['count'] = records['features'].transform(lambda x: len(x))

        return records

    def analyze_motifs(self, feature_conf: FeatureConfiguration, exclude_patients: list, **kwargs) -> AnalyzeMotifsResult:
        n_jobs = kwargs.setdefault("n_jobs", 8)
        prefer = kwargs.setdefault("prefer", 'processes')
        random_state = kwargs.setdefault("random_state",  np.random.RandomState())
        rand_patient_class = kwargs.setdefault("rand_patient_class", False)
        rand_motifs = kwargs.setdefault("rand_motifs", False)

        self._add_color_vec_hash()

        unique_patients_ids = self.get_patients_class(feature_conf.labels).index.unique()
        for patient_id in exclude_patients:
            unique_patients_ids = np.delete(unique_patients_ids, np.where(np.array(unique_patients_ids) == patient_id))

        local_patient_class = self.patient_class_df[self.patient_class_df.patient_class.isin(feature_conf.labels)].copy()

        if rand_patient_class:
            local_patient_class[DiscriminativeMotifs.PATIENT_CLASS] = random_state.permutation(
                local_patient_class[DiscriminativeMotifs.PATIENT_CLASS])

        local_motifs_dataset = self.cism.motifs_dataset

        if rand_motifs:
            local_motifs_dataset = local_motifs_dataset[local_motifs_dataset.Patient_uId.isin(unique_patients_ids)]
            local_motifs_dataset = local_motifs_dataset.copy()
            local_motifs_dataset['ID'] = random_state.permutation(local_motifs_dataset['ID'])

        def __get_motif_dataset_of_patient(motifs_dataset: pd.DataFrame,
                                           unique_patients_ids,
                                           trial_i: int,
                                           train: bool):
            if train:
                return motifs_dataset[motifs_dataset.Patient_uId.isin(unique_patients_ids) &
                                      (motifs_dataset.Patient_uId != unique_patients_ids[trial_i]) &
                                      (motifs_dataset.p_value < 0.05)].copy()
            else:
                return motifs_dataset[motifs_dataset.Patient_uId.isin(unique_patients_ids) &
                                      (motifs_dataset.Patient_uId == unique_patients_ids[trial_i]) &
                                      (motifs_dataset.p_value < 0.05)].copy()

        raw_analyze_result = Parallel(n_jobs=n_jobs, verbose=0, prefer=prefer, pre_dispatch=1)(
            delayed(self._validate)(x_data=__get_motif_dataset_of_patient(local_motifs_dataset,
                                                                          unique_patients_ids,
                                                                          trial_i,
                                                                          True),
                                    x_test=__get_motif_dataset_of_patient(local_motifs_dataset,
                                                                          unique_patients_ids,
                                                                          trial_i,
                                                                          False),
                                    test_patient_uid=unique_patients_ids[trial_i],
                                    feature_conf=feature_conf,
                                    random_state=random_state,
                                    patient_class_df=local_patient_class) for trial_i in tqdm(range(len(unique_patients_ids))))

        return AnalyzeMotifsResult(analyze_results=raw_analyze_result,
                                   patients_ids=unique_patients_ids,
                                   labels=feature_conf.labels)

    def get_features(self, feature_conf: FeatureConfiguration, exclude_patients: list, **kwargs) -> pd.DataFrame:
        n_jobs = kwargs.setdefault("n_jobs", 8)
        prefer = kwargs.setdefault("prefer", 'processes')

        self._add_color_vec_hash()

        unique_patients_ids = self.get_patients_class(feature_conf.labels).index.unique()
        for patient_id in exclude_patients:
            unique_patients_ids = np.delete(unique_patients_ids, np.where(np.array(unique_patients_ids) == patient_id))

        def __get_motif_dataset_of_patient(motifs_dataset: pd.DataFrame,
                                           unique_patients_ids,
                                           trial_i: int,
                                           train: bool):
            if train:
                return motifs_dataset[motifs_dataset.Patient_uId.isin(unique_patients_ids) &
                                      (motifs_dataset.Patient_uId != unique_patients_ids[trial_i]) &
                                      (motifs_dataset.p_value < 0.05)].copy()
            else:
                return motifs_dataset[motifs_dataset.Patient_uId.isin(unique_patients_ids) &
                                      (motifs_dataset.Patient_uId == unique_patients_ids[trial_i]) &
                                      (motifs_dataset.p_value < 0.05)].copy()

        raw_get_features_result = Parallel(n_jobs=n_jobs, verbose=0, prefer=prefer)(
            delayed(self._get_features)(x_data=__get_motif_dataset_of_patient(self.cism.motifs_dataset,
                                                                              unique_patients_ids,
                                                                              trial_i,
                                                                              True),
                                    x_test=__get_motif_dataset_of_patient(self.cism.motifs_dataset,
                                                                          unique_patients_ids,
                                                                          trial_i,
                                                                          False),
                                    test_patient_uid=unique_patients_ids[trial_i],
                                    feature_conf=feature_conf) for trial_i in tqdm(range(len(unique_patients_ids))))

        records = [{'test_patient_id': test_patient_id,
                    'features': features} for test_patient_id, features, _ in raw_get_features_result]

        return pd.DataFrame.from_records(records)

    @staticmethod
    def _load_tissue_state(tissue_state_csv_path: str, tissue_state_to_string: Dict[int, str], tissue_state_func=None) -> pd.DataFrame:
        patient_class_df = pd.read_csv(tissue_state_csv_path, index_col=0, names=['patient_class_id'])

        if tissue_state_func:
            patient_class_df[DiscriminativeMotifs.PATIENT_CLASS] = (patient_class_df['patient_class_id']
                                                                    .transform(tissue_state_func, axis=0))
        else:
            # we remove spaces since later on the group-by optimization depends on that.
            patient_class_df[DiscriminativeMotifs.PATIENT_CLASS] = patient_class_df['patient_class_id'].transform(
                lambda row: tissue_state_to_string[row].replace(" ", ""), axis=0)
        return patient_class_df

    def _add_color_vec_hash(self):
        if 'colors_vec' in self.cism.motifs_dataset.columns:
            # skip if already exist
            return

        # create color vec and color vec hash columns
        common_cell_type_counter = dict(
            zip([str(x) for x in self.common_cells], [0 for _ in range(len(self.common_cells))]))

        def create_colors_vector(motif: nx.Graph):
            base_counter = Counter(common_cell_type_counter)
            base_counter.update(nx.get_node_attributes(motif, name='type').values())
            return np.array(list(base_counter.values()))

        self.cism.motifs_dataset['colors_vec'] = self.cism.motifs_dataset.motif.transform(
            lambda row: create_colors_vector(row))
        self.cism.motifs_dataset['colors_vec_hash'] = self.cism.motifs_dataset.colors_vec.transform(
            lambda row: hash(tuple(row)))
        self.cism.motifs_dataset['colors_vec_hash'] = self.cism.motifs_dataset.colors_vec_hash.astype('int64')

    def _extract_discriminative(self,
                                x_data: pd.DataFrame,
                                discriminative_group_key: str,
                                discriminative_feature_key: str,
                                common_cell_type: list,
                                min_nunique_colors: int = 1,
                                min_patients: int = 1,
                                p_value: float = 0.05) -> typing.Tuple[pd.DataFrame, int]:
        df_copy = x_data[x_data.p_value < p_value].copy()

        special_hash_group = (df_copy.drop_duplicates([discriminative_feature_key, discriminative_group_key])
                              .groupby(discriminative_feature_key, observed=True)[discriminative_group_key]
                              .nunique().reset_index())
        one_class_data = (df_copy[df_copy[discriminative_feature_key]
                          .isin(special_hash_group[special_hash_group[discriminative_group_key] == 1][discriminative_feature_key])
                                  & (df_copy.nunique_colors >= min_nunique_colors)].copy())

        one_class_data.loc[:, DiscriminativeMotifs.PATIENT_COUNT_KEY] = one_class_data.groupby(discriminative_feature_key, observed=True)['Patient_uId'].transform('nunique')
        one_class_num_motifs = (one_class_data[(one_class_data[DiscriminativeMotifs.PATIENT_COUNT_KEY] > min_patients)]
                                .sort_values('Freq', ascending=False).drop_duplicates(subset=[discriminative_feature_key]).shape[0])

        unique_classes = df_copy.groupby(discriminative_group_key, observed=True).Patient_uId.nunique()
        for class_index in range(len(unique_classes.index)):
            class_name = unique_classes.index[class_index]
            one_class_data.loc[one_class_data[discriminative_group_key] == class_name, DiscriminativeMotifs.PATIENT_PERCENTAGE_KEY] = (
                    one_class_data.loc[one_class_data[discriminative_group_key] == class_name, DiscriminativeMotifs.PATIENT_COUNT_KEY] /
                    unique_classes[class_name])

        return one_class_data[(one_class_data[DiscriminativeMotifs.PATIENT_COUNT_KEY] >= min_patients)], one_class_num_motifs

    @staticmethod
    def _sort_features_single(data: pd.DataFrame, motif_ids, labels: list) -> list:
        from scipy.stats import wasserstein_distance

        results = []

        for motif_id in motif_ids:
            group_a = data[(data.ID == motif_id) & (data[DiscriminativeMotifs.PATIENT_CLASS] == labels[0])]
            group_b = data[(data.ID == motif_id) & (data[DiscriminativeMotifs.PATIENT_CLASS] == labels[1])]
            if len(group_a.Freq) == 0 and len(group_b.Freq) == 0:
                wd_score = 0
            elif len(group_a.Freq) == 0 or len(group_b.Freq) == 0:
                wd_score = float('inf')
            else:
                wd_score = wasserstein_distance(group_a.Freq, group_b.Freq)

            group_a_size = group_a.Patient.nunique()
            group_b_size = group_b.Patient.nunique()

            results.append(pd.DataFrame([{'ID': motif_id,
                             'wasserstein_distance': wd_score,
                             'group_a_size': group_a_size,
                             'group_b_size': group_b_size,
                             'group_size_max': max(group_a_size, group_b_size)}]))

        return results

    @staticmethod
    def _sort_features(data: pd.DataFrame, labels: list, n_jobs: int = 16) -> pd.DataFrame:
        wd_results = []
        wd_results = Parallel(n_jobs=n_jobs, prefer='processes', return_as="generator")(
            delayed(TissueStateDiscriminativeMotifs._sort_features_single)(
                data=data[data.ID.isin(motif_ids)][['ID', 'Freq', DiscriminativeMotifs.PATIENT_CLASS, 'Patient']].copy(),
                motif_ids=motif_ids,
                labels=labels) for motif_ids in np.array_split(data.ID.unique(), n_jobs))

        from itertools import chain
        wd_results = list(chain.from_iterable(wd_results))

        wd_results = pd.concat(wd_results, ignore_index=True)
        return wd_results.sort_values(by='wasserstein_distance', ascending=False)

    @staticmethod
    def _extract_features(one_class_data: pd.DataFrame, feature_conf: FeatureConfiguration) -> [list, list, int]:
        # build x data based on x discriminative motifs
        unique_motifs_colors = []
        unique_motifs = []
        shared_percentage = feature_conf.get_shared_percentage()

        one_class_data_filter = one_class_data[(one_class_data[DiscriminativeMotifs.PATIENT_PERCENTAGE_KEY] >
                                                shared_percentage)]

        if isinstance(feature_conf, SoftDiscriminativeFC):
            sort_by = 'wasserstein_distance'
            wd_results = TissueStateDiscriminativeMotifs._sort_features(data=one_class_data_filter, labels=feature_conf.labels)
            one_class_data_filter = pd.merge(one_class_data_filter, wd_results, on='ID')
            one_class_data_filter = one_class_data_filter[one_class_data_filter['wasserstein_distance'] >= 0.05]
            one_class_data_filter = one_class_data_filter.sort_values(by=sort_by, ascending=False)
            one_class_data_filter = one_class_data_filter.drop_duplicates('ID')
        else:
            sort_by = 'Freq'
            # motif_mean_freq = one_class_data_filter.groupby(by=['ID'], observed=True)[sort_by].median().reset_index()
            # motif_mean_freq.columns = ['ID', 'Motif_Mean_Freq']
            # one_class_data_filter = pd.merge(one_class_data_filter, motif_mean_freq, on=['ID'])
            one_class_data_filter = one_class_data_filter.sort_values(by=sort_by, ascending=False)

        if feature_conf.use_cells_type_composition:
            unique_motifs_colors = one_class_data_filter
            unique_motifs_colors = (unique_motifs_colors.drop_duplicates(subset=[DiscriminativeMotifs.PATIENT_CLASS,
                                                                                 'colors_vec_hash'])
                                    .groupby(DiscriminativeMotifs.PATIENT_CLASS, observed=True)
                                    .head(feature_conf.max_class_features)
                                    .reset_index(drop=True).colors_vec_hash.unique())
            unique_motifs_colors = unique_motifs_colors.tolist()
        else:
            for hash in feature_conf.cells_type_composition_vec:
                unique_motifs_colors.append(hash)

        if feature_conf.use_motifs:
            unique_motifs = one_class_data_filter
            unique_motifs = (unique_motifs.groupby(DiscriminativeMotifs.PATIENT_CLASS, observed=True)
                             .head(feature_conf.max_class_features)).reset_index(drop=True).ID.unique()
            unique_motifs = unique_motifs.tolist()

        for motif_id in feature_conf.switch_motif_with_cell_type_composition:
            if motif_id not in unique_motifs:
                continue
            unique_motifs_colors.extend(one_class_data_filter[one_class_data_filter.ID == motif_id]
                                        .colors_vec_hash.unique().tolist())

        for hash in feature_conf.switch_cell_type_composition_with_motif:
            if hash not in unique_motifs_colors:
                continue
            unique_motifs = (one_class_data_filter[one_class_data_filter.colors_vec_hash == hash]
                             .ID.unique().tolist())

        for hash, motif_ids in feature_conf.switch_cell_type_composition_with_motif_map.items():
            if hash not in unique_motifs_colors:
                continue
            unique_motifs.extend(motif_ids)

        for hash in feature_conf.cells_type_composition_hash_remove:
            unique_motifs_colors = np.delete(unique_motifs_colors, np.where(np.array(unique_motifs_colors) == hash))
            unique_motifs_colors = unique_motifs_colors.tolist()

        for motif_id in feature_conf.motifs_ids_remove:
            unique_motifs = np.delete(unique_motifs, np.where(np.array(unique_motifs) == motif_id))
            unique_motifs = unique_motifs.tolist()

        unique_motifs_colors = list(set(unique_motifs_colors))

        c_features = len(unique_motifs_colors) + len(unique_motifs)

        return unique_motifs_colors, unique_motifs, c_features

    def _get_features(self,
                  x_data: pd.DataFrame,
                  x_test: pd.DataFrame,
                  test_patient_uid: str,
                  feature_conf: FeatureConfiguration,
                  patient_class_df: pd.DataFrame = None):

        local_patient_class = self.patient_class_df.copy() if patient_class_df is None else patient_class_df.copy()

        patient_class_dict = local_patient_class[DiscriminativeMotifs.PATIENT_CLASS].to_dict()
        patient_class_df = pd.DataFrame.from_dict(patient_class_dict,
                                                  orient='index',
                                                  columns=[DiscriminativeMotifs.PATIENT_CLASS])
        x_data = (pd.merge(x_data.drop([DiscriminativeMotifs.PATIENT_CLASS], axis=1, errors='ignore'),
                           patient_class_df,
                           left_on='Patient_uId', right_index=True)
                  .astype({DiscriminativeMotifs.PATIENT_CLASS: 'category'}))

        if isinstance(feature_conf, TopNFC):
            unique_motifs = x_data.groupby('ID', observed=True)['Freq'].mean().reset_index().sort_values('Freq', ascending=False).head(
                feature_conf.top_n).ID.tolist()
            unique_motifs_colors = []
            c_features = len(unique_motifs_colors) + len(unique_motifs)
        elif isinstance(feature_conf, HardDiscriminativeFC):
            one_class_data, _ = self._extract_discriminative(
                                    x_data=x_data,
                                    discriminative_group_key=DiscriminativeMotifs.PATIENT_CLASS,
                                    discriminative_feature_key=feature_conf.extract_by.value,
                                    common_cell_type=list(self.common_cells.keys()),
                                    min_nunique_colors=1,
                                    min_patients=1,
                                    p_value=0.05)
            unique_motifs_colors, unique_motifs, c_features = self._extract_features(one_class_data=one_class_data,
                                                                                     feature_conf=feature_conf)
        elif isinstance(feature_conf, SoftDiscriminativeFC):
            unique_classes = x_data.groupby(DiscriminativeMotifs.PATIENT_CLASS, observed=True).Patient_uId.nunique()

            count_by_id = (x_data.groupby(['ID', DiscriminativeMotifs.PATIENT_CLASS], observed=True)['Patient_uId']
                                                                     .agg('nunique').reset_index())

            count_by_id = pd.merge(count_by_id,
                                   unique_classes.reset_index().rename({'Patient_uId': 'group_number'}, axis=1),
                                   left_on='patient_class', right_on='patient_class')
            count_by_id['Patient_uId'] = count_by_id['Patient_uId']/count_by_id['group_number']
            count_by_id = (count_by_id.groupby('ID', observed=True)['Patient_uId'].agg('max')
                           .reset_index().rename({'Patient_uId': DiscriminativeMotifs.PATIENT_PERCENTAGE_KEY}, axis=1))

            x_data = pd.merge(x_data, count_by_id, left_on='ID', right_on='ID')

            unique_motifs_colors, unique_motifs, c_features = self._extract_features(one_class_data=x_data,
                                                                                     feature_conf=feature_conf)
        else:
            raise Exception(f"unsupported feature configuration type {type(feature_conf)}")

        columns = unique_motifs_colors + unique_motifs

        return test_patient_uid, columns, c_features

    def _validate(self,
                  x_data: pd.DataFrame,
                  x_test: pd.DataFrame,
                  test_patient_uid: str,
                  feature_conf: FeatureConfiguration,
                  random_state: int = 0,
                  rand_patient_class: bool = False,
                  patient_class_df: pd.DataFrame = None):
        TP, TN, FP, FN = 0, 0, 0, 0

        local_patient_class = self.patient_class_df.copy() if patient_class_df is None else patient_class_df.copy()

        if rand_patient_class:
            local_patient_class[DiscriminativeMotifs.PATIENT_CLASS] = random_state.permutation(
                local_patient_class[DiscriminativeMotifs.PATIENT_CLASS])

        patient_class_dict = local_patient_class[DiscriminativeMotifs.PATIENT_CLASS].to_dict()
        patient_class_df = pd.DataFrame.from_dict(patient_class_dict,
                                                 orient='index',
                                                 columns=[DiscriminativeMotifs.PATIENT_CLASS])
        x_data = (pd.merge(x_data.drop([DiscriminativeMotifs.PATIENT_CLASS], axis=1, errors='ignore'),
                          patient_class_df,
                          left_on='Patient_uId', right_index=True)
                  .astype({DiscriminativeMotifs.PATIENT_CLASS:'category'}))

        if (feature_conf.cell_type_composition_patient_map is not None) and (feature_conf.motifs_patient_map is not None):

            unique_motifs_colors = feature_conf.cell_type_composition_patient_map[test_patient_uid]
            unique_motifs = feature_conf.motifs_patient_map[test_patient_uid]
            c_features = len(unique_motifs_colors) + len(unique_motifs)

        elif isinstance(feature_conf, TopNFC):
            unique_motifs = x_data.groupby('ID', observed=True)['Freq'].mean().reset_index().sort_values('Freq', ascending=False).head(
                feature_conf.top_n).ID.tolist()
            unique_motifs_colors = []
            c_features = len(unique_motifs_colors) + len(unique_motifs)
        elif isinstance(feature_conf, InferenceFC):
            unique_motifs = feature_conf.motifs_ids
            unique_motifs_colors = []
            c_features = len(unique_motifs_colors) + len(unique_motifs)
        elif isinstance(feature_conf, HardDiscriminativeFC):
            one_class_data, _ = self._extract_discriminative(
                x_data=x_data,
                discriminative_group_key=DiscriminativeMotifs.PATIENT_CLASS,
                discriminative_feature_key=feature_conf.extract_by.value,
                common_cell_type=list(self.common_cells.keys()),
                min_nunique_colors=1,
                min_patients=1,
                p_value=0.05)
            unique_motifs_colors, unique_motifs, c_features = self._extract_features(one_class_data=one_class_data,
                                                                                     feature_conf=feature_conf)
        elif isinstance(feature_conf, SoftDiscriminativeFC):
            unique_classes = x_data.groupby(DiscriminativeMotifs.PATIENT_CLASS, observed=True).Patient_uId.nunique()

            count_by_id = (x_data.groupby(['ID', DiscriminativeMotifs.PATIENT_CLASS], observed=True)['Patient_uId']
                           .agg('nunique').reset_index())

            count_by_id = pd.merge(count_by_id,
                                   unique_classes.reset_index().rename({'Patient_uId': 'group_number'}, axis=1),
                                   left_on='patient_class', right_on='patient_class')
            count_by_id['Patient_uId'] = count_by_id['Patient_uId'] / count_by_id['group_number']
            count_by_id = (count_by_id.groupby('ID', observed=True)['Patient_uId'].agg('max')
                           .reset_index().rename({'Patient_uId': DiscriminativeMotifs.PATIENT_PERCENTAGE_KEY}, axis=1))

            x_data = pd.merge(x_data, count_by_id, left_on='ID', right_on='ID')

            unique_motifs_colors, unique_motifs, c_features = self._extract_features(one_class_data=x_data,
                                                                                     feature_conf=feature_conf)
        else:
            raise Exception(f"unsupported feature configuration type {type(feature_conf)}")

        # if fuzzy is enabled mapping all the matched motifs for each motif
        fuzzy_match_map = dict()
        if feature_conf.fuzzy_match:
            for motif_id in unique_motifs:
                colors_vec_hash = x_data.colors_vec_hash.iloc[0]
                top_n = feature_conf.top_n_similar
                start = 1 if feature_conf.fuzzy_match_exclude_original else 0
                fuzzy_match_motifs = TissueStateDiscriminativeMotifs._get_top_similar(motif_id=motif_id,
                                                                                      color_vec_hash=colors_vec_hash,
                                                                                      motif_dataset=x_data,
                                                                                      top_n=top_n)[start:].ID.tolist()
                fuzzy_match_map[motif_id] = fuzzy_match_motifs

        one_class_data = x_data[x_data.p_value < 0.05]

        columns = unique_motifs_colors + unique_motifs + [DiscriminativeMotifs.PATIENT_CLASS]

        x_train_dataset = None
        patient_classes = list(local_patient_class[DiscriminativeMotifs.PATIENT_CLASS].unique())
        one_class_data_color_index = one_class_data.set_index(['Patient_uId', 'colors_vec_hash'])
        one_class_data_color_index.sort_index(inplace=True)
        one_class_data_index = one_class_data.set_index(['Patient_uId', 'ID'])
        one_class_data_index.sort_index(inplace=True)
        for patient_class in patient_classes:
            for patient_uId in one_class_data[one_class_data[DiscriminativeMotifs.PATIENT_CLASS] == patient_class]['Patient_uId'].unique():
                vector_dict = defaultdict()
                self._add_cell_type_composition_freq_feature(one_class_data_color_index, unique_motifs_colors, vector_dict, patient_uId)
                self._add_motif_freq_feature(one_class_data_index,
                                             unique_motifs,
                                             vector_dict,
                                             patient_uId,
                                             fuzzy_match_map=fuzzy_match_map)
                vector_dict[DiscriminativeMotifs.PATIENT_CLASS] = patient_class
                x_train_dataset = pd.concat([x_train_dataset, pd.DataFrame(vector_dict, index=[patient_uId])])

        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        clf.fit(x_train_dataset.drop(DiscriminativeMotifs.PATIENT_CLASS, axis=1),
                x_train_dataset[DiscriminativeMotifs.PATIENT_CLASS])

        # build validation dataset
        x_validation_dataset = None
        patient_class_dict = local_patient_class[DiscriminativeMotifs.PATIENT_CLASS].to_dict()
        try:
            patient_class = x_test.Patient_uId.astype('str').transform(lambda row: patient_class_dict[row]).iloc[0]
        except:
            raise Exception(f"cannot find {test_patient_uid} in dataset")
        vector_dict = defaultdict()
        x_test_color_index = x_test.set_index(['Patient_uId', 'colors_vec_hash'])
        x_test_color_index.sort_index(inplace=True)
        self._add_cell_type_composition_freq_feature(x_test_color_index, unique_motifs_colors, vector_dict, test_patient_uid)
        x_test_index = x_test.set_index(['Patient_uId', 'ID'])
        x_test_index.sort_index(inplace=True)
        self._add_motif_freq_feature(x_test_index,
                                     unique_motifs,
                                     vector_dict,
                                     test_patient_uid,
                                     fuzzy_match_map=fuzzy_match_map)
        vector_dict[DiscriminativeMotifs.PATIENT_CLASS] = patient_class
        x_validation_dataset = pd.concat([x_validation_dataset, pd.DataFrame(vector_dict, index=[test_patient_uid])])
        instance = x_validation_dataset.drop(DiscriminativeMotifs.PATIENT_CLASS, axis=1)

        y_pred_validation_dataset = clf.predict(instance)
        y_prob_validation_dataset = clf.predict_proba(instance)
        # model expandability
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(instance)

        i = 0
        for class_i in x_validation_dataset[DiscriminativeMotifs.PATIENT_CLASS]:
            pred_result = y_pred_validation_dataset[i]
            if (pred_result == class_i) and (class_i == feature_conf.labels[0]):
                TP = + 1
            elif (pred_result == class_i) and (class_i == feature_conf.labels[1]):
                TN = + 1
            elif pred_result == feature_conf.labels[1]:
                FN = + 1
            elif pred_result == feature_conf.labels[0]:
                FP = + 1
            i = + 1

        return (TP, TN, FN, FP,
                c_features,
                y_prob_validation_dataset, class_i, pred_result, clf.classes_,
                zip(unique_motifs_colors + unique_motifs),
                (shap_values, instance))

    @staticmethod
    def _add_cell_identity_feature(data: pd.DataFrame, common_cell_type: Dict[int, str]):
        from cism import helpers

        # create color vec and color vec hash columns
        common_cell_type_counter = dict(
            zip([str(x) for x in common_cell_type], [0 for _ in range(len(common_cell_type))]))

        def create_colors_vector(motif: nx.Graph):
            base_counter = Counter(common_cell_type_counter)
            base_counter.update(nx.get_node_attributes(motif, name='type').values())
            return np.array(list(base_counter.values()))

        data['colors_vec'] = data.motif.transform(lambda row: create_colors_vector(helpers.string_base64_pickle(row)))
        data['colors_vec_hash'] = data.colors_vec.transform(lambda row: hash(tuple(row)))
        data['colors_vec_hash'] = data.colors_vec_hash.astype('category')

    @staticmethod
    def _add_cell_type_composition_freq_feature(one_class_data: pd.DataFrame,
                                                unique_motifs_colors: list,
                                                feature_set: dict,
                                                patient_uId: str):
        for motif_color_id in unique_motifs_colors:
            if (patient_uId, motif_color_id) in one_class_data.index:
                # if there is more than one field of view then the total sub-graphs could be changed
                # here we calculate the probability of getting the cell identity composition
                total_count = one_class_data.loc[(patient_uId, motif_color_id)].Count.sum()
                total_sub_graphs = (one_class_data.loc[(patient_uId, motif_color_id)].drop_duplicates('FOV')
                                   .apply(lambda row: row['Count']/row['Freq'], axis=1).sum())
                feature_set[motif_color_id] = total_count/total_sub_graphs
            else:
                feature_set[motif_color_id] = 0

    @staticmethod
    def _return_cost(node1, node2):
        return 0 if node1['type'] == node2['type'] else 1

    @staticmethod
    def _get_top_similar(motif_id: int,
                         color_vec_hash: int,
                         motif_dataset: pd.DataFrame,
                         top_n: int):
        ged_compare = None

        # get the specific motif
        selected_motif = motif_dataset[motif_dataset.ID == motif_id].motif.iloc[0]

        for motif_id in motif_dataset[motif_dataset.colors_vec_hash == color_vec_hash].ID.unique():
            motif_compare = motif_dataset[motif_dataset.ID == motif_id].motif.iloc[0]

            ged_compare = pd.concat([ged_compare, pd.DataFrame([{'ID': motif_id,
                                                                 'GED': nx.graph_edit_distance(selected_motif,
                                                                                               motif_compare,
                                                                                               node_subst_cost=TissueStateDiscriminativeMotifs._return_cost)}])],
                                    ignore_index=True)
        return ged_compare.sort_values(by='GED')[:top_n]

    @staticmethod
    def _add_motif_freq_feature(one_class_data_index: pd.DataFrame,
                                unique_motifs: list,
                                feature_set: dict,
                                patient_uId: str,
                                fuzzy_match_map: dict):
        for motif_id in unique_motifs:
            if (len(fuzzy_match_map) > 0) and (patient_uId, one_class_data_index.index.get_level_values('ID').isin(fuzzy_match_map[motif_id])) in one_class_data_index.index:
                filter_data = one_class_data_index.loc[[(patient_uId, one_class_data_index.index.get_level_values('ID').isin(fuzzy_match_map[motif_id]))]]
            elif (len(fuzzy_match_map) == 0) and (patient_uId, motif_id) in one_class_data_index.index:
                filter_data = one_class_data_index.loc[[(patient_uId, motif_id)]]
            else:
                feature_set[motif_id] = 0
                continue

            total_count = filter_data.Count.sum()
            total_sub_graphs = (
                filter_data.drop_duplicates('FOV').apply(lambda row: row['Count'] / row['Freq'], axis=1).sum())
            feature_set[motif_id] = total_count / total_sub_graphs

