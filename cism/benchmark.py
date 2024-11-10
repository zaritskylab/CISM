import pandas as pd
import numpy as np
from cism.cism import DiscriminativeFeatureKey, HardDiscriminativeFC
from cism.cism import TissueStateDiscriminativeMotifs


class DiscriminatorBenchmark:
    def __init__(self, **kwargs):
        self.extract_by = kwargs.setdefault('extract_by', DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES)
        self.shared_percentage = kwargs.setdefault('shared_percentage', None)
        self.trials = kwargs.setdefault('trials', 0)

        self._extract_by_options = self.extract_by
        if not isinstance(self._extract_by_options, list):
            self._extract_by_options = [self._extract_by_options]

        if self.shared_percentage is None:
            raise Exception("shared_percentage must be provided")
        self._shared_percentage_options = self.shared_percentage
        if not isinstance(self._shared_percentage_options, list):
            self._shared_percentage_options = [self._shared_percentage_options]

        self._trials = self.trials
        if not isinstance(self._trials, list):
            self._trials = [self._trials]

    def run(self,
            discriminator: TissueStateDiscriminativeMotifs,
            labels: list,
            max_class_features: int = 10,
            fuzzy_match: bool = False,
            top_n_similar: int = 10,
            rand_patient_class: bool = False,
            rand_motifs: bool = False,
            benchmark_results: pd.DataFrame = None,
            debug_print: bool = True,
            n_jobs: int=1) -> pd.DataFrame:

        for extract_by in self._extract_by_options:
            for shared_percentage in self._shared_percentage_options:
                for trial in self._trials:
                    use_motif = True if extract_by is DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES else False
                    random_state = np.random.RandomState(trial)
                    feature_conf = HardDiscriminativeFC(extract_by=extract_by,
                                                        use_cells_type_composition=not use_motif,
                                                        use_motifs=use_motif,
                                                        shared_percentage=shared_percentage,
                                                        max_class_features=max_class_features,
                                                        fuzzy_match=fuzzy_match,
                                                        top_n_similar=top_n_similar,
                                                        labels=labels)

                    try:
                        if debug_print:
                            print('\n')
                            print(f'extract_by: {extract_by}, '
                                  f'use_motif:{use_motif}, '
                                  f'shared_percentage: {shared_percentage},'
                                  f'rand_patient_class: {rand_patient_class},'
                                  f'rand_motifs: {rand_motifs}')

                        analyze_motifs_result = discriminator.analyze_motifs(feature_conf=feature_conf,
                          exclude_patients=[],
                          n_jobs=n_jobs,
                          random_state=random_state,
                          rand_patient_class=rand_patient_class,
                          rand_motifs=rand_motifs)

                        if debug_print:
                            print(analyze_motifs_result.get_roc_auc_score())
                            print(analyze_motifs_result.results['cFeatures'].std())
                            print(analyze_motifs_result.results['cFeatures'].mean())

                        benchmark_results = pd.concat(
                            [benchmark_results,
                             pd.DataFrame([{'extract_by': extract_by,
                                            'use_motif': use_motif,
                                            'shared_percentage': shared_percentage,
                                            'cFeatures_sum': analyze_motifs_result.results[
                                              'cFeatures'].sum(),
                                            'cFeatures_std': analyze_motifs_result.results[
                                              'cFeatures'].std(),
                                            'cFeatures_mean': analyze_motifs_result.results[
                                              'cFeatures'].mean(),
                                            'random_state': trial,
                                            'rand_patient_class': rand_patient_class,
                                            'rand_motifs': rand_motifs,
                                            'roc_auc_score': analyze_motifs_result.get_roc_auc_score(),
                                            'full_results_debug': analyze_motifs_result.results,
                                            'error': False
                                            }])], ignore_index=True)
                    except:
                        if debug_print:
                            print('error probably due to zero discriminative features')
                        benchmark_results = pd.concat(
                            [benchmark_results,
                             pd.DataFrame([{'extract_by': extract_by,
                                            'use_motif': use_motif,
                                            'shared_percentage': shared_percentage,
                                            'cFeatures_sum': 0,
                                            'cFeatures_std': 0,
                                            'cFeatures_mean': 0,
                                            'random_state': trial,
                                            'rand_patient_class': rand_patient_class,
                                            'rand_motifs': rand_motifs,
                                            'roc_auc_score': 0,
                                            'full_results_debug': None,
                                            'error': True
                                            }])], ignore_index=True)
        return benchmark_results
