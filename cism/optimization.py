from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cism.cism import HardDiscriminativeFC, SoftDiscriminativeFC


def _clone_feature_conf(feature_conf, **overrides):
    if isinstance(feature_conf, HardDiscriminativeFC):
        return HardDiscriminativeFC(
            labels=overrides.get("labels", feature_conf.labels),
            extract_by=overrides.get("extract_by", feature_conf.extract_by),
            use_cells_type_composition=overrides.get(
                "use_cells_type_composition", feature_conf.use_cells_type_composition
            ),
            use_motifs=overrides.get("use_motifs", feature_conf.use_motifs),
            shared_percentage=overrides.get("shared_percentage", feature_conf.shared_percentage),
            max_class_features=overrides.get("max_class_features", feature_conf.max_class_features),
            fuzzy_match=overrides.get("fuzzy_match", feature_conf.fuzzy_match),
            top_n_similar=overrides.get("top_n_similar", feature_conf.top_n_similar),
            fuzzy_match_exclude_original=overrides.get(
                "fuzzy_match_exclude_original", feature_conf.fuzzy_match_exclude_original
            ),
            cell_type_composition_patient_map=overrides.get(
                "cell_type_composition_patient_map", feature_conf.cell_type_composition_patient_map
            ),
            motifs_patient_map=overrides.get("motifs_patient_map", feature_conf.motifs_patient_map),
        )

    if isinstance(feature_conf, SoftDiscriminativeFC):
        return SoftDiscriminativeFC(
            labels=overrides.get("labels", feature_conf.labels),
            extract_by=overrides.get("extract_by", feature_conf.extract_by),
            use_cells_type_composition=overrides.get(
                "use_cells_type_composition", feature_conf.use_cells_type_composition
            ),
            use_motifs=overrides.get("use_motifs", feature_conf.use_motifs),
            shared_percentage=overrides.get("shared_percentage", feature_conf.shared_percentage),
            max_class_features=overrides.get("max_class_features", feature_conf.max_class_features),
            fuzzy_match=overrides.get("fuzzy_match", feature_conf.fuzzy_match),
            top_n_similar=overrides.get("top_n_similar", feature_conf.top_n_similar),
            fuzzy_match_exclude_original=overrides.get(
                "fuzzy_match_exclude_original", feature_conf.fuzzy_match_exclude_original
            ),
            cell_type_composition_patient_map=overrides.get(
                "cell_type_composition_patient_map", feature_conf.cell_type_composition_patient_map
            ),
            motifs_patient_map=overrides.get("motifs_patient_map", feature_conf.motifs_patient_map),
        )

    raise TypeError(
        "Optuna stringency tuning currently supports HardDiscriminativeFC and SoftDiscriminativeFC templates."
    )


@dataclass
class OptunaTuningResult:
    study: Any
    metric_name: str

    @property
    def best_params(self) -> dict:
        return dict(self.study.best_params)

    @property
    def best_score(self) -> float:
        return float(self.study.best_value)

    def trials_dataframe(self) -> pd.DataFrame:
        return self.study.trials_dataframe()
