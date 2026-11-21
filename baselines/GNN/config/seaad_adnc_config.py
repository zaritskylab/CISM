import torch

from config.settings.column_mapping_settings import ColumnMappingSettings
from config.settings.cross_validation_settings import LopoCVSettings
from config.settings.data_path_settings import DataPathSettings
from config.settings.evaluation_settings import EvaluationSettings
from config.settings.fit_settings import (
    EarlyStoppingSettings,
    FitSettings,
    OptimizerSettings,
    RegularizationSettings,
)
from config.settings.model_settings import ModelSettings
from config.settings.plots_settings import PlotsSettings
from config.settings.preprocess_settings import (
    AugmentationSettings,
    PreprocessSettings,
)
from config.settings.training_settings import TrainingSettings
from config.project_paths import REPO_ROOT

from graph_sage_model import GraphSAGEModel
from utils.aggregation_options import PatientAggregation

# SEA-AD MTG MERFISH, cortical-column FOVs.
# The table is written by analysis/seaad/prepare_seaad_gnn_cells.py: the CISM cells table with
# Group = binary ADNC (Not AD + Low vs Intermediate + High, 13 vs 14 donors).

DATA_SETTINGS = DataPathSettings(
    dataset_base_path=REPO_ROOT / "analysis" / "seaad" / "adnc_neighborhoods",
    bottom_up_csv_filename="seaad_cells_column_cortex_neighborhood_gnn_adnc.csv",
    # meta_pred = the 9 neighborhood colors CISM used; pred holds the 137 supertypes
    column_mapping=ColumnMappingSettings(cell_id="uuid", cell_type="meta_pred"),
)

# Centroids are in µm (obsm['spatial']), so the cutoff is in µm too: 130 µm, the
# CISM run's graph cutoff (~99% of Delaunay edges kept, like TNBC's 100 px).
PREPROCESS_SETTINGS = PreprocessSettings(
    groups=["adnc_not_low", "adnc_int_high"],
    max_distance=130,
    augmentation=AugmentationSettings(
        sub_ratio=0.7,  # 70% of FOV size
        n_subs_per_fov=5,  # 5 sub-FOVs per FOV
    ),
)

# Model and training protocol
MODEL_SETTINGS = ModelSettings(
    model_class=GraphSAGEModel,
    hidden_layers=64,
    dropout_rate=0.1,
    loss_fn=torch.nn.BCEWithLogitsLoss,
)

TRAINING_SETTINGS = TrainingSettings(
    cross_validation=LopoCVSettings(
        validation_patient_ratio=0,
    ),
    fit=FitSettings(
        epochs=100,
        optimizer=OptimizerSettings(
            learning_rate=1e-3,
            weight_decay=1e-4,
        ),
        # no validation patients, so no early stopping: every fold trains all epochs
        early_stopping=EarlyStoppingSettings(
            enabled=False,
            patience=15,
        ),
        regularization=RegularizationSettings(
            label_smoothing=0.05,
            gradient_clip_norm=5.0,
        ),
    ),
    evaluation=EvaluationSettings(
        patient_aggregation=PatientAggregation.MAX,
    ),
    random_seed=42,
)

PLOTS_SETTINGS = PlotsSettings(
    dataset_name="seaad_adnc",
    groups=PREPROCESS_SETTINGS.groups,
    cv_full_type=TRAINING_SETTINGS.cv_full_type,
    random_seed=TRAINING_SETTINGS.random_seed,
)
