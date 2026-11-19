import torch

from config.project_paths import REPO_ROOT
from config.settings.column_mapping_settings import ColumnMappingSettings
from config.settings.cross_validation_settings import (
    LopoCVSettings,
)
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

from graph_sage_model import GraphSAGEModel
from utils.aggregation_options import PatientAggregation


DATA_SETTINGS = DataPathSettings(
    dataset_base_path=REPO_ROOT / "gliomas",
    bottom_up_csv_filename="gliomas_short_vs_long_transcriptomics_paper_aligned.csv",
    column_mapping=ColumnMappingSettings(cell_id="uuid"),
)

# MIBI-TOF FOVs were captured at (800um)^2 / (2048px)^2 or, for
# limited tissue, (400um)^2 / (1024px)^2 -- both give 0.390625
# um/pixel (Piyadasa & Oberlton et al., 2025, Methods).
PREPROCESS_SETTINGS = PreprocessSettings(
    groups=["short", "long"],
    max_distance=100,
    augmentation=AugmentationSettings(
        sub_ratio=0.7,  # 70% of FOV size
        n_subs_per_fov=5,  # 5 sub-FOVs per FOV
    ),
)

MODEL_SETTINGS = ModelSettings(
    model_class=GraphSAGEModel,
    hidden_layers=64,
    dropout_rate=0.1,
    loss_fn=torch.nn.BCEWithLogitsLoss,
)

TRAINING_SETTINGS = TrainingSettings(
    cross_validation=LopoCVSettings(
        validation_patient_ratio=0.2,
    ),
    fit=FitSettings(
        epochs=100,
        optimizer=OptimizerSettings(
            learning_rate=1e-3,
            weight_decay=1e-4,
        ),
        early_stopping=EarlyStoppingSettings(
            enabled=True,
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
    dataset_name="human_glioma",
    groups=PREPROCESS_SETTINGS.groups,
    cv_full_type=TRAINING_SETTINGS.cv_full_type,
    random_seed=TRAINING_SETTINGS.random_seed,
)
