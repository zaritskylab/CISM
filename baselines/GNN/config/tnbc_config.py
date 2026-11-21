import torch

from config.project_paths import REPO_ROOT
from config.settings.data_path_settings import DataPathSettings
from config.settings.model_settings import ModelSettings
from config.settings.patient_sampling_settings import (
  MajorityUndersamplingSettings,
)
from config.settings.plots_settings import PlotsSettings
from config.settings.preprocess_settings import AugmentationSettings, PreprocessSettings
from config.settings.training_settings import TrainingSettings

from graph_sage_model import GraphSAGEModel


DATA_SETTINGS = DataPathSettings(
  dataset_base_path= REPO_ROOT / 'tnbc',
  bottom_up_csv_filename='tnbc_prettier_cell_type_names.csv',
)

# MIBI-TOF FOVs were captured at (800um)^2 / (2048px)^2, giving
# 0.390625 um/pixel (Keren et al. 2018, Cell -- "A Structured
# Tumor-Immune Microenvironment in Triple Negative Breast Cancer
# Revealed by Multiplexed Ion Beam Imaging").
PREPROCESS_SETTINGS = PreprocessSettings(
    groups=['Under', 'Over'],
    max_distance=100,
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
  patient_sampling=MajorityUndersamplingSettings(
    majority_n_patients=None,
    n_majority_draws=1,
  ),
)

PLOTS_SETTINGS = PlotsSettings(
  dataset_name='tnbc',
  groups=PREPROCESS_SETTINGS.groups,
  cv_full_type=TRAINING_SETTINGS.cv_full_type,
  random_seed=TRAINING_SETTINGS.random_seed,
)
