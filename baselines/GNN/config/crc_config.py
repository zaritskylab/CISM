import torch

from config.project_paths import REPO_ROOT
from config.settings.column_mapping_settings import ColumnMappingSettings
from config.settings.data_path_settings import DataPathSettings
from config.settings.model_settings import ModelSettings
from config.settings.plots_settings import PlotsSettings
from config.settings.preprocess_settings import (
  AugmentationSettings,
  PreprocessSettings,
)
from config.settings.training_settings import TrainingSettings
from graph_sage_model import GraphSAGEModel

DATA_SETTINGS = DataPathSettings(
  dataset_base_path=REPO_ROOT / 'crc',
  bottom_up_csv_filename='crc_prettier_cell_type_names.csv',
  column_mapping=ColumnMappingSettings(cell_id='cellID'),
)

# This dataset is CODEX (Schurch et al. 2020, Cell).
PREPROCESS_SETTINGS = PreprocessSettings(
    groups=['CLR', 'DII'],
    max_distance=125,
    augmentation=AugmentationSettings(
        sub_ratio=0.7, # 70% of FOV size
        n_subs_per_fov=5, # 5 sub-FOVs per FOV
    )
)

MODEL_SETTINGS = ModelSettings(
  model_class=GraphSAGEModel,
  hidden_layers=64,
  dropout_rate=0.1,
  loss_fn=torch.nn.BCEWithLogitsLoss,
)

TRAINING_SETTINGS = TrainingSettings()

PLOTS_SETTINGS = PlotsSettings(
  dataset_name='crc',
  groups=PREPROCESS_SETTINGS.groups,
  cv_full_type=TRAINING_SETTINGS.cv_full_type,
  random_seed=TRAINING_SETTINGS.random_seed,
)
