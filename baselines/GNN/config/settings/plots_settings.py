from pydantic import BaseModel, computed_field

from config.project_paths import REPO_ROOT

RESULTS_ROOT = REPO_ROOT / 'baselines' / 'GNN' / 'runs'


class PlotsSettings(BaseModel):
  dataset_name: str
  groups: list[str]
  cv_full_type: str
  random_seed: int

  @computed_field
  @property
  def plots_prefix(self) -> str:
    return (
      f'{self.dataset_name}'
      f'_{self.groups[0].lower()}'
      f'_vs_{self.groups[1].lower()}'
    )

  @computed_field
  @property
  def full_plots_prefix(self) -> str:
    return (
      f'{self.plots_prefix}'
      f'_{self.cv_full_type}'
      f'_seed{self.random_seed}'
    )
  
  @computed_field
  @property
  def results_folder_path_prefix(self) -> str:
    base_path = RESULTS_ROOT / self.dataset_name
    return str(base_path)
