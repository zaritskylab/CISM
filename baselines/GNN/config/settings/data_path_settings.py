from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from config.settings.column_mapping_settings import ColumnMappingSettings


class DataPathSettings(BaseModel):
  dataset_base_path: Path
  bottom_up_csv_filename: str
  dataset_cell_type_folder: str | None = None
  column_mapping: ColumnMappingSettings = Field(
    default_factory=ColumnMappingSettings
  )

  @computed_field
  @property
  def resolution_dir(self) -> Path:
    if self.dataset_cell_type_folder:
      return self.dataset_base_path / self.dataset_cell_type_folder.strip('/')
    return self.dataset_base_path

  @computed_field
  @property
  def bottom_up_data_full_dir(self) -> Path:
    return self.resolution_dir / self.bottom_up_csv_filename
