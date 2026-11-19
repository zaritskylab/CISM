from pydantic import BaseModel


class ColumnMappingSettings(BaseModel):
  """
  Maps canonical column keys to a dataset's actual CSV headers.

  Defaults match the original hardcoded names used across the
  project's datasets. Datasets with different headers (e.g. gvhd's
  'patient_id'/'x'/'y') override only the fields that differ.

  """
  patient: str = 'patient number'
  fov: str = 'fov'
  label: str = 'Group'
  cell_type: str = 'pred'
  x: str = 'centroid-0'
  y: str = 'centroid-1'
  # Name of the column that uniquely identifies a cell/row in this
  # dataset's CSVs. Differs per dataset (e.g. 'uuid' for gliomas,
  # 'id' for melanoma's 26-cell version).
  cell_id: str = 'cell_id'
