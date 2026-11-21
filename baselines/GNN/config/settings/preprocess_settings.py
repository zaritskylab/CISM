from pydantic import BaseModel, ConfigDict, Field


class AugmentationSettings(BaseModel):
  sub_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
  n_subs_per_fov: int = Field(default=5, ge=1)

  model_config = ConfigDict(extra='forbid')


class PreprocessSettings(BaseModel):
  # Graph-building edge cutoff, in coordinates induced from the csv
  # pay attention if the coordinates are in pixels or microns, as this will affect the cutoff
  max_distance: float = Field(default=100, gt=0.0)
  # clinical labels for down-stream classification tasks
  groups: list[str] = Field(
    default_factory=lambda: ['NN', 'NP'],
    min_length=2,
  )
  augmentation: AugmentationSettings = Field(
    default_factory=AugmentationSettings
  )

  model_config = ConfigDict(extra='forbid')
