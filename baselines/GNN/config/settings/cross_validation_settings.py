from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from utils.cv_options import CVOptions, InnerCVOptions


class TrainTestSplitSettings(BaseModel):
  """Single grouped train/validation split inside an outer fold."""

  option: Literal[InnerCVOptions.TRAIN_TEST_SPLIT] = (
    InnerCVOptions.TRAIN_TEST_SPLIT
  )
  validation_patient_ratio: float = Field(
    default=0.2,
    gt=0.0,
    lt=1.0,
  )

  model_config = ConfigDict(extra='forbid')


class NestedInnerCVSettings(BaseModel):
  """Grouped inner folds used for selection before an outer refit."""

  option: Literal[InnerCVOptions.NESTEDCV] = InnerCVOptions.NESTEDCV
  num_folds: int = Field(default=5, ge=2)

  model_config = ConfigDict(extra='forbid')


InnerCVSettings = Annotated[
  TrainTestSplitSettings | NestedInnerCVSettings,
  Field(discriminator='option'),
]


class FoldCVSettings(BaseModel):
  """Stratified grouped outer folds with an inner selection policy."""

  cv_type: Literal[CVOptions.FOLDS] = CVOptions.FOLDS
  num_folds: int = Field(default=3, ge=2)
  inner: InnerCVSettings = Field(
    default_factory=TrainTestSplitSettings,
  )

  @computed_field
  @property
  def cv_full_type(self) -> str:
    if isinstance(self.inner, NestedInnerCVSettings):
      inner_name = f'Nested{self.inner.num_folds}f'
    else:
      validation_percent = round(
        self.inner.validation_patient_ratio * 100
      )
      training_percent = 100 - validation_percent
      inner_name = f'Split{training_percent}{validation_percent}'
    return f'{self.num_folds}Fold_{inner_name}'

  model_config = ConfigDict(extra='forbid')


class LopoCVSettings(BaseModel):
  """Leave-one-patient-out CV with optional grouped validation."""

  cv_type: Literal[CVOptions.LOPO] = CVOptions.LOPO
  validation_patient_ratio: float = Field(
    default=0.0,
    ge=0.0,
    lt=1.0,
  )

  @computed_field
  @property
  def cv_full_type(self) -> str:
    if self.validation_patient_ratio == 0:
      return 'LOPO_NoVal'

    validation_percent = round(self.validation_patient_ratio * 100)
    training_percent = 100 - validation_percent
    return f'LOPO_Split{training_percent}{validation_percent}'

  model_config = ConfigDict(extra='forbid')


CrossValidationSettings = Annotated[
  FoldCVSettings | LopoCVSettings,
  Field(discriminator='cv_type'),
]
