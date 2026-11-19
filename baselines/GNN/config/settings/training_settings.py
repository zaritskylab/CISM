from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from config.settings.cross_validation_settings import (
  CrossValidationSettings,
  FoldCVSettings,
  LopoCVSettings,
  NestedInnerCVSettings,
  TrainTestSplitSettings,
)
from config.settings.evaluation_settings import EvaluationSettings
from config.settings.fit_settings import (
  EarlyStoppingSettings,
  FitSettings,
  OptimizerSettings,
  RegularizationSettings,
)
from config.settings.patient_sampling_settings import (
  AllPatientsSettings,
  MajorityUndersamplingSettings,
  PatientSamplingSettings,
)


class TrainingSettings(BaseModel):
  """Top-level composition of protocol, fit, and evaluation settings."""

  cross_validation: CrossValidationSettings = Field(
    default_factory=LopoCVSettings,
  )
  fit: FitSettings = Field(default_factory=FitSettings)
  evaluation: EvaluationSettings = Field(
    default_factory=EvaluationSettings,
  )
  patient_sampling: PatientSamplingSettings = Field(
    default_factory=AllPatientsSettings,
  )
  random_seed: int = 42

  @model_validator(mode='after')
  def validate_protocol_compatibility(self):
    is_lopo = isinstance(self.cross_validation, LopoCVSettings)
    if isinstance(
      self.patient_sampling,
      MajorityUndersamplingSettings,
    ) and not is_lopo:
      raise ValueError(
        'Majority patient undersampling currently requires LOPO CV.'
      )

    if is_lopo:
      has_validation_split = (
        self.cross_validation.validation_patient_ratio > 0
      )
      if self.fit.early_stopping.enabled != has_validation_split:
        raise ValueError(
          'LOPO validation_patient_ratio must be non-zero exactly '
          'when early stopping is enabled.'
        )
    return self

  @computed_field
  @property
  def cv_full_type(self) -> str:
    return self.cross_validation.cv_full_type

  model_config = ConfigDict(extra='forbid')


__all__ = [
  'AllPatientsSettings',
  'CrossValidationSettings',
  'EarlyStoppingSettings',
  'EvaluationSettings',
  'FitSettings',
  'FoldCVSettings',
  'LopoCVSettings',
  'MajorityUndersamplingSettings',
  'NestedInnerCVSettings',
  'OptimizerSettings',
  'PatientSamplingSettings',
  'RegularizationSettings',
  'TrainingSettings',
  'TrainTestSplitSettings',
]
