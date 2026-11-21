from pydantic import BaseModel, ConfigDict, Field


class OptimizerSettings(BaseModel):
  """Optimizer hyperparameters shared by all training protocols."""

  learning_rate: float = Field(default=1e-3, gt=0)
  weight_decay: float = Field(default=1e-5, ge=0)

  model_config = ConfigDict(extra='forbid')


class EarlyStoppingSettings(BaseModel):
  """Validation-based stopping behavior."""

  enabled: bool = False
  patience: int = Field(default=15, ge=1)

  model_config = ConfigDict(extra='forbid')


class RegularizationSettings(BaseModel):
  """Loss and gradient regularization applied during fitting."""

  label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
  gradient_clip_norm: float | None = Field(default=None, gt=0)

  model_config = ConfigDict(extra='forbid')


class FitSettings(BaseModel):
  """Settings that control optimization of one fitted model."""

  epochs: int = Field(default=100, ge=1)
  optimizer: OptimizerSettings = Field(
    default_factory=OptimizerSettings,
  )
  early_stopping: EarlyStoppingSettings = Field(
    default_factory=EarlyStoppingSettings,
  )
  regularization: RegularizationSettings = Field(
    default_factory=RegularizationSettings,
  )

  model_config = ConfigDict(extra='forbid')
