from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class AllPatientsSettings(BaseModel):
  """Keep every available training patient in each fold."""

  strategy: Literal['all_patients'] = 'all_patients'

  model_config = ConfigDict(extra='forbid')


class MajorityUndersamplingSettings(BaseModel):
  """Undersample majority-class patients within each training fold."""

  strategy: Literal['majority_undersampling'] = 'majority_undersampling'
  majority_n_patients: int | None = Field(default=None, ge=1)
  n_majority_draws: int = Field(default=1, ge=1)

  model_config = ConfigDict(extra='forbid')


PatientSamplingSettings = Annotated[
  AllPatientsSettings | MajorityUndersamplingSettings,
  Field(discriminator='strategy'),
]


__all__ = [
  'AllPatientsSettings',
  'MajorityUndersamplingSettings',
  'PatientSamplingSettings',
]
