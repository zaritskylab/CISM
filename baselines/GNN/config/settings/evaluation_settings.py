from pydantic import BaseModel, ConfigDict

from utils.aggregation_options import PatientAggregation


class EvaluationSettings(BaseModel):
  """Settings for converting FOV predictions into reported metrics."""

  patient_aggregation: PatientAggregation = PatientAggregation.MEAN

  model_config = ConfigDict(extra='forbid')
