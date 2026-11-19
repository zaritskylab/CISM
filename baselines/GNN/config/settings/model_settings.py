from collections.abc import Callable

import torch
from pydantic import BaseModel, ConfigDict, Field

from graph_sage_model import GraphSAGEModel


class ModelSettings(BaseModel):
  """Model and loss constructors shared by all training protocols."""

  model_class: type[torch.nn.Module] = GraphSAGEModel
  hidden_layers: int = Field(default=64, ge=1)
  dropout_rate: float = Field(default=0.1, ge=0.0, lt=1.0)
  loss_fn: Callable[..., torch.nn.Module] = torch.nn.BCEWithLogitsLoss

  model_config = ConfigDict(
    arbitrary_types_allowed=True,
    extra='forbid',
  )
