from collections.abc import Callable

import numpy as np
import torch


def build_pos_weighted_criterion(
  labels: np.ndarray,
  criterion_factory: Callable[..., torch.nn.Module],
  device: torch.device,
  context: str,
) -> torch.nn.Module:
  """Build a loss using only the labels in the current training split.

  Args:
    labels (np.ndarray): Binary labels from the samples used for fitting.
    criterion_factory (Callable): Loss constructor accepting pos_weight.
    device (torch.device): Device on which the loss will be evaluated.
    context (str): Human-readable split name for logging and errors.

  Returns:
    (torch.nn.Module): Loss configured from the training-only labels.

  Raises:
    ValueError: If labels are empty, non-binary, or contain one class.

  """
  labels = np.asarray(labels).reshape(-1)
  if labels.size == 0:
    raise ValueError(f"{context} has no training labels.")

  unique_labels = set(np.unique(labels).tolist())
  if not unique_labels.issubset({0, 1}):
    raise ValueError(
      f"{context} labels must be binary; received {unique_labels}."
    )

  negative_count = int((labels == 0).sum())
  positive_count = int((labels == 1).sum())
  if negative_count == 0 or positive_count == 0:
    raise ValueError(
      f"{context} must contain both classes to calculate pos_weight; "
      f"received n_neg={negative_count}, n_pos={positive_count}."
    )

  pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float,
    device=device,
  )
  print(
    f"{context} pos_weight: {pos_weight.item():.3f} "
    f"(n_neg={negative_count}, n_pos={positive_count})"
  )
  return criterion_factory(pos_weight=pos_weight)
