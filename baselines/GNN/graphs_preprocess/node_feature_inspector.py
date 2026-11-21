import numpy as np


def format_vector(vec: np.ndarray) -> str:
  """
  Format a binary feature vector as a compact, readable string.

  Truncates long vectors to a head + tail preview to keep output
  concise when the feature dimension is large.

  Args:
    vec (np.ndarray): Binary float32 feature vector.

  Returns:
    (str): Human-readable vector string, e.g. "[0, 0, 1, ..., 0]".

  """
  if len(vec) <= 12:
    vals = ", ".join(str(int(v)) for v in vec)
    return f"[{vals}]"
  head = ", ".join(str(int(v)) for v in vec[:6])
  tail = ", ".join(str(int(v)) for v in vec[-3:])
  return f"[{head}, ..., {tail}]  (length {len(vec)})"
