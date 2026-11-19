import warnings

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_roc_auc(y_true, y_score, *, context="evaluation"):
    """Return NaN, with an explicit warning, when ROC AUC is undefined."""
    unique_classes = np.unique(np.asarray(y_true))
    if unique_classes.size < 2:
        warnings.warn(
            f"{context}: ROC AUC is unavailable because the evaluation "
            f"set contains only class {unique_classes.tolist()}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
