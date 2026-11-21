"""Shared metadata contract for cell-graph datasets."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from utils.patient_id_utils import normalize_patient_ids


DEFAULT_COLUMN_MAP: dict[str, str] = {
    "patient": "patient number",
    "fov": "fov",
    "label": "Group",
    "cell_type": "pred",
    "x": "centroid-0",
    "y": "centroid-1",
}

_REQUIRED_COLUMN_KEYS = tuple(DEFAULT_COLUMN_MAP)


def prepare_dataset_table(
    *,
    csv_path: str | Path | None,
    df: pd.DataFrame | None,
    groups: Sequence[Any],
    column_map: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load, validate, and index a dataset using canonical column roles.

    All graph encodings rely on the same patient/FOV/label metadata
    contract. This helper keeps alternative encodings from silently
    falling back to dataset-specific column names and guarantees that
    patient identifiers have one sortable string type before grouped CV.

    Returns:
      The filtered cell table, unique patient/FOV/label table, and
      resolved column mapping.

    Raises:
      ValueError: If no input source is supplied or a required mapped
        column is absent.

    """
    resolved_map = {**DEFAULT_COLUMN_MAP, **dict(column_map or {})}

    missing_keys = [
        key for key in _REQUIRED_COLUMN_KEYS
        if not resolved_map.get(key)
    ]
    if missing_keys:
        raise ValueError(
            "Column mapping is missing required role(s): "
            f"{missing_keys}"
        )

    patient_col = resolved_map["patient"]
    if df is not None:
        raw_df = df.copy()
    else:
        if csv_path is None:
            raise ValueError("csv_path is required when df is not provided")
        raw_df = pd.read_csv(
            csv_path,
            dtype={patient_col: "string"},
        )

    required_columns = {
        resolved_map[key] for key in _REQUIRED_COLUMN_KEYS
    }
    missing_columns = sorted(required_columns.difference(raw_df.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing mapped column(s): "
            f"{missing_columns}. Resolved mapping: {resolved_map}"
        )

    label_col = resolved_map["label"]
    if raw_df[label_col].isna().any():
        raise ValueError(
            f"Dataset label column {label_col!r} contains missing values."
        )
    dataset_df = raw_df[raw_df[label_col].isin(groups)].copy()
    dataset_df[patient_col] = normalize_patient_ids(
        dataset_df[patient_col],
        column_name=patient_col,
    )

    fov_col = resolved_map["fov"]
    fov_label_counts = dataset_df.groupby(
        [patient_col, fov_col], dropna=False
    )[label_col].nunique(dropna=False)
    conflicting_fovs = fov_label_counts[fov_label_counts != 1]
    if not conflicting_fovs.empty:
        raise ValueError(
            "Each patient/FOV must have exactly one outcome label. "
            f"Found {len(conflicting_fovs)} conflicting groups; examples: "
            f"{conflicting_fovs.index.tolist()[:10]}."
        )

    patient_label_counts = dataset_df.groupby(
        patient_col, dropna=False
    )[label_col].nunique(dropna=False)
    conflicting_patients = patient_label_counts[
        patient_label_counts != 1
    ]
    if not conflicting_patients.empty:
        raise ValueError(
            "Each patient must have exactly one outcome label across FOVs. "
            f"Found {len(conflicting_patients)} conflicting patients; "
            f"examples: {conflicting_patients.index.tolist()[:10]}."
        )

    patient_fov_combos = (
        dataset_df[[patient_col, fov_col, label_col]]
        .drop_duplicates([patient_col, fov_col])
        .sort_values([patient_col, fov_col])
        .reset_index(drop=True)
    )

    return dataset_df, patient_fov_combos, resolved_map
