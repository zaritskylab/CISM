import pandas as pd


def normalize_patient_ids(
    patient_ids: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """Return patient identifiers as validated, trimmed Python strings.

    CSV chunk inference can produce a single object column containing both
    integers and strings.  NumPy/scikit-learn group splitters sort group
    values internally, and mixed Python types fail that sort.  Normalizing at
    ingestion also prevents the same textual identifier (for example ``123``)
    from becoming separate integer and string group keys.

    Args:
      patient_ids: Patient identifier column to normalize.
      column_name: Source column name, used in validation errors.

    Raises:
      ValueError: If any patient identifier is null or blank.

    """
    normalized = patient_ids.astype("string").str.strip()
    null_count = int(normalized.isna().sum())
    blank_count = int(normalized.eq("").fillna(False).sum())

    if null_count or blank_count:
        raise ValueError(
            f"Patient ID column '{column_name}' contains "
            f"{null_count} null and {blank_count} blank value(s)."
        )

    # Use ordinary Python strings rather than pandas' nullable scalar type so
    # downstream NumPy and scikit-learn code receives one sortable type.
    return normalized.astype(str)
