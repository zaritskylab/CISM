from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Direction = Literal["all", "proliferation", "absence", "neutral"]
MatrixValue = Literal["observed_frequency", "signed_delta", "depletion"]


FANMOD_COLUMNS = [
    "ID",
    "Freq",
    "Count",
    "Mean_Freq",
    "STD",
    "z_score",
    "fanmod_empirical_p_value",
]


def normal_two_tailed_p_value(z_score: object) -> float:
    try:
        z = float(z_score)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(z):
        return np.nan
    return math.erfc(abs(z) / math.sqrt(2.0))


def direction_from_z_score(z_score: object) -> str:
    try:
        z = float(z_score)
    except (TypeError, ValueError):
        return "neutral"
    if not np.isfinite(z) or z == 0:
        return "neutral"
    if z > 0:
        return "proliferation"
    return "absence"


def add_normal_significance_columns(motifs: pd.DataFrame) -> pd.DataFrame:
    motifs = motifs.copy()
    motifs["p_value_normal_two_tailed"] = motifs["z_score"].map(normal_two_tailed_p_value)
    motifs["p_value_direction"] = motifs["z_score"].map(direction_from_z_score)
    motifs["frequency_delta"] = motifs["Freq"] - motifs["Mean_Freq"]
    motifs["depletion"] = (motifs["Mean_Freq"] - motifs["Freq"]).clip(lower=0.0)
    motifs["proliferation"] = motifs["frequency_delta"].clip(lower=0.0)
    return motifs


def parse_fanmod_output_csv(path: str | Path, dataset_name: str | None = None) -> pd.DataFrame:
    path = Path(path)
    rows: list[list[str]] = []
    header_seen = False

    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("ID,Frequency,Count,Mean-Freq,Standard-Dev,Z-Score,p-Value"):
                header_seen = True
                continue
            if not header_seen or not line:
                continue
            fields = line.split(",")
            if len(fields) < 7:
                continue
            if len(fields) >= 7 and _looks_like_fanmod_record(fields):
                rows.append(fields[:7])

    if not rows:
        return pd.DataFrame(columns=_artifact_columns())

    motifs = pd.DataFrame(rows, columns=FANMOD_COLUMNS)
    for column in FANMOD_COLUMNS:
        motifs[column] = pd.to_numeric(motifs[column], errors="coerce")

    patient, fov = _patient_fov_from_artifact_name(path)
    motifs["Patient"] = patient
    motifs["FOV"] = fov
    motifs["Patient_uId"] = f"{dataset_name}{patient}" if dataset_name else str(patient)
    motifs["source_file"] = str(path)
    motifs = motifs.dropna(subset=["ID", "Freq", "Count", "Mean_Freq", "STD", "z_score"])
    motifs["ID"] = motifs["ID"].astype("int64")
    return add_normal_significance_columns(motifs)


def load_fanmod_output_tree(
    fanmod_output_root: str | Path,
    dataset_name: str | None = None,
    motif_size: int | None = None,
) -> pd.DataFrame:
    root = Path(fanmod_output_root)
    if motif_size is not None:
        candidates = sorted(root.rglob(f"{motif_size}/*.csv"))
    else:
        candidates = sorted(root.rglob("*.csv"))

    frames = [
        parse_fanmod_output_csv(path, dataset_name=dataset_name)
        for path in candidates
        if path.is_file()
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_artifact_columns())
    return pd.concat(frames, ignore_index=True)


def filter_by_normal_significance(
    motifs: pd.DataFrame,
    alpha: float = 0.05,
    direction: Direction = "all",
) -> pd.DataFrame:
    if direction not in {"all", "proliferation", "absence", "neutral"}:
        raise ValueError(f"Unsupported direction: {direction}")
    filtered = motifs[motifs["p_value_normal_two_tailed"] < alpha].copy()
    if direction != "all":
        filtered = filtered[filtered["p_value_direction"] == direction].copy()
    return filtered


def build_patient_motif_matrix_from_fanmod(
    motifs: pd.DataFrame,
    patient_ids: list[object] | pd.Index | None = None,
    value: MatrixValue = "observed_frequency",
) -> pd.DataFrame:
    if value not in {"observed_frequency", "signed_delta", "depletion"}:
        raise ValueError(f"Unsupported matrix value: {value}")

    required = {"Patient_uId", "ID", "FOV", "Freq", "Count", "Mean_Freq"}
    missing = required.difference(motifs.columns)
    if missing:
        raise ValueError(f"motifs is missing required columns: {sorted(missing)}")

    if patient_ids is None:
        patient_ids = pd.Index(motifs["Patient_uId"].dropna().unique(), name="Patient_uId")
    else:
        patient_ids = pd.Index(patient_ids, name="Patient_uId")

    if motifs.empty:
        return pd.DataFrame(index=patient_ids)

    data = motifs[["Patient_uId", "ID", "FOV", "Freq", "Count", "Mean_Freq"]].copy()
    data = data[data["Freq"] > 0]
    data["_total_subgraphs"] = data["Count"] / data["Freq"]
    data["_weighted_expected"] = data["Mean_Freq"] * data["_total_subgraphs"]

    grouped = (
        data.groupby(["Patient_uId", "ID"], observed=True)
        .agg(
            total_count=("Count", "sum"),
            total_subgraphs=("_total_subgraphs", "sum"),
            total_expected_count=("_weighted_expected", "sum"),
        )
        .reset_index()
    )
    grouped["observed_frequency"] = grouped["total_count"] / grouped["total_subgraphs"]
    grouped["expected_frequency"] = grouped["total_expected_count"] / grouped["total_subgraphs"]
    grouped["signed_delta"] = grouped["observed_frequency"] - grouped["expected_frequency"]
    grouped["depletion"] = (grouped["expected_frequency"] - grouped["observed_frequency"]).clip(lower=0.0)

    matrix = grouped.pivot(index="Patient_uId", columns="ID", values=value)
    matrix = matrix.reindex(index=patient_ids).fillna(0.0)
    matrix.columns.name = None
    return matrix


def summarize_directional_motifs(motifs: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    significant = filter_by_normal_significance(motifs, alpha=alpha, direction="all")
    if significant.empty:
        return pd.DataFrame(
            columns=[
                "p_value_direction",
                "n_rows",
                "n_motifs",
                "n_patients",
                "median_abs_z_score",
                "median_frequency_delta",
            ]
        )
    return (
        significant.groupby("p_value_direction", observed=True)
        .agg(
            n_rows=("ID", "size"),
            n_motifs=("ID", "nunique"),
            n_patients=("Patient_uId", "nunique"),
            median_abs_z_score=("z_score", lambda values: float(np.nanmedian(np.abs(values)))),
            median_frequency_delta=("frequency_delta", "median"),
        )
        .reset_index()
        .sort_values("p_value_direction")
    )


def _looks_like_fanmod_record(fields: list[str]) -> bool:
    try:
        int(fields[0])
        float(fields[1])
        float(fields[2])
    except (TypeError, ValueError):
        return False
    return True


def _patient_fov_from_artifact_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "_" not in stem:
        return stem, stem
    patient, fov = stem.split("_", 1)
    return patient, fov


def _artifact_columns() -> list[str]:
    return [
        *FANMOD_COLUMNS,
        "Patient",
        "FOV",
        "Patient_uId",
        "source_file",
        "p_value_normal_two_tailed",
        "p_value_direction",
        "frequency_delta",
        "depletion",
        "proliferation",
    ]
