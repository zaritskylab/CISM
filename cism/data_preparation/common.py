from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Iterable, Mapping

import networkx as nx
import pandas as pd


CANONICAL_COLUMNS = {
    "patient_id",
    "fov",
    "cell_id",
    "centroid-0",
    "centroid-1",
    "cell_type",
    "source_id",
    "target_id",
    "source_type",
    "target_type",
    "graph",
}

LEGACY_COLUMN_ALIASES = {
    "cell_types": "cell_type",
    "patient": "patient_id",
    "patient_number": "patient_id",
    "patient_num": "patient_id",
    "id": "cell_id",
    "type": "cell_type",
    "id1": "source_id",
    "id2": "target_id",
    "type1": "source_type",
    "type2": "target_type",
}

NETWORK_FILE_PATTERN = re.compile(r"^Patient_(?P<patient>.+)_FOV(?P<fov>.+)\.txt$")


class DatasetValidationError(ValueError):
    """Raised when an input dataset does not match the expected CISM contract."""


@dataclass
class PreparationResult:
    output_dir: str
    files: list[str]
    cell_type_to_id: dict
    route: str


def normalize_column_mapping(column_mapping: Mapping[str, str] | None) -> dict[str, str]:
    if column_mapping is None:
        return {}

    normalized_mapping = {}
    for requested_name, actual_name in column_mapping.items():
        canonical_name = LEGACY_COLUMN_ALIASES.get(requested_name, requested_name)
        normalized_mapping[canonical_name] = actual_name

    unknown_columns = set(normalized_mapping) - CANONICAL_COLUMNS
    if unknown_columns:
        raise DatasetValidationError(
            "Unknown canonical columns in column mapping: "
            f"{sorted(unknown_columns)}. Expected a subset of {sorted(CANONICAL_COLUMNS)}."
        )

    return normalized_mapping


def rename_columns_copy(
    dataframe: pd.DataFrame,
    column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    renamed = dataframe.copy()
    normalized_mapping = normalize_column_mapping(column_mapping)
    rename_dict = {}

    for canonical_name, actual_name in normalized_mapping.items():
        if actual_name in renamed.columns and actual_name != canonical_name:
            rename_dict[actual_name] = canonical_name

    return renamed.rename(columns=rename_dict, errors="ignore")


def assert_required_columns(dataframe: pd.DataFrame, required_columns: Iterable[str], route_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise DatasetValidationError(
            f"{route_name} input is missing required columns after normalization: {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}."
        )


def validate_centroid_dataframe(dataframe: pd.DataFrame, route_name: str = "centroid route") -> None:
    required_columns = ["patient_id", "fov", "centroid-0", "centroid-1", "cell_type"]
    assert_required_columns(dataframe, required_columns, route_name)

    numeric_columns = ["centroid-0", "centroid-1"]
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise DatasetValidationError(
                f"{route_name} expects numeric values in '{column}', got dtype {dataframe[column].dtype}."
            )

    if dataframe["patient_id"].isna().any() or dataframe["fov"].isna().any():
        raise DatasetValidationError(f"{route_name} requires non-null patient_id and fov values.")


def validate_edge_dataframe(dataframe: pd.DataFrame, route_name: str = "edge route") -> None:
    required_columns = ["patient_id", "fov", "source_id", "target_id", "source_type", "target_type"]
    assert_required_columns(dataframe, required_columns, route_name)

    if dataframe[required_columns].isna().any().any():
        raise DatasetValidationError(
            f"{route_name} requires non-null patient_id, fov, endpoints, and endpoint type columns."
        )


def validate_graph_dataframe(dataframe: pd.DataFrame, route_name: str = "graph route") -> None:
    required_columns = ["patient_id", "fov", "graph"]
    assert_required_columns(dataframe, required_columns, route_name)

    for _, row in dataframe.iterrows():
        graph = row["graph"]
        if not isinstance(graph, nx.Graph):
            raise DatasetValidationError(
                f"{route_name} requires each 'graph' value to be a networkx.Graph, got {type(graph)}."
            )


def encode_cell_types(values: Iterable) -> tuple[dict, pd.Series]:
    labels = pd.Series(list(values))
    unique_labels = labels.dropna().astype(str).unique().tolist()

    def is_int_like(value: str) -> bool:
        try:
            int(value)
        except ValueError:
            return False
        return True

    if unique_labels and all(is_int_like(label) for label in unique_labels):
        ordered_labels = sorted(unique_labels, key=lambda label: int(label))
    else:
        ordered_labels = sorted(unique_labels)

    cell_type_to_id = {label: idx for idx, label in enumerate(ordered_labels)}
    encoded = labels.astype(str).map(cell_type_to_id)
    return cell_type_to_id, encoded


def parse_network_filename(filename: str) -> tuple[str, str]:
    match = NETWORK_FILE_PATTERN.match(Path(filename).name)
    if match is None:
        raise DatasetValidationError(
            "Dataset filenames must match 'Patient_<patient_id>_FOV<fov>.txt' so CISM can discover "
            f"patient/FOV metadata. Got: {filename}."
        )

    return match.group("patient"), match.group("fov")


def validate_network_dataset_directory(dataset_dir: str | Path, require_patient_class: bool = False) -> list[str]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise DatasetValidationError(f"Dataset directory does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise DatasetValidationError(f"Dataset path must be a directory: {dataset_path}")

    text_files = sorted(path for path in dataset_path.iterdir() if path.suffix == ".txt")
    if not text_files:
        raise DatasetValidationError(
            "CISM add_dataset expects at least one FANMOD-compatible '.txt' network file in the dataset directory."
        )

    for text_file in text_files:
        parse_network_filename(text_file.name)
        with text_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 4:
                    raise DatasetValidationError(
                        f"Malformed network edge row in {text_file.name}:{line_number}. "
                        "Expected '<src_id> <dst_id> <src_type_id> <dst_type_id>'."
                    )
                try:
                    int(parts[2])
                    int(parts[3])
                except ValueError as exc:
                    raise DatasetValidationError(
                        f"Malformed node type encoding in {text_file.name}:{line_number}. "
                        "CISM expects integer node type ids in columns 3 and 4."
                    ) from exc

    patient_class_file = dataset_path / "patient_class.csv"
    if require_patient_class and not patient_class_file.exists():
        raise DatasetValidationError(
            "patient_class.csv is required for this workflow but was not found in the dataset directory."
        )

    return [path.name for path in text_files]


def load_graph_object(graph_input) -> nx.Graph:
    if isinstance(graph_input, nx.Graph):
        return graph_input

    path = Path(graph_input)
    suffix = path.suffix.lower()

    if suffix in {".pickle", ".pkl", ".gpickle"}:
        with path.open("rb") as handle:
            graph = pickle.load(handle)
    elif suffix == ".gml":
        graph = nx.read_gml(path)
    elif suffix == ".graphml":
        graph = nx.read_graphml(path)
    else:
        raise DatasetValidationError(
            f"Unsupported graph file format '{suffix}' for graph route. "
            "Supported formats: .pickle, .pkl, .gpickle, .gml, .graphml."
        )

    if not isinstance(graph, nx.Graph):
        raise DatasetValidationError(f"Loaded object from {path} is not a networkx.Graph.")

    return graph
