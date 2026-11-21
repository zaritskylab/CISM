from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx
import pandas as pd

from cism.data_preparation.common import (
    DatasetValidationError,
    PreparationResult,
    encode_cell_types,
    load_graph_object,
    rename_columns_copy,
    validate_centroid_dataframe,
    validate_edge_dataframe,
    validate_graph_dataframe,
)
from cism.graph.create_formatted_graph import GraphBuilder


def _ensure_output_dir(path_to_output_dir: str) -> str:
    output_dir = Path(path_to_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def prepare_from_centroids(
    dataframe: pd.DataFrame,
    path_to_output_dir: str,
    column_mapping: Mapping[str, str] | None = None,
    cell_type_mapper: dict | None = None,
    max_distance=None,
    exclude_cell_type: str | None = None,
    removed_cluster_buffer: int = 0,
    removed_cluster_alpha: float = 0.01,
) -> PreparationResult:
    canonical_df = rename_columns_copy(dataframe, column_mapping)
    validate_centroid_dataframe(canonical_df)

    builder = GraphBuilder(
        cells_csv=canonical_df,
        common_cells_mapper=cell_type_mapper,
        colnames_mapper_dict={},
    )
    return builder.build_graph(
        path_to_output_dir=_ensure_output_dir(path_to_output_dir),
        max_distance=max_distance,
        exclude_cell_type=exclude_cell_type,
        removed_cluster_buffer=removed_cluster_buffer,
        removed_cluster_alpha=removed_cluster_alpha,
    )


def prepare_from_edge_annotations(
    dataframe: pd.DataFrame,
    path_to_output_dir: str,
    column_mapping: Mapping[str, str] | None = None,
) -> PreparationResult:
    canonical_df = rename_columns_copy(dataframe, column_mapping)
    validate_edge_dataframe(canonical_df)
    output_dir = Path(_ensure_output_dir(path_to_output_dir))

    all_type_values = canonical_df["source_type"].astype(str).tolist() + canonical_df["target_type"].astype(str).tolist()
    source_type_map, _ = encode_cell_types(all_type_values)
    canonical_df = canonical_df.copy()
    canonical_df["source_type_id"] = canonical_df["source_type"].astype(str).map(source_type_map)
    canonical_df["target_type_id"] = canonical_df["target_type"].astype(str).map(source_type_map)

    if canonical_df["target_type_id"].isna().any():
        raise DatasetValidationError(
            "edge route found target_type values that were not present in source_type. "
            "Use consistent endpoint type labels across the dataset."
        )

    created_files = []
    for (patient_id, fov), group in canonical_df.groupby(["patient_id", "fov"], dropna=False, observed=True):
        file_path = output_dir / f"Patient_{patient_id}_FOV{fov}.txt"
        with file_path.open("w", encoding="utf-8") as handle:
            for _, row in group.iterrows():
                handle.write(
                    f"{row['source_id']} {row['target_id']} "
                    f"{int(row['source_type_id'])} {int(row['target_type_id'])}\n"
                )
        created_files.append(file_path.name)

    return PreparationResult(
        output_dir=str(output_dir),
        files=created_files,
        cell_type_to_id=source_type_map,
        route="edge_annotations",
    )


def prepare_from_graphs(
    graphs: Iterable | pd.DataFrame,
    path_to_output_dir: str,
    column_mapping: Mapping[str, str] | None = None,
    node_type_attribute: str = "cell_type",
) -> PreparationResult:
    output_dir = Path(_ensure_output_dir(path_to_output_dir))

    if isinstance(graphs, pd.DataFrame):
        graph_df = rename_columns_copy(graphs, column_mapping)
    else:
        graph_df = pd.DataFrame(list(graphs))
        graph_df = rename_columns_copy(graph_df, column_mapping)

    if "graph" not in graph_df.columns and "graph_path" in graph_df.columns:
        graph_df = graph_df.copy()
        graph_df["graph"] = graph_df["graph_path"].transform(load_graph_object)
    elif "graph" in graph_df.columns:
        graph_df = graph_df.copy()
        graph_df["graph"] = graph_df["graph"].transform(
            lambda graph_value: load_graph_object(graph_value) if not isinstance(graph_value, nx.Graph) else graph_value
        )

    validate_graph_dataframe(graph_df)

    observed_types = set()
    for graph in graph_df["graph"]:
        for _, attributes in graph.nodes(data=True):
            if node_type_attribute not in attributes:
                raise DatasetValidationError(
                    f"graph route requires node attribute '{node_type_attribute}' on every node."
                )
            observed_types.add(str(attributes[node_type_attribute]))

    cell_type_to_id = {label: idx for idx, label in enumerate(sorted(observed_types))}
    created_files = []

    for _, row in graph_df.iterrows():
        patient_id = row["patient_id"]
        fov = row["fov"]
        graph = row["graph"]
        file_path = output_dir / f"Patient_{patient_id}_FOV{fov}.txt"

        with file_path.open("w", encoding="utf-8") as handle:
            for left_node, right_node in graph.edges():
                left_type = str(graph.nodes[left_node][node_type_attribute])
                right_type = str(graph.nodes[right_node][node_type_attribute])
                handle.write(
                    f"{left_node} {right_node} {cell_type_to_id[left_type]} {cell_type_to_id[right_type]}\n"
                )

        created_files.append(file_path.name)

    return PreparationResult(
        output_dir=str(output_dir),
        files=created_files,
        cell_type_to_id=cell_type_to_id,
        route="graphs",
    )
