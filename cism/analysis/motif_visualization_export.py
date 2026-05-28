from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Optional

import networkx as nx
import pandas as pd

from .. import helpers

try:
    from motif_hits_from_csv import motif_instances_per_fov_from_csv
except ImportError:  # pragma: no cover - notebook/runtime-dependent import path
    motif_instances_per_fov_from_csv = None


def _load_motif_graph(motif_value) -> nx.Graph:
    if isinstance(motif_value, nx.Graph):
        return motif_value
    if isinstance(motif_value, str):
        return helpers.string_base64_pickle(motif_value)
    raise TypeError(f"Unsupported motif value type: {type(motif_value)}")


def motif_to_annotation_text(motif: nx.Graph, common_cells_type: dict[int, str]) -> str:
    undirected_motif = nx.Graph(motif)
    ordered_nodes = sorted(undirected_motif.nodes())
    if len(ordered_nodes) > len(ascii_uppercase):
        raise ValueError("Motif has too many nodes to annotate with simple A/B/C labels.")

    node_to_alias = {node_id: ascii_uppercase[index] for index, node_id in enumerate(ordered_nodes)}
    lines = []

    for node_id in ordered_nodes:
        cell_type_id = int(undirected_motif.nodes[node_id]["type"])
        cell_type_name = common_cells_type[cell_type_id]
        lines.append(f"{node_to_alias[node_id]}.type = {cell_type_name}")

    for left_node, right_node in sorted(
        (tuple(sorted(edge, key=lambda node_id: node_to_alias[node_id])) for edge in undirected_motif.edges()),
        key=lambda edge: (node_to_alias[edge[0]], node_to_alias[edge[1]]),
    ):
        lines.append(f"{node_to_alias[left_node]} -> {node_to_alias[right_node]}")

    return "\n".join(lines) + "\n"


def rank_motifs_by_stringency_count(discover_result, top_k: int = 5) -> pd.DataFrame:
    motif_df = discover_result.get_discriminative_motifs().copy()
    if motif_df.empty:
        return pd.DataFrame(columns=["ID", "stringency", "total_count", "score", "motif", "patient_class"])

    ranked = (
        motif_df.groupby("ID", observed=True)
        .agg(
            stringency=("patient_percentage", "max"),
            total_count=("Count", "sum"),
            motif=("motif", "first"),
            patient_class=("patient_class", lambda values: values.mode().iloc[0]),
        )
        .reset_index()
    )
    ranked["score"] = ranked["stringency"] * ranked["total_count"]
    ranked = ranked.sort_values(["score", "stringency", "total_count"], ascending=False).reset_index(drop=True)
    return ranked.head(top_k).copy()


def export_top_motif_visualization_inputs(
    *,
    discriminator,
    discover_result,
    raw_cells_csv_path: str,
    output_dir: str,
    top_k: int = 5,
    fov_col: str = "fov",
    id_col: str = "cell_id",
    x_col: str = "centroid-0",
    y_col: str = "centroid-1",
    type_col: str = "pred",
    #class_col: str = "Group",
    classes: Optional[list[str]] = None,
    patient_col: Optional[str] = "patient number",
    induced: bool = False,
) -> pd.DataFrame:
    if motif_instances_per_fov_from_csv is None:
        raise ImportError(
            "motif_hits_from_csv.py could not be imported. "
            "Make sure it is available on the Python path before exporting motif visualization inputs."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ranked_motifs = rank_motifs_by_stringency_count(discover_result=discover_result, top_k=top_k)
    classes = classes or list(discriminator.get_patients_class().patient_class.unique())

    export_records = []
    for _, row in ranked_motifs.iterrows():
        motif_id = int(row["ID"])
        motif_graph = _load_motif_graph(row["motif"])
        motif_text = motif_to_annotation_text(motif_graph, discriminator.common_cells)

        annotation_path = output_path / f"{motif_id}.txt"
        annotation_path.write_text(motif_text, encoding="utf-8")

        _, hits_wide = motif_instances_per_fov_from_csv(
            csv_path=raw_cells_csv_path,
            motif_text=motif_text,
            fov_col=fov_col,
            id_col=id_col,
            x_col=x_col,
            y_col=y_col,
            type_col=type_col,
            #class_col=class_col,
            classes=classes,
            patient_col=patient_col,
            induced=induced,
        )

        hits_path = output_path / f"{motif_id}_hits_wide.csv"
        if hits_wide is None:
            hits_wide = pd.DataFrame(columns=["patient", "FOV"])
        hits_wide.to_csv(hits_path, index=False)

        export_records.append(
            {
                "motif_id": motif_id,
                "score": float(row["score"]),
                "stringency": float(row["stringency"]),
                "total_count": float(row["total_count"]),
                "patient_class": row["patient_class"],
                "annotation_path": str(annotation_path),
                "hits_wide_path": str(hits_path),
                "n_hits": int(len(hits_wide)),
            }
        )

    return pd.DataFrame.from_records(export_records)
