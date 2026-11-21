from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase
from typing import Optional

import networkx as nx
import pandas as pd

from .. import helpers
from .motif_hits import motif_instances_per_fov_from_csv


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


def rank_discriminative_motifs(discriminator, motif_ids_by_class: dict, top_k: Optional[int] = None) -> pd.DataFrame:
    """Describe the discriminative motifs (class-exclusive AND above the stringency cut).

    `motif_ids_by_class` is `GetMotifsResult.get_motifs_by_class()`. Metadata comes from
    `cism.motifs_dataset` rather than `discover_result.get_discriminative_motifs()`, which
    is the class-exclusive tier and holds only the subset that stayed class-exclusive over
    the full cohort. Counts, FOVs and stringency are restricted to the motif's own class,
    so `significant_fovs` is the evidence the motif was selected on.
    """
    motif_table = discriminator.cism.motifs_dataset
    patient_classes = discriminator.get_patients_class(list(motif_ids_by_class))
    patient_classes = patient_classes.loc[~patient_classes.index.duplicated(keep="first")]
    class_sizes = patient_classes.groupby("patient_class", observed=True).size()

    records = []
    for patient_class, motif_ids in motif_ids_by_class.items():
        for motif_id in motif_ids:
            own_class_rows = motif_table[(motif_table.ID == motif_id) & (motif_table.patient_class == patient_class)]
            if own_class_rows.empty:
                continue
            stringency = own_class_rows.Patient_uId.nunique() / class_sizes[patient_class]
            records.append(
                {
                    "ID": int(motif_id),
                    "stringency": float(stringency),
                    "total_count": float(own_class_rows.Count.sum()),
                    "motif": own_class_rows.motif.iloc[0],
                    "patient_class": patient_class,
                    "n_patients": int(own_class_rows.Patient_uId.nunique()),
                    "significant_fovs": sorted(own_class_rows.FOV.astype(str).unique()),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["ID", "stringency", "total_count", "motif", "patient_class",
                     "n_patients", "significant_fovs", "score"]
        )

    ranked = pd.DataFrame.from_records(records)
    ranked["score"] = ranked["stringency"] * ranked["total_count"]
    ranked = ranked.sort_values(["score", "stringency", "total_count"], ascending=False).reset_index(drop=True)
    return ranked if top_k is None else ranked.head(top_k).copy()


def export_top_motif_visualization_inputs(
    *,
    discriminator,
    discover_result,
    raw_cells_csv_path: str,
    output_dir: str,
    top_k: Optional[int] = 5,
    motif_ids_by_class: Optional[dict] = None,
    max_distance: Optional[float] = None,
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # With motif_ids_by_class the export covers the discriminative tier (class-exclusive and
    # above the stringency cut); without it, the class-exclusive tier ranked by stringency x count.
    ranked_motifs = (
        rank_motifs_by_stringency_count(discover_result=discover_result, top_k=top_k)
        if motif_ids_by_class is None
        else rank_discriminative_motifs(
            discriminator=discriminator, motif_ids_by_class=motif_ids_by_class, top_k=top_k
        )
    )
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
            fov_whitelist=row.get("significant_fovs"),
            max_distance=max_distance,
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
                "n_patients": int(row["n_patients"]) if "n_patients" in row else None,
                "n_scanned_fovs": len(row["significant_fovs"]) if row.get("significant_fovs") is not None else None,
                "annotation_path": str(annotation_path),
                "hits_wide_path": str(hits_path),
                "n_hits": int(len(hits_wide)),
            }
        )

    return pd.DataFrame.from_records(export_records)
