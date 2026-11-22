#!/usr/bin/env python3
"""Prepare the complete published CRC table for CISM using 14 groups plus Unidentified."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, distance

from cism import prepare_from_centroids, validate_network_dataset_directory
from cism.graph.plugin_clean_tumor_clusters import process_graph


SOURCE_TO_GROUP = {
    "B cells": "B cells",
    "CD11c+ DCs": "CD11c+ DCs",
    "CD4+ T cells": "CD4+ T cells",
    "CD4+ T cells CD45RO+": "CD4+ T cells",
    "CD4+ T cells GATA3+": "CD4+ T cells",
    "CD8+ T cells": "CD8+ T cells",
    "granulocytes": "Granulocytes",
    "CD11b+ monocytes": "Macrophages",
    "CD11b+CD68+ macrophages": "Macrophages",
    "CD163+ macrophages": "Macrophages",
    "CD68+ macrophages": "Macrophages",
    "CD68+ macrophages GzmB+": "Macrophages",
    "CD68+CD163+ macrophages": "Macrophages",
    "NK cells": "NK cells",
    "CD3+ T cells": "Other immune cells",
    "immune cells": "Other immune cells",
    "plasma cells": "Plasma cells",
    "smooth muscle": "Smooth muscle",
    "adipocytes": "Stroma",
    "nerves": "Stroma",
    "stroma": "Stroma",
    "Tregs": "Tregs",
    "tumor cells": "Tumor cells",
    "lymphatics": "Vasculature",
    "vasculature": "Vasculature",
    # Dirt is retained for triangulation and removed from the graph afterward,
    # matching the authors' contact-analysis sequence.
    "dirt": None,
    # These labels do not support a unique biological assignment.
    "undefined": "Unidentified",
    "immune cells / vasculature": "Unidentified",
    "tumor cells / immune cells": "Unidentified",
}

EXPECTED_GROUPS = {
    "B cells", "CD11c+ DCs", "CD4+ T cells", "CD8+ T cells",
    "Granulocytes", "Macrophages", "NK cells", "Other immune cells",
    "Plasma cells", "Smooth muscle", "Stroma", "Tregs", "Tumor cells",
    "Vasculature", "Unidentified",
}


def find_alpha_removed_cells(
    source: pd.DataFrame,
    *,
    max_distance: float | None,
    alpha: float,
    buffer: float,
) -> tuple[set[int], int]:
    """Reproduce GraphBuilder's dirt cleanup and return removed biological rows."""
    removed: set[int] = set()
    cluster_count = 0
    for (_, _), group in source.groupby(["patient_id", "fov"], sort=False, observed=True):
        row_ids = group.index.to_numpy()
        points = group[["centroid-0", "centroid-1"]].to_numpy()
        graph = nx.Graph()
        for row_id, point, cell_type in zip(row_ids, points, group["graph_cell_type"]):
            graph.add_node(int(row_id), pos=tuple(point), cell_type=str(cell_type))

        if len(points) >= 3:
            indptr, neighbours = Delaunay(points).vertex_neighbor_vertices
            pairs = ((left, right) for left in range(len(points))
                     for right in neighbours[indptr[left]:indptr[left + 1]])
        else:
            pairs = ((left, right) for left in range(len(points))
                     for right in range(len(points)) if left != right)
        for left, right in pairs:
            if max_distance is None or distance.euclidean(points[left], points[right]) <= max_distance:
                graph.add_edge(int(row_ids[left]), int(row_ids[right]))

        cleaned, clusters = process_graph(
            graph,
            "__DIRT__",
            buffer=buffer,
            alpha=alpha,
        )
        cluster_count += len(clusters)
        biological = set(group.index[group["cell_type"].notna()].astype(int))
        removed.update(biological - set(cleaned.nodes))
    return removed, cluster_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=125.0,
                        help="Maximum Delaunay-edge length in source pixels.")
    parser.add_argument("--dirt-alpha", type=float, default=0.01,
                        help="Alpha used to delineate connected dirt clusters.")
    parser.add_argument("--dirt-buffer", type=float, default=0.0,
                        help="Buffer in source pixels around each dirt alpha shape.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    usecols = ["CellID", "File Name", "groups", "patients", "X:X", "Y:Y", "ClusterName"]
    source = pd.read_csv(args.source, usecols=usecols)
    raw_source_rows = len(source)
    raw_dirt_cells = int((source["ClusterName"] == "dirt").sum())
    if source[usecols].isna().any().any():
        raise ValueError(f"Null values in required source columns: {source[usecols].isna().sum().to_dict()}")
    observed = set(source["ClusterName"].astype(str).unique())
    missing_mapping = sorted(observed - set(SOURCE_TO_GROUP))
    stale_mapping = sorted(set(SOURCE_TO_GROUP) - observed)
    if missing_mapping:
        raise ValueError(f"Unmapped source labels: {missing_mapping}")
    if {value for value in SOURCE_TO_GROUP.values() if value is not None} != EXPECTED_GROUPS:
        raise AssertionError("Mapping does not produce exactly 14 paper groups plus Unidentified")

    source["cell_type"] = source["ClusterName"].map(SOURCE_TO_GROUP)
    source["graph_cell_type"] = source["cell_type"].fillna("__DIRT__")
    source["patient_id"] = pd.to_numeric(source["patients"], errors="raise").astype(int)
    source["fov"] = source["File Name"].astype(str)
    source["cell_id"] = pd.to_numeric(source["CellID"], errors="raise").astype(int)
    source["centroid-0"] = pd.to_numeric(source["X:X"], errors="raise")
    source["centroid-1"] = pd.to_numeric(source["Y:Y"], errors="raise")

    if source["cell_id"].duplicated().any():
        raise ValueError("Published CellID is not globally unique")
    if source.groupby("fov")["patient_id"].nunique().max() != 1:
        raise ValueError("At least one FOV contains multiple patient identifiers")

    clinical = pd.read_csv(args.clinical, encoding="utf-8-sig")[["Patient", "Group"]].dropna()
    clinical["Patient"] = pd.to_numeric(clinical["Patient"], errors="raise").astype(int)
    clinical["Group"] = pd.to_numeric(clinical["Group"], errors="raise").astype(int)
    clinical = clinical[clinical.Patient.isin(sorted(source.patient_id.unique()))].drop_duplicates("Patient")
    if clinical.Patient.nunique() != source.patient_id.nunique():
        raise ValueError("Clinical table does not match the 35 source patients")
    source_group = source.groupby("patient_id")["groups"].nunique()
    if source_group.max() != 1:
        raise ValueError("A patient has inconsistent source group labels")
    source_group = source.groupby("patient_id")["groups"].first().astype(int)
    clinical_group = clinical.set_index("Patient")["Group"].sort_index()
    if not source_group.sort_index().equals(clinical_group):
        mismatches = pd.concat([source_group.rename("source"), clinical_group.rename("clinical")], axis=1)
        raise ValueError(f"Source and clinical group labels disagree:\n{mismatches[mismatches.source != mismatches.clinical]}")

    duplicate_xy = source.duplicated(["fov", "centroid-0", "centroid-1"], keep="first")
    duplicate_xy_count = int(duplicate_xy.sum())
    if duplicate_xy_count:
        source.loc[duplicate_xy, ["CellID", "File Name", "X:X", "Y:Y", "ClusterName"]].to_csv(
            args.output_dir / "excluded_duplicate_coordinates.csv", index=False
        )
        source = source.loc[~duplicate_xy].copy()

    mapping = pd.DataFrame(sorted(SOURCE_TO_GROUP.items()), columns=["source_cluster", "analysis_group"])
    mapping["analysis_group"] = mapping["analysis_group"].fillna("Dirt alpha-shape removal seed")
    mapping.to_csv(args.output_dir / "cell_type_mapping.csv", index=False)
    source.assign(mapping_result=source["cell_type"].fillna("Dirt alpha-shape removal seed")).groupby(
        ["ClusterName", "mapping_result"], observed=True
    ).size().rename("cells").reset_index().to_csv(
        args.output_dir / "source_to_group_counts.csv", index=False
    )
    alpha_removed_ids, dirt_cluster_count = find_alpha_removed_cells(
        source,
        max_distance=args.max_distance,
        alpha=args.dirt_alpha,
        buffer=args.dirt_buffer,
    )
    alpha_removed_mask = source.index.isin(alpha_removed_ids)
    alpha_removed = source.loc[alpha_removed_mask].copy()
    alpha_removed[[
        "patient_id", "fov", "cell_id", "centroid-0", "centroid-1", "ClusterName", "cell_type"
    ]].to_csv(args.output_dir / "alpha_shape_removed_cells.csv.gz", index=False, compression="gzip")
    retained = source[source["cell_type"].notna() & ~alpha_removed_mask].copy()
    retained.groupby("cell_type", observed=True).size().rename("cells").sort_values(ascending=False).to_csv(
        args.output_dir / "analysis_group_counts.csv"
    )
    region_source = source.groupby(["patient_id", "fov"], observed=True).size().rename("source_cells")
    region_retained = retained.groupby(["patient_id", "fov"], observed=True).size().rename("retained_cells")
    pd.concat([region_source, region_retained], axis=1).fillna(0).reset_index().to_csv(
        args.output_dir / "region_manifest.csv", index=False)

    canonical = retained[["patient_id", "fov", "cell_id", "centroid-0", "centroid-1", "cell_type"]].copy()
    canonical.to_csv(args.output_dir / "mapped_cells.csv.gz", index=False, compression="gzip")
    pd.DataFrame({
        "patient_uid": "CRC" + clinical.Patient.astype(str),
        "class": clinical.Group.map({1: "CLR", 2: "DII"}),
    }).to_csv(args.output_dir / "patient_class.csv", index=False, header=False)

    # Build the Delaunay graph with dirt present, then remove dirt plus biological
    # cells enclosed by the alpha shapes of connected dirt clusters.
    graph_input = source[["patient_id", "fov", "cell_id", "centroid-0", "centroid-1", "graph_cell_type"]].rename(
        columns={"graph_cell_type": "cell_type"}
    )
    graph_labels = sorted(EXPECTED_GROUPS | {"__DIRT__"})
    prep = prepare_from_centroids(
        dataframe=graph_input,
        path_to_output_dir=str(args.output_dir),
        cell_type_mapper={group: group for group in graph_labels},
        max_distance=args.max_distance,
        exclude_cell_type="__DIRT__",
        removed_cluster_alpha=args.dirt_alpha,
        removed_cluster_buffer=args.dirt_buffer,
    )
    dirt_id = prep.cell_type_to_id["__DIRT__"]
    source_id_to_label = {value: key for key, value in prep.cell_type_to_id.items()}
    final_label_to_id = {label: index for index, label in enumerate(sorted(EXPECTED_GROUPS))}
    unexpected_dirt_edges = 0
    for graph_path in sorted(args.output_dir.glob("Patient_*_FOV*.txt")):
        rewritten = []
        with graph_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) != 4:
                    continue
                left, right, left_type, right_type = fields
                left_type, right_type = int(left_type), int(right_type)
                if left_type == dirt_id or right_type == dirt_id:
                    unexpected_dirt_edges += 1
                    continue
                rewritten.append(
                    f"{left} {right} {final_label_to_id[source_id_to_label[left_type]]} "
                    f"{final_label_to_id[source_id_to_label[right_type]]}\n"
                )
        temporary = graph_path.with_suffix(".tmp")
        temporary.write_text("".join(rewritten), encoding="utf-8")
        temporary.replace(graph_path)
    if unexpected_dirt_edges:
        raise RuntimeError(
            f"Alpha-shape cleanup left {unexpected_dirt_edges} directed edges incident to dirt"
        )
    pd.DataFrame(
        sorted(final_label_to_id.items(), key=lambda item: item[1]),
        columns=["node_color", "color_id"],
    )[["color_id", "node_color"]].to_csv(args.output_dir / "node_color_map.csv", index=False)
    graph_files = validate_network_dataset_directory(args.output_dir, require_patient_class=True)
    if len(graph_files) != canonical.fov.nunique():
        raise ValueError(f"Expected {canonical.fov.nunique()} graph files, found {len(graph_files)}")

    summary = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "clinical_source": str(args.clinical),
        "clinical_sha256": sha256(args.clinical),
        "source_rows": raw_source_rows,
        "retained_cells": int(len(canonical)),
        "excluded_dirt_cells_raw": raw_dirt_cells,
        "excluded_dirt_cells_after_coordinate_deduplication": int((source["ClusterName"] == "dirt").sum()),
        "alpha_shape_removed_biological_cells": int(len(alpha_removed)),
        "connected_dirt_clusters": dirt_cluster_count,
        "dirt_alpha": args.dirt_alpha,
        "dirt_buffer_source_pixels": args.dirt_buffer,
        "dirt_removal_policy": "Delaunay triangulation includes dirt; dirt and biological cells inside buffered dirt alpha shapes are removed",
        "excluded_duplicate_coordinates": duplicate_xy_count,
        "patients": int(canonical.patient_id.nunique()),
        "fovs": int(canonical.fov.nunique()),
        "source_labels": int(len(observed)),
        "analysis_labels": int(canonical.cell_type.nunique()),
        "analysis_label_names": sorted(canonical.cell_type.unique()),
        "class_counts": clinical.Group.map({1: "CLR", 2: "DII"}).value_counts().to_dict(),
        "max_distance_source_pixels": args.max_distance,
        "max_distance_um_at_0.37744_um_per_pixel": args.max_distance * 0.37744,
        "stale_mapping_labels": stale_mapping,
        "graph_files": len(graph_files),
    }
    (args.output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
