#!/usr/bin/env python3
"""Prepare cSCC CosMx metadata for a CISM run.

This script expects cell-level metadata exported from the Dryad Seurat object
with ``analysis/cosmx_preparation/cscc_cosmx_export_metadata.R``. It writes FANMOD/CISM-compatible
colored edge-list files and preview summaries, but it does not run FANMOD.
"""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_COLUMNS = {
    "patient": "Subject_STUDY_ID",
    "response": "response",
    "pathologic_response": "Pathologic_response",
    "timepoint": "Biopsy_Timepoint",
    "x": "x_slide_mm",
    "y": "y_slide_mm",
    "cell_type": "CoarseAnnotationNew",
    "spatial_niche": "SpatialClusterNames",
    "cell_id": "cell_id",
}

COLUMN_ALIASES = {
    "pathologic_response": ["Pathologic_response", "Pathologic response"],
    "x": ["x_slide_mm", "x", "CenterX_global_px", "CenterX_local_px"],
    "y": ["y_slide_mm", "y", "CenterY_global_px", "CenterY_local_px"],
    "cell_type": ["CoarseAnnotationNew", "CoarseAnnotation", "NewAnnotation"],
    "spatial_niche": ["SpatialClusterNames", "SpatialCluster"],
}

RESPONSE_MAP = {
    "Responder": "Responder",
    "NonResponder": "NonResponder",
    "Non-mPR/pCR": "NonResponder",
    "mPR": "Responder",
    "pCR": "Responder",
}


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved = dict(DEFAULT_COLUMNS)
    for key, candidates in COLUMN_ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                resolved[key] = candidate
                break
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/cSCC_CosMx/source_metadata/cscc_cosmx_metadata.csv"),
        help="Cell metadata CSV exported from the Seurat object.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cSCC_CosMx"),
        help="Output directory for CISM-ready graph files.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("tutorial_runtime/cscc_cosmx_preparation"),
        help="Output directory for preview summaries.",
    )
    parser.add_argument(
        "--dataset-name",
        default="cSCC",
        help="CISM dataset_name prefix used in patient_class.csv keys.",
    )
    parser.add_argument(
        "--timepoint",
        default="Baseline",
        help="Timepoint to retain. Use 'all' to retain all timepoints.",
    )
    parser.add_argument(
        "--color-mode",
        choices=["cell_type", "spatial_niche", "cell_type_spatial_niche"],
        default="cell_type_spatial_niche",
        help="Node coloring scheme for CISM.",
    )
    parser.add_argument(
        "--max-cells-per-tile",
        type=int,
        default=2500,
        help="Target upper bound for cells per spatial tile before graph construction.",
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=0.04,
        help="Neighbor radius in slide millimeters. 0.04 mm = 40 microns.",
    )
    parser.add_argument(
        "--min-cells-per-tile",
        type=int,
        default=25,
        help="Skip very small tiles below this cell count.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write summaries and patient_class.csv, but do not write graph txt files.",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help=(
            "Allow preparation when patient/response/timepoint columns are absent. "
            "Uses slide+FOV as the sample key and writes patient_class_template.csv."
        ),
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: dict[str, str]) -> None:
    missing = [column for column in columns.values() if column not in df.columns]
    if missing:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )


def clean_response(df: pd.DataFrame, response_column: str, pathologic_column: str) -> pd.Series:
    if response_column in df.columns:
        response = df[response_column].astype(str).map(RESPONSE_MAP)
    else:
        response = pd.Series(index=df.index, dtype=object)

    if pathologic_column in df.columns:
        fallback = df[pathologic_column].astype(str).map(RESPONSE_MAP)
        response = response.fillna(fallback)

    return response


def make_color(df: pd.DataFrame, mode: str, cell_type_col: str, niche_col: str) -> pd.Series:
    cell_type = df[cell_type_col].astype(str).str.strip()
    niche = df[niche_col].astype(str).str.strip()
    if mode == "cell_type":
        return cell_type
    if mode == "spatial_niche":
        return niche
    return cell_type + "__" + niche


def assign_tiles(group: pd.DataFrame, max_cells_per_tile: int) -> pd.Series:
    n_cells = len(group)
    if n_cells <= max_cells_per_tile:
        return pd.Series(["tile_000_000"] * n_cells, index=group.index)

    n_tiles = math.ceil(n_cells / max_cells_per_tile)
    n_x = max(1, math.ceil(math.sqrt(n_tiles)))
    n_y = max(1, math.ceil(n_tiles / n_x))

    x = group["centroid-0"].to_numpy()
    y = group["centroid-1"].to_numpy()
    x_edges = np.linspace(x.min(), x.max(), n_x + 1)
    y_edges = np.linspace(y.min(), y.max(), n_y + 1)
    x_bin = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, n_x - 1)
    y_bin = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, n_y - 1)
    return pd.Series([f"tile_{xb:03d}_{yb:03d}" for xb, yb in zip(x_bin, y_bin)], index=group.index)


def write_radius_graph(group: pd.DataFrame, output_file: Path, radius_mm: float) -> int:
    coords = group[["centroid-0", "centroid-1"]].to_numpy(dtype=float)
    if len(coords) < 2:
        output_file.write_text("", encoding="utf-8")
        return 0

    tree = cKDTree(coords)
    pairs = sorted(tree.query_pairs(r=radius_mm))
    if not pairs and len(coords) > 1:
        _, nearest = tree.query(coords, k=2)
        pairs = sorted({tuple(sorted((i, int(row[1])))) for i, row in enumerate(nearest) if i != int(row[1])})

    node_ids = group["local_node_id"].to_numpy()
    colors = group["color_id"].to_numpy()
    with output_file.open("w", encoding="utf-8") as handle:
        for left, right in pairs:
            handle.write(
                f"{node_ids[left]} {node_ids[right]} "
                f"{int(colors[left])} {int(colors[right])}\n"
            )
    return len(pairs)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.metadata_csv)
    columns = resolve_columns(df)
    required_keys = {"x", "y", "cell_type", "spatial_niche"}
    if not args.allow_unlabeled:
        required_keys.update({"patient", "response", "timepoint"})
    require_columns(df, {key: columns[key] for key in sorted(required_keys)})

    df = df.copy()
    if columns["patient"] in df.columns:
        df["patient_id"] = df[columns["patient"]].astype(str)
    else:
        if not {"slide_ID_numeric", "fov"}.issubset(df.columns):
            raise ValueError("Unlabeled preparation requires slide_ID_numeric and fov columns.")
        df["patient_id"] = (
            "slide"
            + df["slide_ID_numeric"].astype(str)
            + "-fov"
            + df["fov"].astype(str)
        )

    if columns["response"] in df.columns or columns["pathologic_response"] in df.columns:
        df["response_clean"] = clean_response(df, columns["response"], columns["pathologic_response"])
    else:
        df["response_clean"] = pd.NA
    df["centroid-0"] = pd.to_numeric(df[columns["x"]], errors="coerce")
    df["centroid-1"] = pd.to_numeric(df[columns["y"]], errors="coerce")

    if columns["timepoint"] in df.columns and args.timepoint.lower() != "all":
        df = df[df[columns["timepoint"]].astype(str) == args.timepoint].copy()

    if not args.allow_unlabeled:
        df = df[df["response_clean"].isin(["Responder", "NonResponder"])].copy()
    df = df[df["centroid-0"].notna() & df["centroid-1"].notna()].copy()
    df = df[df[columns["cell_type"]].notna() & df[columns["spatial_niche"]].notna()].copy()
    df = df.reset_index(drop=True)

    df["cell_type"] = df[columns["cell_type"]].astype(str)
    df["spatial_niche"] = df[columns["spatial_niche"]].astype(str)
    df["node_color"] = make_color(df, args.color_mode, columns["cell_type"], columns["spatial_niche"])
    color_map = {label: idx for idx, label in enumerate(sorted(df["node_color"].unique()))}
    df["color_id"] = df["node_color"].map(color_map).astype(int)

    tile_parts = []
    for _, group in df.groupby("patient_id", sort=False):
        tile_parts.append(assign_tiles(group, args.max_cells_per_tile))
    df["tile_id"] = pd.concat(tile_parts).sort_index()
    df["fov"] = df["tile_id"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    patient_class = (
        df[["patient_id", "response_clean"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .assign(cism_patient=lambda x: args.dataset_name + x["patient_id"])
    )
    labeled_patient_class = patient_class.dropna(subset=["response_clean"])
    duplicated = labeled_patient_class["patient_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError("A patient has conflicting response labels after filtering.")

    if labeled_patient_class.empty:
        patient_class[["cism_patient", "response_clean"]].to_csv(
            args.output_dir / "patient_class_template.csv",
            header=False,
            index=False,
        )
    else:
        labeled_patient_class[["cism_patient", "response_clean"]].to_csv(
            args.output_dir / "patient_class.csv",
            header=False,
            index=False,
        )

    graph_rows = []
    if not args.preview_only:
        for (patient_id, fov), group in df.groupby(["patient_id", "fov"], sort=True):
            if len(group) < args.min_cells_per_tile:
                continue
            group = group.reset_index(drop=True).copy()
            group["local_node_id"] = np.arange(len(group), dtype=int)
            output_file = args.output_dir / f"Patient_{patient_id}_FOV{fov}.txt"
            edge_count = write_radius_graph(group, output_file, args.radius_mm)
            graph_rows.append(
                {
                    "patient_id": patient_id,
                    "fov": fov,
                    "cells": len(group),
                    "edges": edge_count,
                    "file": output_file.name,
                }
            )
    else:
        for (patient_id, fov), group in df.groupby(["patient_id", "fov"], sort=True):
            if len(group) < args.min_cells_per_tile:
                continue
            graph_rows.append(
                {
                    "patient_id": patient_id,
                    "fov": fov,
                    "cells": len(group),
                    "edges": None,
                    "file": f"Patient_{patient_id}_FOV{fov}.txt",
                }
            )

    graph_summary = pd.DataFrame(graph_rows)
    response_patient_counts = labeled_patient_class["response_clean"].value_counts().to_dict()
    response_cell_counts = df["response_clean"].dropna().value_counts().to_dict()

    cell_summary = {
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "timepoint": args.timepoint,
        "color_mode": args.color_mode,
        "radius_mm": args.radius_mm,
        "max_cells_per_tile": args.max_cells_per_tile,
        "cells": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "graphs": int(len(graph_summary)),
        "node_colors": int(len(color_map)),
        "response_patient_counts": response_patient_counts,
        "response_cell_counts": response_cell_counts,
        "clinical_labels_present": bool(not labeled_patient_class.empty),
    }

    preview_columns = [
        "patient_id",
        "response_clean",
        "fov",
        "cell_type",
        "spatial_niche",
        "node_color",
        "centroid-0",
        "centroid-1",
    ]
    if columns["timepoint"] in df.columns:
        preview_columns.insert(2, columns["timepoint"])
    df[preview_columns].head(5000).to_csv(args.summary_dir / "metadata_preview_head.csv", index=False)
    graph_summary.to_csv(args.summary_dir / "graph_summary.csv", index=False)
    pd.Series(color_map, name="color_id").rename_axis("node_color").to_csv(
        args.summary_dir / "node_color_map.csv"
    )
    with (args.summary_dir / "preparation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(cell_summary, handle, indent=2, sort_keys=True)

    print(json.dumps(cell_summary, indent=2, sort_keys=True))
    if labeled_patient_class.empty:
        print(f"Wrote patient_class_template.csv to {args.output_dir / 'patient_class_template.csv'}")
    else:
        print(f"Wrote patient_class.csv to {args.output_dir / 'patient_class.csv'}")
    print(f"Wrote preview summaries to {args.summary_dir}")


if __name__ == "__main__":
    main()
