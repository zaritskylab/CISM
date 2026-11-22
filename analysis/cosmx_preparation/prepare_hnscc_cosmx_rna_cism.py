#!/usr/bin/env python3
"""Prepare reconstructed HNSCC CosMx RNA cell metadata for CISM.

This consumes the output of ``hnscc_cosmx_reconstruct_rna_subtypes.py`` and
writes FANMOD/CISM-compatible colored edge-list files. One graph is written per
CosMx TMA core/FOV, with node colors defined by the reconstructed 16 RNA
subtypes by default.
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_METADATA = Path(
    "analysis/hnscc_cosmx/reconstructed_rna_subtypes/"
    "hnscc_cosmx_rna_reconstructed_cell_metadata.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=Path("data/HNSCC_CosMx_RNA_reconstructed_16color"))
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("tutorial_runtime/hnscc_cosmx_rna_reconstructed_16color"),
    )
    parser.add_argument("--dataset-name", default="HNSCC")
    parser.add_argument(
        "--color-column",
        default="predicted_annot_cluster2",
        help="Cell metadata column to use as CISM node color.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.168,
        help="CosMx pixel size in microns/pixel.",
    )
    parser.add_argument(
        "--radius-um",
        type=float,
        default=30.0,
        help="Neighbor radius in microns for spatial graph construction.",
    )
    parser.add_argument("--min-cells-per-graph", type=int, default=25)
    parser.add_argument(
        "--response-csv",
        type=Path,
        default=None,
        help=(
            "Optional patient response table. Expected columns: PatientID and response_clean "
            "or response. Values should be Responder/NonResponder."
        ),
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write summaries and patient_class template, but skip graph txt creation.",
    )
    return parser.parse_args()


def clean_patient_id(value: object) -> str:
    return str(value).strip()


def load_response_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["PatientID", "response_clean"])

    response = pd.read_csv(path)
    if "PatientID" not in response.columns:
        raise ValueError(f"Response table must contain PatientID. Available columns: {list(response.columns)}")
    if "response_clean" not in response.columns:
        if "response" not in response.columns:
            raise ValueError("Response table must contain response_clean or response.")
        response = response.rename(columns={"response": "response_clean"})

    response = response[["PatientID", "response_clean"]].copy()
    response["PatientID"] = response["PatientID"].map(clean_patient_id)
    response["response_clean"] = response["response_clean"].astype(str).str.strip()
    response["response_clean"] = response["response_clean"].replace(
        {
            "R": "Responder",
            "NR": "NonResponder",
            "Non-Responder": "NonResponder",
            "Non-responder": "NonResponder",
            "non-responder": "NonResponder",
            "responder": "Responder",
        }
    )
    allowed = {"Responder", "NonResponder"}
    bad = sorted(set(response["response_clean"]) - allowed)
    if bad:
        raise ValueError(f"Unsupported response labels: {bad}. Expected {sorted(allowed)}.")
    if response["PatientID"].duplicated().any():
        raise ValueError("Response table contains duplicated PatientID values.")
    return response


def write_radius_graph(group: pd.DataFrame, output_file: Path, radius_px: float) -> int:
    coords = group[["x_px", "y_px"]].to_numpy(dtype=float)
    if len(coords) < 2:
        output_file.write_text("", encoding="utf-8")
        return 0

    tree = cKDTree(coords)
    pairs = sorted(tree.query_pairs(r=radius_px))
    if not pairs:
        _, nearest = tree.query(coords, k=2)
        pairs = sorted({tuple(sorted((idx, int(row[1])))) for idx, row in enumerate(nearest)})

    node_ids = group["local_node_id"].to_numpy()
    colors = group["color_id"].to_numpy()
    with output_file.open("w", encoding="utf-8") as handle:
        for left, right in pairs:
            handle.write(f"{node_ids[left]} {node_ids[right]} {int(colors[left])} {int(colors[right])}\n")
    return len(pairs)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metadata_csv)
    required = {
        "PatientID",
        "tma_fov_id",
        "cell_ID",
        "CenterX_global_px",
        "CenterY_global_px",
        args.color_column,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    df = df.copy()
    df["patient_id"] = df["PatientID"].map(clean_patient_id)
    df["fov"] = df["tma_fov_id"].astype(str)
    df["cell_type"] = df[args.color_column].astype(str).str.strip()
    df["x_px"] = pd.to_numeric(df["CenterX_global_px"], errors="coerce")
    df["y_px"] = pd.to_numeric(df["CenterY_global_px"], errors="coerce")
    df = df[df["x_px"].notna() & df["y_px"].notna() & df["cell_type"].notna()].copy()

    color_map = {label: idx for idx, label in enumerate(sorted(df["cell_type"].unique()))}
    df["color_id"] = df["cell_type"].map(color_map).astype(int)

    response = load_response_table(args.response_csv)
    if not response.empty:
        df = df.merge(response, on="PatientID", how="left")
    else:
        df["response_clean"] = pd.NA

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    patient_class = (
        df[["patient_id", "response_clean"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .assign(cism_patient=lambda table: args.dataset_name + table["patient_id"].astype(str))
    )
    labeled = patient_class.dropna(subset=["response_clean"])
    if not labeled.empty and labeled["patient_id"].duplicated().any():
        raise ValueError("A patient has conflicting response labels.")

    if labeled.empty:
        patient_class[["cism_patient", "response_clean"]].to_csv(
            args.output_dir / "patient_class_template.csv",
            header=False,
            index=False,
        )
    else:
        labeled[["cism_patient", "response_clean"]].to_csv(
            args.output_dir / "patient_class.csv",
            header=False,
            index=False,
        )

    radius_px = args.radius_um / args.pixel_size_um
    graph_rows = []
    for (patient_id, fov), group in df.groupby(["patient_id", "fov"], sort=True):
        if len(group) < args.min_cells_per_graph:
            continue
        output_file = args.output_dir / f"Patient_{patient_id}_FOV{fov}.txt"
        edge_count = None
        if not args.preview_only:
            graph_group = group.reset_index(drop=True).copy()
            graph_group["local_node_id"] = np.arange(len(graph_group), dtype=int)
            edge_count = write_radius_graph(graph_group, output_file, radius_px)
        graph_rows.append(
            {
                "patient_id": patient_id,
                "fov": fov,
                "cells": int(len(group)),
                "edges": edge_count,
                "file": output_file.name,
            }
        )

    graph_summary = pd.DataFrame(graph_rows)
    graph_summary.to_csv(args.summary_dir / "graph_summary.csv", index=False)
    pd.Series(color_map, name="color_id").rename_axis("node_color").to_csv(
        args.summary_dir / "node_color_map.csv"
    )
    df[
        [
            "patient_id",
            "fov",
            "cell_ID",
            "cell_type",
            "color_id",
            "x_px",
            "y_px",
            "response_clean",
        ]
    ].head(5000).to_csv(args.summary_dir / "metadata_preview_head.csv", index=False)

    summary = {
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "summary_dir": str(args.summary_dir),
        "dataset_name": args.dataset_name,
        "color_column": args.color_column,
        "pixel_size_um": args.pixel_size_um,
        "radius_um": args.radius_um,
        "radius_px": radius_px,
        "cells": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "graphs": int(len(graph_summary)),
        "node_colors": int(len(color_map)),
        "clinical_labels_present": bool(not labeled.empty),
        "response_patient_counts": labeled["response_clean"].value_counts().to_dict(),
        "response_cell_counts": df["response_clean"].dropna().value_counts().to_dict(),
    }
    with (args.summary_dir / "preparation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if labeled.empty:
        print(f"Wrote patient_class_template.csv to {args.output_dir / 'patient_class_template.csv'}")
    else:
        print(f"Wrote patient_class.csv to {args.output_dir / 'patient_class.csv'}")


if __name__ == "__main__":
    main()
