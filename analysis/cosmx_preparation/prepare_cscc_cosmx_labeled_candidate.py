#!/usr/bin/env python3
"""Build a labeled cSCC CosMx candidate dataset for CISM.

The public Dryad release includes the Seurat object metadata but not the
authors' private TMA-ID/FOV mapping files. This script reconstructs that
mapping from FOV centroid clusters and the published supplemental clinical
table, then writes a baseline-only responder/non-responder CISM graph dataset.
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
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from analysis.cosmx_preparation.prepare_cscc_cosmx_cism import make_color, write_radius_graph


CLINICAL_ROWS = [
    (1, "840003001", "Baseline", "Cemiplimab", "Unknown", "mPR"),
    (2, "840003001", "Baseline", "Cemiplimab", "Unknown", "mPR"),
    (3, "840003003", "Baseline", "Cemiplimab", "Unknown", "pCR"),
    (4, "840003003", "Post-dose", "Cemiplimab", "Unknown", "pCR"),
    (5, "Nontrial1", "Baseline", "Cemiplimab", "Unknown", "Unknown"),
    (6, "Nontrial3", "Unknown", "Unknown", "Unknown", "Unknown"),
    (7, "840003005", "Baseline", "Cemiplimab", "Unknown", "Unknown"),
    (8, "840003007", "Baseline", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (9, "840003007", "Baseline", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (10, "840003007", "Post-dose", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (11, "840003007", "Surgery", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (12, "840003007", "Surgery", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (13, "840003007", "Surgery", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (14, "840003009", "Baseline", "Cemiplimab", "Unknown", "pCR"),
    (15, "840003009", "Baseline", "Cemiplimab", "Unknown", "pCR"),
    (16, "840003009", "Post-dose", "Cemiplimab", "Unknown", "pCR"),
    (17, "840003009", "Post-dose", "Cemiplimab", "Unknown", "pCR"),
    (18, "SU001", "Baseline", "Atezolizumab", "20", "mPR"),
    (19, "SU001", "Post-dose", "Atezolizumab", "20", "mPR"),
    (20, "SU003", "Baseline", "Atezolizumab", "15", "Non-mPR/pCR"),
    (21, "SU003", "Baseline", "Atezolizumab", "15", "Non-mPR/pCR"),
    (22, "SU003", "Post-dose", "Atezolizumab", "15", "Non-mPR/pCR"),
    (23, "SU003", "Post-dose", "Atezolizumab", "15", "Non-mPR/pCR"),
    (24, "SU005", "Baseline", "Atezolizumab", "100", "Non-mPR/pCR"),
    (25, "Nontrial2", "Post-dose", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (26, "Nontrial2", "Post-dose", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (27, "Nontrial2", "Surgery", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (28, "Nontrial2", "Surgery", "Cemiplimab", "Unknown", "Non-mPR/pCR"),
    (29, "SU006", "Baseline", "Atezolizumab", "Unknown", "pCR"),
    (30, "SU006", "Baseline", "Atezolizumab", "Unknown", "pCR"),
    (31, "SU009", "Baseline", "Atezolizumab", "7.5", "Non-mPR/pCR"),
    (32, "SU009", "Baseline", "Atezolizumab", "7.5", "Non-mPR/pCR"),
    (33, "SU009", "Post-dose", "Atezolizumab", "7.5", "Non-mPR/pCR"),
    (34, "SU009", "Post-dose", "Atezolizumab", "7.5", "Non-mPR/pCR"),
    (35, "SU009", "Post-dose", "Atezolizumab", "7.5", "Non-mPR/pCR"),
    (36, "SU009", "Surgery", "Atezolizumab", "7.5", "Non-mPR/pCR"),
    (37, "SU009", "Surgery", "Atezolizumab", "7.5", "Non-mPR/pCR"),
    (38, "SU011", "Baseline", "Atezolizumab", "1", "mPR"),
    (39, "SU011", "Post-dose", "Atezolizumab", "1", "mPR"),
    (40, "SU010", "Baseline", "Atezolizumab", "30", "Non-mPR/pCR"),
    (41, "SU010", "Baseline", "Atezolizumab", "30", "Non-mPR/pCR"),
    (42, "SU010", "Post-dose", "Atezolizumab", "30", "Non-mPR/pCR"),
    (43, "SU013", "Baseline", "Atezolizumab", "90", "pCR"),
    (44, "SU013", "Post-dose", "Atezolizumab", "90", "pCR"),
    (45, "SU014", "Baseline", "Atezolizumab", "40", "pCR"),
    (46, "SU014", "Post-dose", "Atezolizumab", "40", "pCR"),
    (47, "SU017", "Baseline", "Atezolizumab", "<1", "Non-mPR/pCR"),
    (48, "SU018", "Baseline", "Atezolizumab", "50", "pCR"),
    (49, "SU019", "Baseline", "Atezolizumab", "2", "Non-mPR/pCR"),
    (50, "SU019", "Post-dose", "Atezolizumab", "2", "Non-mPR/pCR"),
    (51, "SU020", "Baseline", "Atezolizumab", "10", "Non-mPR/pCR"),
    (52, "SU023", "Baseline", "Atezolizumab", "Unknown", "Unknown"),
    (53, "SU024", "Post-dose", "Atezolizumab", "0", "mPR"),
    (54, "SU026", "Baseline", "Atezolizumab", "<1", "Non-mPR/pCR"),
    (55, "SU026", "Post-dose", "Atezolizumab", "<1", "Non-mPR/pCR"),
    (56, "SU029", "Baseline", "Atezolizumab", "80", "pCR"),
    (57, "SU030", "Baseline", "Atezolizumab", "3", "Non-mPR/pCR"),
    (58, "SU030", "Post-dose", "Atezolizumab", "3", "Non-mPR/pCR"),
    (59, "SU021", "Baseline", "Atezolizumab", "60", "pCR"),
]

RESPONSE_MAP = {
    "mPR": "Responder",
    "pCR": "Responder",
    "Non-mPR/pCR": "NonResponder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/cSCC_CosMx/source_metadata/cscc_cosmx_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cSCC_CosMx_baseline_reconstructed"),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("tutorial_runtime/cscc_cosmx_baseline_reconstructed"),
    )
    parser.add_argument("--dataset-name", default="cSCC")
    parser.add_argument(
        "--numeric-patient-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use stable numeric patient IDs in graph filenames and patient_class.csv. "
            "The current CISM parser casts patient IDs to int16 after FANMOD parsing."
        ),
    )
    parser.add_argument(
        "--color-mode",
        choices=["cell_type", "spatial_niche", "cell_type_spatial_niche"],
        default="cell_type_spatial_niche",
    )
    parser.add_argument("--radius-mm", type=float, default=0.04)
    parser.add_argument("--min-cells-per-fov", type=int, default=25)
    return parser.parse_args()


def connected_components(points: np.ndarray, eps: float) -> list[list[int]]:
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, eps)
    seen = np.zeros(len(points), dtype=bool)
    components = []

    for start in range(len(points)):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)

    return components


def clinical_table() -> pd.DataFrame:
    columns = [
        "tma_id",
        "subject_study_id",
        "biopsy_timepoint",
        "treatment_drug",
        "baseline_pd_l1",
        "pathologic_response",
    ]
    clinical = pd.DataFrame(CLINICAL_ROWS, columns=columns)
    clinical["response_clean"] = clinical["pathologic_response"].map(RESPONSE_MAP)
    return clinical


def reconstruct_fov_map(metadata: pd.DataFrame) -> pd.DataFrame:
    fovs = (
        metadata.groupby(["slide_ID_numeric", "fov"], as_index=False)
        .agg(
            x=("x_slide_mm", "mean"),
            y=("y_slide_mm", "mean"),
            cells=("fov", "size"),
        )
        .sort_values(["slide_ID_numeric", "fov"])
    )

    reconstruction_specs = {
        2: {"eps": 1.05, "tma_ids": list(range(1, 34)), "tma_map": "TMA1"},
        1: {"eps": 0.75, "tma_ids": list(range(34, 55)) + list(range(56, 60)), "tma_map": "TMA2"},
    }
    rows = []
    for slide_id, spec in reconstruction_specs.items():
        slide_fovs = fovs[fovs["slide_ID_numeric"] == slide_id].reset_index(drop=True)
        points = slide_fovs[["x", "y"]].to_numpy(dtype=float)
        components = connected_components(points, spec["eps"])
        clusters = []
        for core_index, component in enumerate(components):
            core_fovs = slide_fovs.iloc[component].copy()
            clusters.append(
                {
                    "core_index": core_index,
                    "min_fov": int(core_fovs["fov"].min()),
                    "max_fov": int(core_fovs["fov"].max()),
                    "n_fovs": int(core_fovs["fov"].nunique()),
                    "cells": int(core_fovs["cells"].sum()),
                    "center_x": float(core_fovs["x"].mean()),
                    "center_y": float(core_fovs["y"].mean()),
                    "fovs": sorted(core_fovs["fov"].astype(int).tolist()),
                }
            )

        clusters = sorted(clusters, key=lambda row: row["min_fov"])
        if len(clusters) != len(spec["tma_ids"]):
            raise ValueError(
                f"Slide {slide_id} produced {len(clusters)} reconstructed cores, "
                f"expected {len(spec['tma_ids'])}."
            )

        for tma_id, cluster in zip(spec["tma_ids"], clusters):
            for fov in cluster["fovs"]:
                rows.append(
                    {
                        "slide_ID_numeric": slide_id,
                        "fov": fov,
                        "tma_id": tma_id,
                        "tma_map": spec["tma_map"],
                        "reconstruction_eps_mm": spec["eps"],
                        "core_min_fov": cluster["min_fov"],
                        "core_max_fov": cluster["max_fov"],
                        "core_n_fovs": cluster["n_fovs"],
                        "core_cells": cluster["cells"],
                        "core_center_x": cluster["center_x"],
                        "core_center_y": cluster["center_y"],
                    }
                )

    return pd.DataFrame(rows).sort_values(["slide_ID_numeric", "fov"])


def reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("Patient_*.txt"):
        path.unlink()
    for filename in ["patient_class.csv", "patient_class_template.csv"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def safe_fov_id(row: pd.Series) -> str:
    return f"TMA{int(row['tma_id']):03d}_fov{int(row['fov']):03d}"


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata_csv)
    required = {
        "slide_ID_numeric",
        "fov",
        "x_slide_mm",
        "y_slide_mm",
        "CoarseAnnotationNew",
        "SpatialClusterNames",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    fov_map = reconstruct_fov_map(metadata)
    clinical = clinical_table()
    fov_map = fov_map.merge(clinical, on="tma_id", how="left", validate="many_to_one")
    joined = metadata.merge(fov_map, on=["slide_ID_numeric", "fov"], how="left", validate="many_to_one")
    joined["response_clean"] = joined["pathologic_response"].map(RESPONSE_MAP)

    candidate = joined[
        (joined["biopsy_timepoint"] == "Baseline")
        & joined["response_clean"].isin(["Responder", "NonResponder"])
    ].copy()
    candidate = candidate[
        candidate["x_slide_mm"].notna()
        & candidate["y_slide_mm"].notna()
        & candidate["CoarseAnnotationNew"].notna()
        & candidate["SpatialClusterNames"].notna()
    ].copy()

    candidate["subject_study_id"] = candidate["subject_study_id"].astype(str)
    subject_order = sorted(candidate["subject_study_id"].unique())
    subject_to_cism_id = {subject: index + 1 for index, subject in enumerate(subject_order)}
    if args.numeric_patient_ids:
        candidate["patient_id"] = candidate["subject_study_id"].map(subject_to_cism_id).astype(str)
    else:
        candidate["patient_id"] = candidate["subject_study_id"]
    candidate["cism_fov"] = candidate.apply(safe_fov_id, axis=1)
    candidate["node_color"] = make_color(
        candidate,
        args.color_mode,
        "CoarseAnnotationNew",
        "SpatialClusterNames",
    )
    color_map = {
        label: idx for idx, label in enumerate(sorted(candidate["node_color"].unique()))
    }
    candidate["color_id"] = candidate["node_color"].map(color_map).astype(int)
    candidate["centroid-0"] = pd.to_numeric(candidate["x_slide_mm"], errors="coerce")
    candidate["centroid-1"] = pd.to_numeric(candidate["y_slide_mm"], errors="coerce")

    reset_output_dir(args.output_dir)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    patient_class = (
        candidate[["patient_id", "subject_study_id", "response_clean"]]
        .drop_duplicates()
        .sort_values("patient_id")
    )
    if patient_class["patient_id"].duplicated().any():
        conflicts = patient_class[patient_class["patient_id"].duplicated(keep=False)]
        raise ValueError(f"Conflicting patient labels:\n{conflicts}")
    patient_class.assign(cism_patient=args.dataset_name + patient_class["patient_id"])[
        ["cism_patient", "response_clean"]
    ].to_csv(args.output_dir / "patient_class.csv", header=False, index=False)
    patient_class[
        ["patient_id", "subject_study_id", "response_clean"]
    ].to_csv(args.summary_dir / "patient_id_codebook.csv", index=False)

    graph_rows = []
    for (patient_id, cism_fov), group in candidate.groupby(["patient_id", "cism_fov"], sort=True):
        if len(group) < args.min_cells_per_fov:
            continue
        graph = group.reset_index(drop=True).copy()
        graph["local_node_id"] = np.arange(len(graph), dtype=int)
        output_file = args.output_dir / f"Patient_{patient_id}_FOV{cism_fov}.txt"
        edge_count = write_radius_graph(graph, output_file, args.radius_mm)
        first = graph.iloc[0]
        graph_rows.append(
            {
                "patient_id": patient_id,
                "subject_study_id": first["subject_study_id"],
                "response_clean": first["response_clean"],
                "cism_fov": cism_fov,
                "tma_id": int(first["tma_id"]),
                "slide_ID_numeric": int(first["slide_ID_numeric"]),
                "source_fov": int(first["fov"]),
                "cells": int(len(graph)),
                "edges": int(edge_count),
                "file": output_file.name,
            }
        )

    graph_summary = pd.DataFrame(graph_rows).sort_values(["patient_id", "tma_id", "source_fov"])
    fov_map.to_csv(args.summary_dir / "reconstructed_tma_fov_map.csv", index=False)
    clinical.to_csv(args.summary_dir / "supplemental_clinical_table.csv", index=False)
    graph_summary.to_csv(args.summary_dir / "graph_summary.csv", index=False)
    pd.Series(color_map, name="color_id").rename_axis("node_color").to_csv(
        args.summary_dir / "node_color_map.csv"
    )
    candidate[
        [
            "patient_id",
            "subject_study_id",
            "response_clean",
            "biopsy_timepoint",
            "tma_id",
            "slide_ID_numeric",
            "fov",
            "cism_fov",
            "CoarseAnnotationNew",
            "SpatialClusterNames",
            "node_color",
            "x_slide_mm",
            "y_slide_mm",
        ]
    ].head(5000).to_csv(args.summary_dir / "metadata_preview_head.csv", index=False)

    patient_summary = (
        patient_class.groupby("response_clean", observed=True)["patient_id"]
        .nunique()
        .to_dict()
    )
    tissue_summary = (
        clinical[
            (clinical["biopsy_timepoint"] == "Baseline")
            & clinical["response_clean"].isin(["Responder", "NonResponder"])
        ]
        .groupby("response_clean", observed=True)["tma_id"]
        .nunique()
        .to_dict()
    )
    graph_response_summary = (
        graph_summary.groupby("response_clean", observed=True)
        .agg(graphs=("file", "count"), cells=("cells", "sum"), edges=("edges", "sum"))
        .reset_index()
    )
    graph_response_summary.to_csv(args.summary_dir / "response_graph_summary.csv", index=False)

    summary = {
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "summary_dir": str(args.summary_dir),
        "dataset_name": args.dataset_name,
        "filter": "Baseline only; excludes Unknown pathologic response",
        "graph_unit": "one graph per original CosMx FOV, grouped under true subject/patient IDs",
        "patient_id_encoding": (
            "numeric CISM patient IDs with real subject IDs in patient_id_codebook.csv"
            if args.numeric_patient_ids
            else "real subject IDs used directly"
        ),
        "radius_mm": args.radius_mm,
        "radius_microns": args.radius_mm * 1000,
        "color_mode": args.color_mode,
        "node_colors": len(color_map),
        "cells": int(candidate.shape[0]),
        "graphs": int(graph_summary.shape[0]),
        "patients": int(patient_class["patient_id"].nunique()),
        "tissue_cores": int(candidate["tma_id"].nunique()),
        "response_patient_counts": patient_summary,
        "response_tissue_core_counts": tissue_summary,
        "fov_cluster_reconstruction": {
            "slide2_TMA1": "eps=1.05 mm; TMA IDs 1-33 by increasing FOV acquisition order",
            "slide1_TMA2": "eps=0.75 mm; TMA IDs 34-54 and 56-59 by increasing FOV acquisition order; TMA ID 55 omitted as not imaged",
        },
        "caveat": (
            "Clinical labels are reconstructed because the public Dryad release omits "
            "the authors' TMA ID labelled.csv and CosMx TMA mapping.xlsx files."
        ),
    }
    with (args.summary_dir / "preparation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
