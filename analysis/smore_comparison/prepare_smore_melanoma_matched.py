#!/usr/bin/env python3
"""Prepare SMORE cell input from the exact CISM NN/NP graph-vertex cohort."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = _REPO_ROOT
RAW = Path("/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv")
GRAPH = Path(
    "/groups/assafza_group/assafza/Omri/processed_data_23_10_2025/data/"
    "melanoma_distance_100_XX"
)
LABELS = GRAPH.parents[1] / "patient_class.csv"
OUTPUT = ROOT / "analysis/outputs/smore_matched"
TYPES = [
    "b cell", "cd4 apc", "cd4 t cell", "treg", "cd8 t cell", "DC", "gc",
    "hevs", "mac", "memory%cd4 t cell", "nk cell", "neutrophil", "stroma",
    "tumor", "unidentified", "blood vessels",
]


def graph_vertices(path: Path) -> dict[int, int]:
    vertices: dict[int, int] = {}
    with path.open() as handle:
        for line in handle:
            u, v, a, b = (int(x) for x in line.split())
            for node, code in ((u, a), (v, b)):
                if node in vertices and vertices[node] != code:
                    raise ValueError(f"Conflicting cell types for {path.name} node {node}")
                vertices[node] = code
    return vertices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(LABELS, header=None, names=["patient", "class"])
    labels = labels[labels["class"].isin(["NN", "NP"])].copy()
    labels["patient_number"] = labels.patient.str.removeprefix("Melanoma").astype(int)
    counts = labels["class"].value_counts().to_dict()
    if counts != {"NN": 20, "NP": 18}:
        raise ValueError(f"Unexpected cohort: {counts}")
    label_by_patient = labels.set_index("patient_number")["class"].to_dict()

    columns = ["patient number", "fov", "Group", "pred_mapped", "centroid-0", "centroid-1"]
    raw = pd.read_csv(RAW, usecols=columns)
    raw = raw[raw["patient number"].isin(label_by_patient)].copy()
    by_fov = {(int(patient), fov): part.reset_index(drop=True)
              for (patient, fov), part in raw.groupby(["patient number", "fov"], sort=False)}
    del raw

    frames = []
    manifest = []
    fov_dir = args.output_dir / "fovs"
    fov_dir.mkdir(exist_ok=True)
    for path in sorted(GRAPH.glob("Patient_*_FOV*.txt")):
        match = re.fullmatch(r"Patient_(\d+)_(FOV\d+)\.txt", path.name)
        if match is None:
            continue
        patient, fov = int(match[1]), match[2]
        if patient not in label_by_patient:
            continue
        source = by_fov.get((patient, fov))
        if source is None:
            raise ValueError(f"Raw cells missing for patient {patient}, {fov}")
        if set(source.Group) != {label_by_patient[patient]}:
            raise ValueError(f"Class disagreement for patient {patient}, {fov}")
        vertices = graph_vertices(path)
        node_ids = sorted(vertices)
        if not node_ids or node_ids[-1] >= len(source):
            raise ValueError(f"Invalid graph vertex index in {path}")
        if any(source.iloc[node].pred_mapped != TYPES[vertices[node]] for node in node_ids):
            raise ValueError(f"Graph/raw cell-type mismatch in {path}")
        if any(vertices[node] == 13 for node in node_ids):
            raise ValueError(f"Tumor cell in original CISM graph {path}")
        selected = source.iloc[node_ids]
        frame = pd.DataFrame({
            "Cell_ID": [node + 1 for node in node_ids],
            "SID": fov,
            "cellType": [vertices[node] + 1 for node in node_ids],
            "Centroid_X": selected["centroid-0"].to_numpy(),
            "Centroid_Y": selected["centroid-1"].to_numpy(),
        })
        input_file = fov_dir / f"{fov}.csv"
        frame.to_csv(input_file, index=False)
        frames.append(frame)
        manifest.append({
            "patient": patient, "class": label_by_patient[patient], "fov": fov,
            "graph_vertices": len(node_ids), "raw_cells": len(source),
            "graph_file": str(path), "smore_input": str(input_file),
        })

    cohort = pd.DataFrame(manifest)
    if cohort.patient.nunique() != 38 or cohort.fov.nunique() != 76:
        raise ValueError(f"Incomplete cohort: {cohort.patient.nunique()} patients, "
                         f"{cohort.fov.nunique()} FOVs")
    cells = pd.concat(frames, ignore_index=True)
    if len(cells) != 863448 or cells.isna().any().any():
        raise ValueError(f"Unexpected cell count or missing coordinate: {len(cells)}")
    cells.to_csv(args.output_dir / "smore_cells.csv", index=False)
    cohort.to_csv(args.output_dir / "cohort_fovs.csv", index=False)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "raw_cells": str(RAW), "graph_root": str(GRAPH), "labels": str(LABELS),
        "n_patients": 38, "class_counts": counts, "n_fovs": 76,
        "n_cells": len(cells), "cell_id": "original graph vertex ID plus one",
        "cell_type": "CISM graph code plus one; zero is reserved by SMORE",
        "note": "Only original CISM graph vertices are retained; tumor and isolated cells are absent.",
    }, indent=2) + "\n")
    print(f"Prepared {len(cells):,} cells from {len(cohort)} FOVs and 38 patients: {counts}")


if __name__ == "__main__":
    main()
