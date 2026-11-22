#!/usr/bin/env python3
"""Create a balanced lightweight cSCC CosMx CISM screening subset."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/cSCC_CosMx_baseline_reconstructed"),
    )
    parser.add_argument(
        "--graph-summary",
        type=Path,
        default=Path("tutorial_runtime/cscc_cosmx_baseline_reconstructed/graph_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cSCC_CosMx_baseline_reconstructed_screening"),
    )
    parser.add_argument("--patients-per-class", type=int, default=4)
    parser.add_argument("--graphs-per-patient", type=int, default=2)
    parser.add_argument("--dataset-name", default="cSCC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.graph_summary)
    selected_rows = []

    for label, label_df in summary.groupby("response_clean", sort=True):
        patient_cost = (
            label_df.groupby("patient_id", as_index=False)
            .agg(total_edges=("edges", "sum"), total_cells=("cells", "sum"), graph_count=("file", "count"))
            .sort_values(["total_edges", "total_cells", "patient_id"])
            .head(args.patients_per_class)
        )
        for patient_id in patient_cost["patient_id"]:
            patient_graphs = (
                label_df[label_df["patient_id"] == patient_id]
                .sort_values(["edges", "cells", "file"])
                .head(args.graphs_per_patient)
            )
            selected_rows.append(patient_graphs)

    selected = pd.concat(selected_rows, ignore_index=True).sort_values(
        ["response_clean", "patient_id", "edges", "file"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.output_dir.glob("Patient_*.txt"):
        path.unlink()
    for path in ["patient_class.csv", "subset_graph_summary.csv", "subset_summary.json"]:
        target = args.output_dir / path
        if target.exists():
            target.unlink()

    for filename in selected["file"]:
        shutil.copy2(args.source_dir / filename, args.output_dir / filename)

    patient_class = (
        selected[["patient_id", "response_clean"]]
        .drop_duplicates()
        .sort_values("patient_id")
    )
    patient_class.assign(cism_patient=args.dataset_name + patient_class["patient_id"].astype(str))[
        ["cism_patient", "response_clean"]
    ].to_csv(args.output_dir / "patient_class.csv", header=False, index=False)
    selected.to_csv(args.output_dir / "subset_graph_summary.csv", index=False)

    subset_summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "graphs": int(selected.shape[0]),
        "patients": int(selected["patient_id"].nunique()),
        "cells": int(selected["cells"].sum()),
        "edges": int(selected["edges"].sum()),
        "response_patient_counts": (
            patient_class.groupby("response_clean", observed=True)["patient_id"].nunique().to_dict()
        ),
        "response_graph_counts": selected["response_clean"].value_counts().to_dict(),
        "selection": (
            f"{args.patients_per_class} lowest-edge patients per class; "
            f"{args.graphs_per_patient} lowest-edge graphs per patient"
        ),
    }
    with (args.output_dir / "subset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(subset_summary, handle, indent=2, sort_keys=True)
    print(json.dumps(subset_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
