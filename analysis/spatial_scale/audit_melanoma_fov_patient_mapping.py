#!/usr/bin/env python3
"""Audit patient ownership of original and prepared melanoma FOV graph files."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import re
from pathlib import Path

import pandas as pd


ROOT = _REPO_ROOT
MELANOMA = ROOT / "melanoma_tutorials" / "pet_data" / "Melanoma"
RUNTIME = ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
OUTPUT = RUNTIME / "crop_fov_scale_experiment" / "cell_count_scale_control"
PATTERN = re.compile(r"Patient_(\d+)_FOV(FOV\d+)\.txt")


def main() -> None:
    classes = pd.read_csv(MELANOMA / "patient_class.csv", header=None, names=["patient_uid", "class"])
    class_lookup = classes.set_index("patient_uid")["class"]
    original = []
    for graph in MELANOMA.glob("Patient_*_FOV*.txt"):
        match = PATTERN.fullmatch(graph.name)
        if match is None:
            raise ValueError(f"Unrecognized original graph filename: {graph}")
        patient = int(match.group(1))
        patient_uid = f"Melanoma{patient}"
        original.append({
            "dataset": "original",
            "patient_uid": patient_uid,
            "patient_class": class_lookup.loc[patient_uid],
            "source_fov": match.group(2),
            "derived_fov": match.group(2),
            "graph_file": str(graph.relative_to(ROOT)),
        })
    original_df = pd.DataFrame(original)
    if original_df["source_fov"].duplicated().any():
        raise ValueError("One or more original FOV IDs belong to multiple patients")
    original_pairs = set(zip(original_df.patient_uid, original_df.source_fov))

    rows = list(original)
    for kind, levels in (("split", (2, 4, 8, 16)), ("crop", (50, 60, 70, 80, 90))):
        for level in levels:
            token = f"{level:02d}" if kind == "split" else f"{level:03d}"
            experiment = RUNTIME / f"{kind}_fov_scale_experiment"
            dataset = f"Melanoma_NN_NP_{kind}_{token}"
            folder = experiment / "prepared_datasets" / dataset
            metadata = pd.read_csv(experiment / f"{kind}_{token}_metadata.csv")
            local_classes = pd.read_csv(folder / "patient_class.csv", header=None,
                                        names=["patient_uid", "class"]).set_index("patient_uid")["class"]
            derived_field = "split_fov" if kind == "split" else "crop_fov"
            graph_names = {p.name for p in folder.glob("Patient_*_FOV*.txt")}
            if graph_names != set(metadata.file) or len(graph_names) != len(metadata):
                raise ValueError(f"Graph files and metadata disagree in {dataset}")
            pairs = set(zip(metadata.patient_uid, metadata.source_fov))
            if not pairs.issubset(original_pairs):
                raise ValueError(f"Unknown patient/source-FOV pairs in {dataset}")
            for record in metadata.itertuples(index=False):
                derived_fov = getattr(record, derived_field)
                expected_file = f"Patient_{record.patient_number}_FOV{derived_fov}.txt"
                if record.file != expected_file or not derived_fov.startswith(record.source_fov + "_"):
                    raise ValueError(f"Patient or source FOV mismatch for {record.file}")
                if (record.patient_class != class_lookup.loc[record.patient_uid]
                        or record.patient_class != local_classes.loc[record.patient_uid]):
                    raise ValueError(f"Patient class mismatch for {record.file}")
                rows.append({
                    "dataset": dataset,
                    "patient_uid": record.patient_uid,
                    "patient_class": record.patient_class,
                    "source_fov": record.source_fov,
                    "derived_fov": derived_fov,
                    "graph_file": str((folder / record.file).relative_to(ROOT)),
                })

    manifest = pd.DataFrame(rows).sort_values(["dataset", "patient_uid", "source_fov", "derived_fov"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT / "melanoma_fov_patient_dataset_manifest.csv", index=False)
    summary = manifest.groupby("dataset", observed=True).agg(
        graph_files=("graph_file", "size"),
        patients=("patient_uid", "nunique"),
        source_fovs=("source_fov", "nunique"),
    )
    summary.to_csv(OUTPUT / "melanoma_fov_patient_dataset_audit_summary.csv")
    print(summary.to_string())
    print("All patient/FOV mappings and class labels agree with metadata and original graphs.")


if __name__ == "__main__":
    main()
