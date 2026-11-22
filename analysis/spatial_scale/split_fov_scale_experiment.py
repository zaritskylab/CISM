#!/usr/bin/env python3
"""Prepare and analyze split-FOV melanoma motif-scale experiments.

Stages:
  prepare   Build sub-FOV network datasets for total grid parts 2, 4, 8, and 16.
  summarize Summarize already-extracted split motif artifacts against original FOV motifs.

The FANMOD/CISM extraction itself can be run on the prepared datasets with the
printed commands or by importing the generated dataset paths into the existing
CISM pipeline.
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
import os
import pickle
import sys
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, distance
from tqdm.auto import tqdm

SCRIPT_DIR = ANALYSIS_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.cohort_imbalance.imbalanced_bootstrap_stringency import (  # noqa: E402
    discover_hard_discriminative_motifs,
    evaluate_hard_discriminative_loocv,
    load_patient_classes,
    tune_stringency,
)
from cism.cism import CISM, DiscriminativeMotifs, TissueStateDiscriminativeMotifs  # noqa: E402
from multi_class.config import (  # noqa: E402
    DATASET_NAME,
    DATASET_TYPE,
    FANMOD_BINARY_DIR,
    FANMOD_ITERATIONS,
    MAX_DISTANCE,
    MOTIF_SIZE,
    fanmod_executable_name,
)
from multi_class.data_artifacts import COMMON_CELLS_TYPE, TISSUE_STATE_TO_STRING  # noqa: E402


DEFAULT_RAW_CENTROIDS_CSV = Path("/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv")
DEFAULT_SCALE_TABLE = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "fov_spatial_scale_nn_np"
    / "fov_spatial_scale_per_fov.csv"
)
DEFAULT_PATIENT_CLASS_CSV = REPO_ROOT / "melanoma_tutorials" / "pet_data" / "Melanoma" / "patient_class.csv"
DEFAULT_ORIGINAL_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime" / "split_fov_scale_experiment"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser("prepare", help="Create split-FOV graph datasets.")
    add_common_args(prepare)
    prepare.add_argument("--parts", nargs="+", type=int, default=[2, 4, 8, 16])
    prepare.add_argument("--min-cells-per-tile", type=int, default=4)
    prepare.add_argument("--chunksize", type=int, default=1_000_000)
    prepare.add_argument("--force", action="store_true")

    run_cism = subparsers.add_parser("run-cism", help="Run CISM/FANMOD for one prepared split dataset.")
    add_common_args(run_cism)
    run_cism.add_argument("--parts", type=int, required=True)
    run_cism.add_argument("--n-jobs", type=int, default=8)
    run_cism.add_argument("--force", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Summarize split motif artifacts if present.")
    add_common_args(summarize)
    summarize.add_argument("--parts", nargs="+", type=int, default=[2, 4, 8, 16])
    summarize.add_argument("--p-value-threshold", type=float, default=0.05)
    summarize.add_argument("--stringency-grid", nargs="+", type=float, default=[x / 10 for x in range(0, 11)])
    summarize.add_argument("--n-optuna-trials", type=int, default=20)
    summarize.add_argument("--max-features", type=int, default=30)
    summarize.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-centroids-csv", type=Path, default=DEFAULT_RAW_CENTROIDS_CSV)
    parser.add_argument("--scale-table", type=Path, default=DEFAULT_SCALE_TABLE)
    parser.add_argument("--patient-class-csv", type=Path, default=DEFAULT_PATIENT_CLASS_CSV)
    parser.add_argument("--original-artifact", type=Path, default=DEFAULT_ORIGINAL_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--labels", nargs="+", default=["NN", "NP"])
    parser.add_argument("--patient-col", default="patient number")
    parser.add_argument("--fov-col", default="fov")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--x-col", default="centroid-0")
    parser.add_argument("--y-col", default="centroid-1")
    parser.add_argument("--cell-type-col", default="pred_mapped")
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE)


def grid_shape(total_parts: int) -> tuple[int, int]:
    if total_parts <= 1:
        raise ValueError("Grid parts must be greater than 1.")
    root = int(math.sqrt(total_parts))
    for rows in range(root, 0, -1):
        if total_parts % rows == 0:
            cols = total_parts // rows
            return rows, cols
    return 1, total_parts


def load_selected_fovs(args: argparse.Namespace) -> pd.DataFrame:
    fov_scale = pd.read_csv(args.scale_table)
    fov_scale = fov_scale[fov_scale["patient_class"].isin(args.labels)].copy()
    if "fov" not in fov_scale.columns:
        fov_scale["fov"] = "FOV" + fov_scale["fov_number"].astype(int).astype(str)
    return fov_scale[["patient_number", "patient_uid", "patient_class", "fov", "fov_number"]].copy()


def load_centroids_for_selected_fovs(args: argparse.Namespace, selected_fovs: pd.DataFrame) -> pd.DataFrame:
    usecols = [args.patient_col, args.fov_col, args.cell_id_col, args.x_col, args.y_col, args.cell_type_col]
    patients = set(selected_fovs["patient_number"])
    wanted = set(zip(selected_fovs["patient_number"], selected_fovs["fov"]))
    chunks = []
    for chunk in pd.read_csv(args.raw_centroids_csv, usecols=usecols, chunksize=args.chunksize):
        chunk = chunk[chunk[args.patient_col].isin(patients)]
        if chunk.empty:
            continue
        pairs = pd.MultiIndex.from_frame(chunk[[args.patient_col, args.fov_col]])
        mask = pairs.isin(wanted)
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        raise ValueError("No selected centroid rows found.")
    return pd.concat(chunks, ignore_index=True)


def assign_grid_tiles(cells: pd.DataFrame, args: argparse.Namespace, total_parts: int) -> pd.DataFrame:
    rows, cols = grid_shape(total_parts)
    frames = []
    for (patient, fov), group in cells.groupby([args.patient_col, args.fov_col], observed=True):
        group = group.copy()
        x_min, x_max = group[args.x_col].min(), group[args.x_col].max()
        y_min, y_max = group[args.y_col].min(), group[args.y_col].max()
        x_span = max(float(x_max - x_min), np.finfo(float).eps)
        y_span = max(float(y_max - y_min), np.finfo(float).eps)
        col = np.floor(((group[args.x_col] - x_min) / x_span) * cols).astype(int).clip(0, cols - 1)
        row = np.floor(((group[args.y_col] - y_min) / y_span) * rows).astype(int).clip(0, rows - 1)
        group["source_fov"] = fov
        group["grid_parts"] = total_parts
        group["grid_rows"] = rows
        group["grid_cols"] = cols
        group["tile_row"] = row
        group["tile_col"] = col
        group["split_fov"] = [
            f"{fov}_p{total_parts:02d}_r{r:02d}_c{c:02d}" for r, c in zip(row, col)
        ]
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def write_network_file(group: pd.DataFrame, args: argparse.Namespace, type_to_id: dict[str, int], output_path: Path) -> int:
    coords = group[[args.x_col, args.y_col]].to_numpy(dtype=float)
    types = group[args.cell_type_col].astype(str).map(type_to_id).to_numpy(dtype=int)
    if len(group) < 2:
        output_path.write_text("")
        return 0

    if len(group) < 3:
        neighbor_pairs = [(0, 1), (1, 0)]
    else:
        indptr, neighbors = Delaunay(coords).vertex_neighbor_vertices
        neighbor_pairs = [
            (i, int(j))
            for i in range(len(coords))
            for j in neighbors[indptr[i] : indptr[i + 1]]
            if i != int(j)
        ]

    edge_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for left, right in neighbor_pairs:
            if distance.euclidean(coords[left], coords[right]) <= args.max_distance:
                handle.write(f"{left} {right} {types[left]} {types[right]}\n")
                edge_count += 1
    return edge_count


def prepare_split_dataset(args: argparse.Namespace, total_parts: int) -> dict:
    dataset_root = args.output_root / "prepared_datasets"
    dataset_name = f"Melanoma_NN_NP_split_{total_parts:02d}"
    dataset_dir = dataset_root / dataset_name
    metadata_path = args.output_root / f"split_{total_parts:02d}_metadata.csv"
    summary_path = args.output_root / f"split_{total_parts:02d}_preparation_summary.json"

    if (
        dataset_dir.exists()
        and any(dataset_dir.glob("Patient_*_FOV*.txt"))
        and summary_path.exists()
        and not args.force
    ):
        return json.loads(summary_path.read_text()) if summary_path.exists() else {"dataset_dir": str(dataset_dir)}

    dataset_dir.mkdir(parents=True, exist_ok=True)
    selected_fovs = load_selected_fovs(args)
    cells = load_centroids_for_selected_fovs(args, selected_fovs)
    split_cells = assign_grid_tiles(cells, args, total_parts)

    type_to_id = {label: int(type_id) for type_id, label in COMMON_CELLS_TYPE.items()}
    missing_types = sorted(set(split_cells[args.cell_type_col].astype(str)) - set(type_to_id))
    if missing_types:
        raise ValueError(f"Split cells contain labels absent from COMMON_CELLS_TYPE: {missing_types}")
    tile_sizes = split_cells.groupby([args.patient_col, "split_fov"], observed=True).size()
    eligible_tiles = set(tile_sizes[tile_sizes >= args.min_cells_per_tile].index)

    rows = []
    for (patient, split_fov), group in tqdm(
        split_cells.groupby([args.patient_col, "split_fov"], observed=True),
        desc=f"Writing split {total_parts} graph files",
        unit="tile",
    ):
        if (patient, split_fov) not in eligible_tiles:
            continue
        output_file = dataset_dir / f"Patient_{patient}_FOV{split_fov}.txt"
        edge_count = write_network_file(group, args, type_to_id, output_file)
        source_fov = group["source_fov"].iloc[0]
        rows.append(
            {
                "patient_number": int(patient),
                "patient_uid": f"{DATASET_NAME}{patient}",
                "source_fov": source_fov,
                "split_fov": split_fov,
                "grid_parts": total_parts,
                "grid_rows": int(group["grid_rows"].iloc[0]),
                "grid_cols": int(group["grid_cols"].iloc[0]),
                "tile_row": int(group["tile_row"].iloc[0]),
                "tile_col": int(group["tile_col"].iloc[0]),
                "cells": int(len(group)),
                "edges": int(edge_count),
                "file": output_file.name,
            }
        )

    metadata = pd.DataFrame(rows)
    metadata = metadata.merge(
        selected_fovs[["patient_number", "patient_class", "fov_number"]],
        left_on=["patient_number", "source_fov"],
        right_on=["patient_number", selected_fovs["fov"]],
        how="left",
    ) if False else metadata
    patient_class_lookup = selected_fovs.drop_duplicates("patient_uid").set_index("patient_uid")["patient_class"]
    metadata["patient_class"] = metadata["patient_uid"].map(patient_class_lookup)
    metadata.to_csv(metadata_path, index=False)

    patient_class = pd.read_csv(args.patient_class_csv, header=None)
    patient_class = patient_class[patient_class[1].isin(args.labels)]
    patient_class.to_csv(dataset_dir / "patient_class.csv", header=False, index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "dataset_folder": dataset_name,
        "dataset_dir": str(dataset_dir),
        "metadata_csv": str(metadata_path),
        "total_parts": total_parts,
        "grid_shape": grid_shape(total_parts),
        "min_cells_per_tile": args.min_cells_per_tile,
        "max_distance": args.max_distance,
        "n_source_fovs": int(selected_fovs.shape[0]),
        "n_split_tiles_written": int(metadata.shape[0]),
        "n_empty_or_small_tiles_skipped": int(tile_sizes.shape[0] - metadata.shape[0]),
        "cell_type_to_id": type_to_id,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_cism_for_split(args: argparse.Namespace) -> dict:
    dataset_folder = f"Melanoma_NN_NP_split_{args.parts:02d}"
    dataset_root = args.output_root / "prepared_datasets"
    artifact_path = args.output_root / f"split_{args.parts:02d}_cism_ready.pkl"
    if artifact_path.exists() and not args.force:
        return {"artifact": str(artifact_path), "reused": True}

    cism = CISM(
        fanmod_exe=fanmod_executable_name(),
        fanmod_path=str(FANMOD_BINARY_DIR),
        network_dataset_root_path=str(dataset_root),
        fanmod_output_root_path=str(args.output_root / "fanmod_output"),
        fanmod_cache_root_path=str(args.output_root / "fanmod_cache"),
        motif_size=MOTIF_SIZE,
        iterations=FANMOD_ITERATIONS,
    )
    cism.add_dataset(
        dataset_folder=dataset_folder,
        dataset_type=DATASET_TYPE,
        dataset_name=DATASET_NAME,
        force_run_fanmod=args.force,
        force_parse=args.force,
        n_jobs=args.n_jobs,
        require_patient_class=True,
    )
    discriminator = TissueStateDiscriminativeMotifs(
        cism=cism,
        tissue_state_csv_path=str(dataset_root / dataset_folder / "patient_class.csv"),
        tissue_state_to_string=TISSUE_STATE_TO_STRING,
        common_cells_type=COMMON_CELLS_TYPE,
    )
    artifact = {
        "cism": cism,
        "discriminator": discriminator,
        "dataset_root": str(dataset_root),
        "dataset_folder": dataset_folder,
        "labels": args.labels,
        "grid_parts": args.parts,
    }
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return {"artifact": str(artifact_path), "reused": False, "motif_rows": int(len(cism.motifs_dataset))}


def load_artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def original_fov_from_split(fov: str) -> str:
    return str(fov).split("_p", 1)[0]


def summarize_split_artifact(args: argparse.Namespace, total_parts: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_artifact_path = args.output_root / f"split_{total_parts:02d}_cism_ready.pkl"
    if not split_artifact_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    original_artifact = load_artifact(args.original_artifact)
    split_artifact = load_artifact(split_artifact_path)
    original_motifs = original_artifact["cism"].motifs_dataset.copy()
    split_motifs = split_artifact["cism"].motifs_dataset.copy()
    selected_fovs = load_selected_fovs(args)
    selected_patients = set(selected_fovs["patient_uid"])

    original_motifs = original_motifs[original_motifs["Patient_uId"].isin(selected_patients)].copy()
    split_motifs["source_fov"] = split_motifs["FOV"].map(original_fov_from_split)

    original_sets = (
        original_motifs.drop_duplicates(["Patient_uId", "FOV", "ID"])
        .groupby(["Patient_uId", "FOV"], observed=True)["ID"]
        .agg(lambda values: set(values))
    )
    split_sets = (
        split_motifs.drop_duplicates(["Patient_uId", "source_fov", "ID"])
        .groupby(["Patient_uId", "source_fov"], observed=True)["ID"]
        .agg(lambda values: set(values))
    )

    aggregation_rows = []
    keys = sorted(set(original_sets.index).union(split_sets.index))
    for patient_uid, fov in keys:
        original_set = original_sets.get((patient_uid, fov), set())
        split_set = split_sets.get((patient_uid, fov), set())
        union = original_set | split_set
        intersection = original_set & split_set
        aggregation_rows.append(
            {
                "grid_parts": total_parts,
                "patient_uid": patient_uid,
                "source_fov": fov,
                "original_motif_count": len(original_set),
                "split_aggregated_motif_count": len(split_set),
                "shared_motif_count": len(intersection),
                "split_only_motif_count": len(split_set - original_set),
                "original_only_motif_count": len(original_set - split_set),
                "jaccard": len(intersection) / len(union) if union else np.nan,
            }
        )

    patient_classes = load_patient_classes(split_artifact["discriminator"], args.labels)
    cohort_patients = patient_classes.index.tolist()
    motif_summary = discover_hard_discriminative_motifs(
        motifs_df=split_motifs,
        patient_classes=patient_classes,
        cohort_patients=cohort_patients,
        p_value_threshold=args.p_value_threshold,
    )
    motif_summary.insert(0, "grid_parts", total_parts)

    performance_rows = []
    for stringency in args.stringency_grid:
        metrics = evaluate_hard_discriminative_loocv(
            motifs_df=split_motifs,
            patient_classes=patient_classes,
            cohort_patients=cohort_patients,
            sampled_label=args.labels[0],
            labels_to_compare=args.labels,
            stringency=float(stringency),
            max_features=args.max_features,
            p_value_threshold=args.p_value_threshold,
            random_state=args.random_state,
        )
        performance_rows.append({"grid_parts": total_parts, "stringency": float(stringency), **metrics})

    best, optuna_trials = tune_stringency(
        motifs_df=split_motifs,
        patient_classes=patient_classes,
        cohort_patients=cohort_patients,
        sampled_label=args.labels[0],
        labels_to_compare=args.labels,
        stringency_range=(0.0, 1.0),
        n_trials=args.n_optuna_trials,
        max_features=args.max_features,
        p_value_threshold=args.p_value_threshold,
        random_state=args.random_state,
    )
    performance_rows.append({"grid_parts": total_parts, "stringency": "optuna_best", **best})
    optuna_trials.insert(0, "grid_parts", total_parts)

    return pd.DataFrame(aggregation_rows), pd.DataFrame(performance_rows), motif_summary


def summarize_all(args: argparse.Namespace) -> dict:
    args.output_root.mkdir(parents=True, exist_ok=True)
    aggregation_frames = []
    performance_frames = []
    motif_frames = []
    missing = []
    for total_parts in args.parts:
        aggregation, performance, motifs = summarize_split_artifact(args, total_parts)
        if aggregation.empty and performance.empty and motifs.empty:
            missing.append(total_parts)
            continue
        aggregation_frames.append(aggregation)
        performance_frames.append(performance)
        motif_frames.append(motifs)

    if aggregation_frames:
        pd.concat(aggregation_frames, ignore_index=True).to_csv(
            args.output_root / "split_vs_original_fov_motif_aggregation.csv",
            index=False,
        )
    if performance_frames:
        pd.concat(performance_frames, ignore_index=True).to_csv(
            args.output_root / "split_classification_by_stringency.csv",
            index=False,
        )
    if motif_frames:
        pd.concat(motif_frames, ignore_index=True).to_csv(
            args.output_root / "split_discriminative_motifs.csv",
            index=False,
        )

    summary = {
        "parts_requested": args.parts,
        "missing_split_artifacts": missing,
        "outputs_written": {
            "aggregation": str(args.output_root / "split_vs_original_fov_motif_aggregation.csv"),
            "classification": str(args.output_root / "split_classification_by_stringency.csv"),
            "discriminative_motifs": str(args.output_root / "split_discriminative_motifs.csv"),
        },
    }
    (args.output_root / "split_fov_scale_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "prepare":
        summaries = [prepare_split_dataset(args, total_parts) for total_parts in args.parts]
        print(json.dumps(summaries, indent=2, sort_keys=True))
        print("\nNext CISM commands:")
        for summary in summaries:
            parts = int(summary["total_parts"])
            print(f"python analysis/spatial_scale/split_fov_scale_experiment.py run-cism --parts {parts}")
    elif args.stage == "run-cism":
        print(json.dumps(run_cism_for_split(args), indent=2, sort_keys=True))
    elif args.stage == "summarize":
        print(json.dumps(summarize_all(args), indent=2, sort_keys=True))
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
