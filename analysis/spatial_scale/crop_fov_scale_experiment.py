#!/usr/bin/env python3
"""Bootstrap rectangular FOV crops and analyze motif retention.

The default crop interpretation is area based: ``--crop-fractions 0.9 0.8``
means the crop rectangle covers 90% or 80% of the source FOV bounding-box area
while preserving the source aspect ratio. Use ``--fraction-mode linear`` if the
fraction should instead be applied to both width and height.
"""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import gc
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SCRIPT_DIR = ANALYSIS_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.cohort_imbalance.imbalanced_bootstrap_stringency import (  # noqa: E402
    discover_hard_discriminative_motifs,
    load_patient_classes,
    select_motifs_by_label,
)
from analysis.spatial_scale.split_fov_scale_experiment import (  # noqa: E402
    load_centroids_for_selected_fovs,
    load_selected_fovs,
    write_network_file,
)
from cism.cism import CISM, TissueStateDiscriminativeMotifs  # noqa: E402
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
    REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime" / "serialized" / "Melanoma_cism_ready.pkl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime" / "crop_fov_scale_experiment"
)
DEFAULT_CROP_FRACTIONS = [0.9, 0.8, 0.7, 0.6, 0.5]
LABELS = ["NN", "NP"]
NATIVE_REFERENCE_STRINGENCY = 0.46


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser("prepare", help="Create bootstrapped crop graph datasets.")
    add_common_args(prepare)
    prepare.add_argument("--crop-fractions", nargs="+", type=float, default=DEFAULT_CROP_FRACTIONS)
    prepare.add_argument("--n-bootstraps", type=int, default=10)
    prepare.add_argument("--fraction-mode", choices=["area", "linear"], default="area")
    prepare.add_argument("--min-cells-per-crop", type=int, default=4)
    prepare.add_argument("--chunksize", type=int, default=1_000_000)
    prepare.add_argument("--random-state", type=int, default=0)
    prepare.add_argument("--force", action="store_true")

    run_cism = subparsers.add_parser("run-cism", help="Run CISM/FANMOD for one crop fraction dataset.")
    add_common_args(run_cism)
    run_cism.add_argument("--crop-fraction", type=float, required=True)
    run_cism.add_argument("--n-jobs", type=int, default=8)
    run_cism.add_argument("--force", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Summarize crop motif artifacts.")
    add_common_args(summarize)
    summarize.add_argument("--crop-fractions", nargs="+", type=float, default=DEFAULT_CROP_FRACTIONS)
    summarize.add_argument("--retention-stringency", type=float, default=NATIVE_REFERENCE_STRINGENCY)
    summarize.add_argument("--max-features", type=int, default=30)
    summarize.add_argument("--p-value-threshold", type=float, default=0.05)
    summarize.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-centroids-csv", type=Path, default=DEFAULT_RAW_CENTROIDS_CSV)
    parser.add_argument("--scale-table", type=Path, default=DEFAULT_SCALE_TABLE)
    parser.add_argument("--patient-class-csv", type=Path, default=DEFAULT_PATIENT_CLASS_CSV)
    parser.add_argument("--original-artifact", type=Path, default=DEFAULT_ORIGINAL_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--patient-col", default="patient number")
    parser.add_argument("--fov-col", default="fov")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--x-col", default="centroid-0")
    parser.add_argument("--y-col", default="centroid-1")
    parser.add_argument("--cell-type-col", default="pred_mapped")
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE)


def fraction_token(crop_fraction: float) -> str:
    return f"{int(round(float(crop_fraction) * 100)):03d}"


def dataset_name(crop_fraction: float) -> str:
    return f"Melanoma_NN_NP_crop_{fraction_token(crop_fraction)}"


def artifact_path(output_root: Path, crop_fraction: float) -> Path:
    return output_root / f"crop_{fraction_token(crop_fraction)}_cism_ready.pkl"


def metadata_path(output_root: Path, crop_fraction: float) -> Path:
    return output_root / f"crop_{fraction_token(crop_fraction)}_metadata.csv"


def crop_side_fraction(crop_fraction: float, fraction_mode: str) -> float:
    if not 0 < crop_fraction <= 1:
        raise ValueError(f"Crop fraction must be in (0, 1], got {crop_fraction}")
    if fraction_mode == "area":
        return math.sqrt(crop_fraction)
    if fraction_mode == "linear":
        return crop_fraction
    raise ValueError(f"Unknown fraction mode: {fraction_mode}")


def sample_crop_bounds(
    group: pd.DataFrame,
    args: argparse.Namespace,
    side_fraction: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    x_min, x_max = float(group[args.x_col].min()), float(group[args.x_col].max())
    y_min, y_max = float(group[args.y_col].min()), float(group[args.y_col].max())
    x_span = max(x_max - x_min, np.finfo(float).eps)
    y_span = max(y_max - y_min, np.finfo(float).eps)
    crop_width = x_span * side_fraction
    crop_height = y_span * side_fraction
    left = x_min if crop_width >= x_span else rng.uniform(x_min, x_max - crop_width)
    top = y_min if crop_height >= y_span else rng.uniform(y_min, y_max - crop_height)
    return left, left + crop_width, top, top + crop_height


def prepare_crop_dataset(args: argparse.Namespace, crop_fraction: float) -> dict[str, Any]:
    output_root = args.output_root
    prepared_root = output_root / "prepared_datasets"
    crop_dataset_name = dataset_name(crop_fraction)
    dataset_dir = prepared_root / crop_dataset_name
    metadata_csv = metadata_path(output_root, crop_fraction)
    summary_path = output_root / f"crop_{fraction_token(crop_fraction)}_preparation_summary.json"
    if dataset_dir.exists() and any(dataset_dir.glob("Patient_*_FOV*.txt")) and summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    selected_fovs = load_selected_fovs(args)
    cells = load_centroids_for_selected_fovs(args, selected_fovs)
    type_to_id = {label: int(type_id) for type_id, label in COMMON_CELLS_TYPE.items()}
    missing_types = sorted(set(cells[args.cell_type_col].astype(str)) - set(type_to_id))
    if missing_types:
        raise ValueError(f"Crop cells contain labels absent from COMMON_CELLS_TYPE: {missing_types}")

    side_fraction = crop_side_fraction(crop_fraction, args.fraction_mode)
    rng = np.random.default_rng(args.random_state + int(round(crop_fraction * 10_000)))
    rows = []
    skipped = 0
    grouped = cells.groupby([args.patient_col, args.fov_col], observed=True)
    for (patient, source_fov), group in tqdm(
        grouped,
        desc=f"Writing crop {crop_fraction:.2f} graph files",
        unit="source FOV",
    ):
        for bootstrap_index in range(args.n_bootstraps):
            left, right, top, bottom = sample_crop_bounds(group, args, side_fraction, rng)
            crop = group[
                group[args.x_col].between(left, right, inclusive="both")
                & group[args.y_col].between(top, bottom, inclusive="both")
            ].copy()
            if len(crop) < args.min_cells_per_crop:
                skipped += 1
                continue
            crop_fov = f"{source_fov}_crop{fraction_token(crop_fraction)}_b{bootstrap_index:02d}"
            output_file = dataset_dir / f"Patient_{patient}_FOV{crop_fov}.txt"
            edge_count = write_network_file(crop, args, type_to_id, output_file)
            rows.append(
                {
                    "patient_number": int(patient),
                    "patient_uid": f"{DATASET_NAME}{patient}",
                    "source_fov": source_fov,
                    "crop_fov": crop_fov,
                    "crop_fraction": float(crop_fraction),
                    "fraction_mode": args.fraction_mode,
                    "side_fraction": float(side_fraction),
                    "bootstrap_index": int(bootstrap_index),
                    "crop_left": float(left),
                    "crop_right": float(right),
                    "crop_top": float(top),
                    "crop_bottom": float(bottom),
                    "cells": int(len(crop)),
                    "edges": int(edge_count),
                    "file": output_file.name,
                }
            )

    metadata = pd.DataFrame(rows)
    patient_class_lookup = selected_fovs.drop_duplicates("patient_uid").set_index("patient_uid")["patient_class"]
    metadata["patient_class"] = metadata["patient_uid"].map(patient_class_lookup)
    metadata.to_csv(metadata_csv, index=False)

    patient_class = pd.read_csv(args.patient_class_csv, header=None)
    patient_class = patient_class[patient_class[1].isin(args.labels)]
    patient_class.to_csv(dataset_dir / "patient_class.csv", header=False, index=False)

    summary = {
        "dataset_root": str(prepared_root),
        "dataset_folder": crop_dataset_name,
        "dataset_dir": str(dataset_dir),
        "metadata_csv": str(metadata_csv),
        "crop_fraction": float(crop_fraction),
        "fraction_mode": args.fraction_mode,
        "side_fraction": float(side_fraction),
        "n_bootstraps": int(args.n_bootstraps),
        "min_cells_per_crop": int(args.min_cells_per_crop),
        "n_source_fovs": int(selected_fovs.shape[0]),
        "n_crop_graphs_written": int(metadata.shape[0]),
        "n_crops_skipped": int(skipped),
        "cell_type_to_id": type_to_id,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_cism_for_crop(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root
    prepared_root = output_root / "prepared_datasets"
    crop_dataset_name = dataset_name(args.crop_fraction)
    out_path = artifact_path(output_root, args.crop_fraction)
    if out_path.exists() and not args.force:
        return {"artifact": str(out_path), "reused": True}

    cism = CISM(
        fanmod_exe=fanmod_executable_name(),
        fanmod_path=str(FANMOD_BINARY_DIR),
        network_dataset_root_path=str(prepared_root),
        fanmod_output_root_path=str(output_root / "fanmod_output"),
        fanmod_cache_root_path=str(output_root / "fanmod_cache"),
        motif_size=MOTIF_SIZE,
        iterations=FANMOD_ITERATIONS,
    )
    cism.add_dataset(
        dataset_folder=crop_dataset_name,
        dataset_type=DATASET_TYPE,
        dataset_name=DATASET_NAME,
        force_run_fanmod=args.force,
        force_parse=args.force,
        n_jobs=args.n_jobs,
        require_patient_class=True,
    )
    discriminator = TissueStateDiscriminativeMotifs(
        cism=cism,
        tissue_state_csv_path=str(prepared_root / crop_dataset_name / "patient_class.csv"),
        tissue_state_to_string=TISSUE_STATE_TO_STRING,
        common_cells_type=COMMON_CELLS_TYPE,
    )
    artifact = {
        "cism": cism,
        "discriminator": discriminator,
        "dataset_root": str(prepared_root),
        "dataset_folder": crop_dataset_name,
        "labels": args.labels,
        "crop_fraction": args.crop_fraction,
    }
    with out_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return {"artifact": str(out_path), "reused": False, "motif_rows": int(len(cism.motifs_dataset))}


def load_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def source_fov_from_crop(fov: str) -> str:
    return str(fov).split("_crop", 1)[0]


def bootstrap_index_from_crop(fov: str) -> int:
    return int(str(fov).rsplit("_b", 1)[1])


def fold_union_selected_motifs(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    cohort_patients: list[str],
    *,
    stringency: float,
    max_features: int,
    p_value_threshold: float,
) -> dict[str, set[int]]:
    union_by_label = {label: set() for label in LABELS}
    for held_out in cohort_patients:
        train = [patient for patient in cohort_patients if patient != held_out]
        summary = discover_hard_discriminative_motifs(
            motifs_df=motifs_df,
            patient_classes=patient_classes,
            cohort_patients=train,
            p_value_threshold=p_value_threshold,
        )
        selected = select_motifs_by_label(summary, LABELS, stringency, max_features)
        for label, ids in selected.items():
            union_by_label[label].update(map(int, ids))
    return union_by_label


def motif_set(union_by_label: dict[str, set[int]]) -> set[int]:
    return set().union(*union_by_label.values())


def retention_row(
    *,
    crop_fraction: float,
    scenario: str,
    bootstrap_index: int | None,
    selected_by_label: dict[str, set[int]],
    reference: set[int],
) -> dict[str, Any]:
    selected = motif_set(selected_by_label)
    shared = selected & reference
    union = selected | reference
    return {
        "crop_fraction": float(crop_fraction),
        "scenario": scenario,
        "bootstrap_index": bootstrap_index,
        "reference_motif_count": len(reference),
        "selected_motif_count": len(selected),
        "selected_NN": len(selected_by_label["NN"]),
        "selected_NP": len(selected_by_label["NP"]),
        "shared_with_reference": len(shared),
        "reference_recovered_fraction": len(shared) / len(reference) if reference else np.nan,
        "selected_from_reference_fraction": len(shared) / len(selected) if selected else np.nan,
        "jaccard": len(shared) / len(union) if union else np.nan,
    }


def summarize_crop_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    original = load_artifact(args.original_artifact)
    patient_classes = load_patient_classes(original["discriminator"], LABELS)
    cohort_patients = patient_classes.index.tolist()
    reference_by_label = fold_union_selected_motifs(
        original["cism"].motifs_dataset.copy(),
        patient_classes,
        cohort_patients,
        stringency=NATIVE_REFERENCE_STRINGENCY,
        max_features=args.max_features,
        p_value_threshold=args.p_value_threshold,
    )
    reference = motif_set(reference_by_label)

    motif_count_rows = []
    retention_rows = []
    missing = []
    for crop_fraction in args.crop_fractions:
        path = artifact_path(output_root, crop_fraction)
        if not path.exists():
            missing.append(float(crop_fraction))
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing crop artifact: {path}")
        print(f"Summarizing crop fraction {crop_fraction:.2f}", flush=True)
        artifact = load_artifact(path)
        motifs = artifact["cism"].motifs_dataset.copy()
        motifs["source_fov"] = motifs["FOV"].map(source_fov_from_crop)
        motifs["bootstrap_index"] = motifs["FOV"].map(bootstrap_index_from_crop)

        crop_counts = (
            motifs.drop_duplicates(["Patient_uId", "FOV", "ID"])
            .groupby(["Patient_uId", "FOV", "bootstrap_index"], observed=True)["ID"]
            .nunique()
            .reset_index(name="enriched_motif_count")
        )
        crop_counts["crop_fraction"] = float(crop_fraction)
        motif_count_rows.append(crop_counts)

        selected_all = fold_union_selected_motifs(
            motifs,
            patient_classes,
            cohort_patients,
            stringency=args.retention_stringency,
            max_features=args.max_features,
            p_value_threshold=args.p_value_threshold,
        )
        retention_rows.append(
            retention_row(
                crop_fraction=crop_fraction,
                scenario="all_bootstrap_crops",
                bootstrap_index=None,
                selected_by_label=selected_all,
                reference=reference,
            )
        )

        for bootstrap_index, boot_motifs in motifs.groupby("bootstrap_index", observed=True):
            selected_bootstrap = fold_union_selected_motifs(
                boot_motifs.copy(),
                patient_classes,
                cohort_patients,
                stringency=args.retention_stringency,
                max_features=args.max_features,
                p_value_threshold=args.p_value_threshold,
            )
            retention_rows.append(
                retention_row(
                    crop_fraction=crop_fraction,
                    scenario="one_crop_per_source_fov",
                    bootstrap_index=int(bootstrap_index),
                    selected_by_label=selected_bootstrap,
                    reference=reference,
                )
            )
        del motifs, artifact
        gc.collect()

    motif_counts = pd.concat(motif_count_rows, ignore_index=True) if motif_count_rows else pd.DataFrame()
    retention = pd.DataFrame(retention_rows)
    if not motif_counts.empty:
        motif_counts.to_csv(output_root / "crop_enriched_motif_counts_per_crop.csv", index=False)
        motif_summary = (
            motif_counts.groupby("crop_fraction", observed=True)["enriched_motif_count"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )
        motif_summary.to_csv(output_root / "crop_enriched_motif_count_summary.csv", index=False)
    else:
        motif_summary = pd.DataFrame()

    retention.to_csv(output_root / "crop_motif_retention.csv", index=False)
    retention_summary = (
        retention.groupby(["crop_fraction", "scenario"], observed=True)
        .agg(
            repeats=("bootstrap_index", lambda s: int(s.notna().sum())),
            mean_selected_motif_count=("selected_motif_count", "mean"),
            mean_shared_with_reference=("shared_with_reference", "mean"),
            mean_reference_recovered_fraction=("reference_recovered_fraction", "mean"),
            sd_reference_recovered_fraction=("reference_recovered_fraction", "std"),
            mean_jaccard=("jaccard", "mean"),
            sd_jaccard=("jaccard", "std"),
        )
        .reset_index()
    )
    retention_summary.to_csv(output_root / "crop_motif_retention_summary.csv", index=False)
    plot_summaries(output_root, motif_summary, retention_summary)
    report_path = write_report(output_root, motif_summary, retention_summary, missing)
    summary = {
        "output_root": str(output_root),
        "missing_crop_artifacts": missing,
        "native_reference_motifs": len(reference),
        "native_reference_NN": len(reference_by_label["NN"]),
        "native_reference_NP": len(reference_by_label["NP"]),
        "retention_stringency": args.retention_stringency,
        "report": str(report_path),
    }
    (output_root / "crop_fov_scale_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_summaries(output_root: Path, motif_summary: pd.DataFrame, retention_summary: pd.DataFrame) -> None:
    if not motif_summary.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.errorbar(
            motif_summary["crop_fraction"],
            motif_summary["mean"],
            yerr=motif_summary["std"],
            marker="o",
            capsize=4,
        )
        ax.set_xlabel("Crop area fraction")
        ax.set_ylabel("Enriched motifs per crop")
        ax.set_title("Motifs arising from cropped tissue")
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(output_root / "crop_enriched_motif_count_summary.png", dpi=200)
        plt.close(fig)

    if not retention_summary.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for scenario, group in retention_summary.groupby("scenario", observed=True):
            ax.errorbar(
                group["crop_fraction"],
                group["mean_reference_recovered_fraction"],
                yerr=group["sd_reference_recovered_fraction"].fillna(0),
                marker="o",
                capsize=4,
                label=scenario,
            )
        ax.set_xlabel("Crop area fraction")
        ax.set_ylabel("Fraction of native 19 motifs retained")
        ax.set_ylim(0, 1)
        ax.invert_xaxis()
        ax.set_title("Motif retention under bootstrapped crops")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_root / "crop_motif_retention_summary.png", dpi=200)
        plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy().fillna("")
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(
    output_root: Path,
    motif_summary: pd.DataFrame,
    retention_summary: pd.DataFrame,
    missing: list[float],
) -> Path:
    report = output_root / "crop_fov_scale_report.md"
    lines = [
        "# Bootstrapped Crop-FOV Scale Analysis",
        "",
        "Native reference: 19 LOOCV fold-union motifs at stringency 0.46.",
        "",
        f"Missing crop artifacts: {missing if missing else 'none'}",
        "",
        "## Enriched Motifs Per Crop",
        "",
        md_table(motif_summary),
        "",
        "![motif count summary](crop_enriched_motif_count_summary.png)",
        "",
        "## Motif Retention",
        "",
        md_table(retention_summary),
        "",
        "![retention summary](crop_motif_retention_summary.png)",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        summaries = [prepare_crop_dataset(args, fraction) for fraction in args.crop_fractions]
        print(json.dumps(summaries, indent=2, sort_keys=True))
        print("\nNext CISM commands:")
        for fraction in args.crop_fractions:
            print(f"python analysis/spatial_scale/crop_fov_scale_experiment.py run-cism --crop-fraction {fraction}")
    elif args.stage == "run-cism":
        print(json.dumps(run_cism_for_crop(args), indent=2, sort_keys=True))
    elif args.stage == "summarize":
        print(json.dumps(summarize_crop_artifacts(args), indent=2, sort_keys=True))
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
