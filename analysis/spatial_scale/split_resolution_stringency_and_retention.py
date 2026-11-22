#!/usr/bin/env python3
"""Analyze split-FOV stringency tuning and motif retention under subsampling.

This script answers two related questions for the melanoma NN/NP split-FOV
experiment:

1. How does the optimal stringency parameter change with split resolution?
2. How many of the native fold-union motifs are retained when not all split
   tiles are used?

The native reference is the 19-motif LOOCV fold union at stringency 0.46.
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
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.cohort_imbalance.imbalanced_bootstrap_stringency import (  # noqa: E402
    discover_hard_discriminative_motifs,
    evaluate_hard_discriminative_loocv,
    load_patient_classes,
    select_motifs_by_label,
)


RUNTIME = REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
ORIGINAL_ARTIFACT = RUNTIME / "serialized" / "Melanoma_cism_ready.pkl"
SPLIT_DIR = RUNTIME / "split_fov_scale_experiment"
DEFAULT_OUTPUT_DIR = SPLIT_DIR / "stringency_vs_split_resolution"

LABELS = ["NN", "NP"]
DEFAULT_PARTS = [2, 4, 8, 16]
DEFAULT_STRINGENCY_GRID = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
NATIVE_REFERENCE_STRINGENCY = 0.46


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parts", nargs="+", type=int, default=DEFAULT_PARTS)
    parser.add_argument("--stages", nargs="+", default=["optimal-stringency", "retention"])
    parser.add_argument("--stringency-grid", nargs="+", type=float, default=DEFAULT_STRINGENCY_GRID)
    parser.add_argument("--retention-stringency", type=float, default=NATIVE_REFERENCE_STRINGENCY)
    parser.add_argument("--max-features", type=int, default=30)
    parser.add_argument("--p-value-threshold", type=float, default=0.05)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--sample-tiles-per-source", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing output CSVs where possible.",
    )
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_motifs_from_artifact(path: Path) -> pd.DataFrame:
    return load_artifact(path)["cism"].motifs_dataset.copy()


def split_artifact_path(parts: int) -> Path:
    return SPLIT_DIR / f"split_{parts:02d}_cism_ready.pkl"


def split_metadata_path(parts: int) -> Path:
    return SPLIT_DIR / f"split_{parts:02d}_metadata.csv"


def fold_union_selected_motifs(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    cohort_patients: list[str],
    *,
    stringency: float,
    max_features: int,
    p_value_threshold: float,
) -> tuple[dict[str, set[int]], pd.DataFrame]:
    """Return union of selected motif IDs across LOOCV folds."""
    union_by_label = {label: set() for label in LABELS}
    rows = []
    for held_out_patient in cohort_patients:
        train_patients = [patient for patient in cohort_patients if patient != held_out_patient]
        summary = discover_hard_discriminative_motifs(
            motifs_df=motifs_df,
            patient_classes=patient_classes,
            cohort_patients=train_patients,
            p_value_threshold=p_value_threshold,
        )
        selected = select_motifs_by_label(
            discriminative_summary=summary,
            labels=LABELS,
            stringency=stringency,
            max_features=max_features,
        )
        for label, ids in selected.items():
            for motif_id in map(int, ids):
                union_by_label[label].add(motif_id)
                rows.append(
                    {
                        "held_out_patient": held_out_patient,
                        "motif_label": label,
                        "ID": motif_id,
                    }
                )
    return union_by_label, pd.DataFrame(rows)


def motif_set(union_by_label: dict[str, set[int]]) -> set[int]:
    return set().union(*union_by_label.values())


def stability_row(
    *,
    scenario: str,
    dataset: str,
    grid_parts: int | str,
    repeat: int | None,
    stringency: float,
    reference: set[int],
    selected_by_label: dict[str, set[int]],
    selected_fold_rows: pd.DataFrame,
) -> dict[str, Any]:
    selected = motif_set(selected_by_label)
    shared = selected & reference
    union = selected | reference
    return {
        "scenario": scenario,
        "dataset": dataset,
        "grid_parts": grid_parts,
        "repeat": repeat,
        "stringency": stringency,
        "reference_motif_count": len(reference),
        "selected_motif_count": len(selected),
        "selected_NN": len(selected_by_label["NN"]),
        "selected_NP": len(selected_by_label["NP"]),
        "shared_with_reference": len(shared),
        "reference_recovered_fraction": len(shared) / len(reference) if reference else np.nan,
        "selected_from_reference_fraction": len(shared) / len(selected) if selected else np.nan,
        "jaccard": len(shared) / len(union) if union else np.nan,
        "mean_selected_per_fold": (
            selected_fold_rows.groupby("held_out_patient", observed=True)["ID"].nunique().mean()
            if not selected_fold_rows.empty
            else 0.0
        ),
    }


def build_native_reference(
    original_artifact: dict[str, Any],
    patient_classes: pd.DataFrame,
    cohort_patients: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, set[int]], pd.DataFrame]:
    return fold_union_selected_motifs(
        original_artifact["cism"].motifs_dataset.copy(),
        patient_classes,
        cohort_patients,
        stringency=NATIVE_REFERENCE_STRINGENCY,
        max_features=args.max_features,
        p_value_threshold=args.p_value_threshold,
    )


def run_optimal_stringency(
    original_artifact: dict[str, Any],
    patient_classes: pd.DataFrame,
    cohort_patients: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    dataset_specs: list[tuple[str, int | str, Path | None]] = []
    if args.include_native:
        dataset_specs.append(("native", "native", None))
    for parts in args.parts:
        dataset_specs.append((f"split_{parts:02d}", parts, split_artifact_path(parts)))

    for dataset_name, grid_parts, artifact_path in dataset_specs:
        if artifact_path is None:
            print("[optimal-stringency] Loading native artifact motifs", flush=True)
            motifs_df = original_artifact["cism"].motifs_dataset.copy()
        else:
            print(f"[optimal-stringency] Loading {dataset_name} artifact", flush=True)
            motifs_df = load_motifs_from_artifact(artifact_path)
        print(f"[optimal-stringency] Evaluating {dataset_name}", flush=True)
        for stringency in args.stringency_grid:
            metrics = evaluate_hard_discriminative_loocv(
                motifs_df=motifs_df,
                patient_classes=patient_classes,
                cohort_patients=cohort_patients,
                sampled_label=LABELS[0],
                labels_to_compare=LABELS,
                stringency=float(stringency),
                max_features=args.max_features,
                p_value_threshold=args.p_value_threshold,
                random_state=args.random_state,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "grid_parts": grid_parts,
                    "stringency": float(stringency),
                    **metrics,
                }
            )
        del motifs_df
        gc.collect()

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "stringency_grid_by_split_resolution.csv", index=False)

    best = (
        result.dropna(subset=["roc_auc"])
        .sort_values(
            ["dataset", "roc_auc", "balanced_accuracy", "mean_selected_motifs"],
            ascending=[True, False, False, False],
        )
        .groupby("dataset", observed=True)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(args.output_dir / "optimal_stringency_by_split_resolution.csv", index=False)
    plot_optimal_stringency(result, best, args.output_dir)
    return best


def random_tile_subset(
    split_motifs: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_tiles_per_source: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    chosen = []
    for (_patient_uid, _source_fov), group in metadata.groupby(["patient_uid", "source_fov"], observed=True):
        options = group["split_fov"].astype(str).to_numpy()
        size = min(sample_tiles_per_source, len(options))
        chosen.extend(rng.choice(options, size=size, replace=False).tolist())
    chosen_set = set(chosen)
    return split_motifs[split_motifs["FOV"].astype(str).isin(chosen_set)].copy()


def run_retention(
    original_artifact: dict[str, Any],
    patient_classes: pd.DataFrame,
    cohort_patients: list[str],
    native_reference: dict[str, set[int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    reference = motif_set(native_reference)
    rows = []
    fold_rows = []

    native_reference_table = pd.DataFrame(
        [
            {"motif_label": label, "ID": motif_id}
            for label, ids in native_reference.items()
            for motif_id in sorted(ids)
        ]
    )
    native_reference_table.to_csv(args.output_dir / "native_reference_fold_union_motifs_stringency_046.csv", index=False)

    for parts in args.parts:
        print(f"[retention] Loading split {parts} artifact", flush=True)
        split_motifs = load_motifs_from_artifact(split_artifact_path(parts))
        selected_by_label, selected_rows = fold_union_selected_motifs(
            split_motifs,
            patient_classes,
            cohort_patients,
            stringency=args.retention_stringency,
            max_features=args.max_features,
            p_value_threshold=args.p_value_threshold,
        )
        selected_rows.insert(0, "scenario", "all_split_tiles")
        selected_rows.insert(1, "dataset", f"split_{parts:02d}")
        selected_rows.insert(2, "grid_parts", parts)
        fold_rows.append(selected_rows)
        rows.append(
            stability_row(
                scenario="all_split_tiles",
                dataset=f"split_{parts:02d}",
                grid_parts=parts,
                repeat=None,
                stringency=args.retention_stringency,
                reference=reference,
                selected_by_label=selected_by_label,
                selected_fold_rows=selected_rows,
            )
        )

        metadata = pd.read_csv(split_metadata_path(parts))
        rng = np.random.default_rng(args.random_state + parts)
        for repeat in range(args.random_repeats):
            if repeat % 5 == 0:
                print(f"[retention] split {parts} random repeat {repeat}/{args.random_repeats}", flush=True)
            sampled_motifs = random_tile_subset(
                split_motifs,
                metadata,
                sample_tiles_per_source=args.sample_tiles_per_source,
                rng=rng,
            )
            selected_by_label, selected_rows = fold_union_selected_motifs(
                sampled_motifs,
                patient_classes,
                cohort_patients,
                stringency=args.retention_stringency,
                max_features=args.max_features,
                p_value_threshold=args.p_value_threshold,
            )
            selected_rows.insert(0, "scenario", f"random_{args.sample_tiles_per_source}_tile_per_source")
            selected_rows.insert(1, "dataset", f"split_{parts:02d}")
            selected_rows.insert(2, "grid_parts", parts)
            selected_rows.insert(3, "repeat", repeat)
            fold_rows.append(selected_rows)
            rows.append(
                stability_row(
                    scenario=f"random_{args.sample_tiles_per_source}_tile_per_source",
                    dataset=f"split_{parts:02d}",
                    grid_parts=parts,
                    repeat=repeat,
                    stringency=args.retention_stringency,
                    reference=reference,
                    selected_by_label=selected_by_label,
                    selected_fold_rows=selected_rows,
                )
            )
        del split_motifs
        gc.collect()

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "motif_retention_by_split_subsampling.csv", index=False)
    if fold_rows:
        pd.concat(fold_rows, ignore_index=True).to_csv(
            args.output_dir / "motif_retention_selected_motifs_by_fold.csv",
            index=False,
        )
    plot_retention(result, args.output_dir)
    return result


def plot_optimal_stringency(grid: pd.DataFrame, best: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, group in grid.groupby("dataset", observed=True):
        ax.plot(group["stringency"], group["roc_auc"], marker="o", linewidth=1.5, label=str(dataset))
    ax.set_xlabel("Stringency")
    ax.set_ylabel("LOOCV ROC AUC")
    ax.set_title("Stringency sweep by split resolution")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "stringency_sweep_auc_by_split_resolution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_data = best.copy()
    plot_data["x"] = plot_data["grid_parts"].astype(str)
    ax.bar(plot_data["x"], plot_data["stringency"], color="#4E79A7")
    ax.set_xlabel("Split resolution")
    ax.set_ylabel("Optimal stringency")
    ax.set_ylim(0, 1)
    ax.set_title("Optimal stringency vs split resolution")
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_stringency_vs_split_resolution.png", dpi=200)
    plt.close(fig)


def plot_retention(retention: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_tiles = retention[retention["scenario"].eq("all_split_tiles")]
    ax.plot(
        all_tiles["grid_parts"].astype(int),
        all_tiles["reference_recovered_fraction"],
        marker="o",
        label="all split tiles",
    )
    random = retention[retention["scenario"].str.startswith("random_")].copy()
    if not random.empty:
        summary = (
            random.groupby("grid_parts", observed=True)["reference_recovered_fraction"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            summary["grid_parts"].astype(int),
            summary["mean"],
            yerr=summary["std"],
            marker="o",
            capsize=4,
            label="random subset",
        )
    ax.set_xlabel("Split resolution")
    ax.set_ylabel("Fraction of native 19 motifs retained")
    ax.set_ylim(0, 1)
    ax.set_title("Native motif retention under split subsampling")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "motif_retention_by_split_subsampling.png", dpi=200)
    plt.close(fig)


def write_report(args: argparse.Namespace, best: pd.DataFrame | None, retention: pd.DataFrame | None) -> Path:
    def md_table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_Not run._"
        view = df.copy().fillna("")
        header = "| " + " | ".join(map(str, view.columns)) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
        return "\n".join([header, sep, *rows])

    retention_summary = None
    if retention is not None and not retention.empty:
        retention_summary = (
            retention.groupby(["scenario", "grid_parts"], observed=True)
            .agg(
                repeats=("repeat", lambda s: int(s.notna().sum())),
                mean_selected_motif_count=("selected_motif_count", "mean"),
                mean_shared_with_reference=("shared_with_reference", "mean"),
                mean_reference_recovered_fraction=("reference_recovered_fraction", "mean"),
                sd_reference_recovered_fraction=("reference_recovered_fraction", "std"),
                mean_jaccard=("jaccard", "mean"),
                sd_jaccard=("jaccard", "std"),
            )
            .reset_index()
        )
        retention_summary.to_csv(args.output_dir / "motif_retention_summary.csv", index=False)

    report = args.output_dir / "split_resolution_stringency_and_retention_report.md"
    lines = [
        "# Split Resolution Stringency And Motif Retention",
        "",
        f"Native reference: LOOCV fold-union motifs at stringency {NATIVE_REFERENCE_STRINGENCY}.",
        f"Retention stringency: {args.retention_stringency}.",
        "",
        "## Optimal Stringency Vs Split Resolution",
        "",
        md_table(best),
        "",
        "## Motif Retention When Not Taking All Split Tiles",
        "",
        md_table(retention_summary),
        "",
        f"![stringency sweep]({(args.output_dir / 'stringency_sweep_auc_by_split_resolution.png').resolve()})",
        "",
        f"![optimal stringency]({(args.output_dir / 'optimal_stringency_vs_split_resolution.png').resolve()})",
        "",
        f"![retention]({(args.output_dir / 'motif_retention_by_split_subsampling.png').resolve()})",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_artifact = load_artifact(ORIGINAL_ARTIFACT)
    patient_classes = load_patient_classes(original_artifact["discriminator"], LABELS)
    cohort_patients = patient_classes.index.tolist()

    native_reference, native_rows = build_native_reference(
        original_artifact,
        patient_classes,
        cohort_patients,
        args,
    )
    native_rows.to_csv(args.output_dir / "native_reference_selected_motifs_by_fold.csv", index=False)

    best = None
    retention = None
    stages = set(args.stages)
    if "optimal-stringency" in stages:
        best_path = args.output_dir / "optimal_stringency_by_split_resolution.csv"
        if args.skip_existing and best_path.exists():
            best = pd.read_csv(best_path)
        else:
            best = run_optimal_stringency(original_artifact, patient_classes, cohort_patients, args)
    if "retention" in stages:
        retention_path = args.output_dir / "motif_retention_by_split_subsampling.csv"
        if args.skip_existing and retention_path.exists():
            retention = pd.read_csv(retention_path)
        else:
            retention = run_retention(original_artifact, patient_classes, cohort_patients, native_reference, args)

    report = write_report(args, best, retention)
    summary = {
        "output_dir": str(args.output_dir),
        "report": str(report),
        "stages": sorted(stages),
        "parts": args.parts,
        "native_reference_motifs": len(motif_set(native_reference)),
        "native_reference_NN": len(native_reference["NN"]),
        "native_reference_NP": len(native_reference["NP"]),
        "retention_stringency": args.retention_stringency,
        "random_repeats": args.random_repeats,
        "sample_tiles_per_source": args.sample_tiles_per_source,
    }
    (args.output_dir / "split_resolution_stringency_and_retention_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
