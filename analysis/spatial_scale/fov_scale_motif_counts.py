#!/usr/bin/env python3
"""Compare significant motif counts across native melanoma FOV spatial scales."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SCALE_TABLE = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "fov_spatial_scale_nn_np"
    / "fov_spatial_scale_per_fov.csv"
)
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
DEFAULT_ABSENCE_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "Melanoma_NN_NP_absent_cism_ready.pkl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime" / "fov_scale_motif_counts_nn_np"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-table", type=Path, default=DEFAULT_SCALE_TABLE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--absence-artifact", type=Path, default=DEFAULT_ABSENCE_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs="+", default=["NN", "NP"])
    parser.add_argument("--small-scale-threshold", type=float, default=0.75)
    return parser.parse_args()


def load_motif_table(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    return artifact["cism"].motifs_dataset.copy()


def count_unique_motifs(motifs: pd.DataFrame, value_name: str) -> pd.DataFrame:
    required = {"Patient_uId", "FOV", "ID"}
    missing = required.difference(motifs.columns)
    if missing:
        raise ValueError(f"Motif table is missing required columns: {sorted(missing)}")

    return (
        motifs.dropna(subset=["Patient_uId", "FOV", "ID"])
        .drop_duplicates(["Patient_uId", "FOV", "ID"])
        .groupby(["Patient_uId", "FOV"], observed=True)
        .agg(**{value_name: ("ID", "count")})
        .reset_index()
    )


def add_scale_labels(fov_scale: pd.DataFrame, small_scale_threshold: float) -> pd.DataFrame:
    fov_scale = fov_scale.copy()
    fov_scale["linear_scale"] = np.sqrt(fov_scale["size"])
    full_linear_scale = fov_scale["linear_scale"].max()
    fov_scale["relative_linear_scale"] = fov_scale["linear_scale"] / full_linear_scale
    fov_scale["scale_label"] = np.where(
        fov_scale["relative_linear_scale"] < small_scale_threshold,
        "small",
        "full",
    )
    return fov_scale


def build_per_fov_counts(args: argparse.Namespace) -> pd.DataFrame:
    fov_scale = pd.read_csv(args.scale_table)
    fov_scale = fov_scale[fov_scale["patient_class"].isin(args.labels)].copy()
    if "fov" not in fov_scale.columns:
        fov_scale["fov"] = "FOV" + fov_scale["fov_number"].astype(int).astype(str)
    fov_scale = add_scale_labels(fov_scale, args.small_scale_threshold)

    motif_counts = count_unique_motifs(load_motif_table(args.artifact), "significant_motif_count")
    merged = fov_scale.merge(
        motif_counts,
        left_on=["patient_uid", "fov"],
        right_on=["Patient_uId", "FOV"],
        how="left",
    ).drop(columns=["Patient_uId", "FOV"])
    merged["significant_motif_count"] = merged["significant_motif_count"].fillna(0).astype(int)

    if args.absence_artifact.exists():
        absence_counts = count_unique_motifs(
            load_motif_table(args.absence_artifact),
            "absence_significant_motif_count",
        )
        merged = merged.merge(
            absence_counts,
            left_on=["patient_uid", "fov"],
            right_on=["Patient_uId", "FOV"],
            how="left",
        ).drop(columns=["Patient_uId", "FOV"])
        merged["absence_significant_motif_count"] = (
            merged["absence_significant_motif_count"].fillna(0).astype(int)
        )

    return merged.sort_values(["patient_class", "patient_number", "fov_number"])


def build_same_patient_pairs(per_fov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "significant_motif_count",
        *(
            ["absence_significant_motif_count"]
            if "absence_significant_motif_count" in per_fov.columns
            else []
        ),
    ]
    for patient_uid, patient_fovs in per_fov.groupby("patient_uid", observed=True):
        scale_labels = set(patient_fovs["scale_label"])
        if not {"small", "full"}.issubset(scale_labels):
            continue
        small_rows = patient_fovs[patient_fovs["scale_label"] == "small"]
        full_rows = patient_fovs[patient_fovs["scale_label"] == "full"]
        for _, small in small_rows.iterrows():
            for _, full in full_rows.iterrows():
                row = {
                    "patient_uid": patient_uid,
                    "patient_class": small["patient_class"],
                    "small_fov": small["fov"],
                    "full_fov": full["fov"],
                    "small_width": small["width"],
                    "small_height": small["height"],
                    "small_size": small["size"],
                    "full_width": full["width"],
                    "full_height": full["height"],
                    "full_size": full["size"],
                    "linear_scale_ratio": small["linear_scale"] / full["linear_scale"],
                }
                for metric in metric_cols:
                    row[f"small_{metric}"] = small[metric]
                    row[f"full_{metric}"] = full[metric]
                    row[f"small_minus_full_{metric}"] = small[metric] - full[metric]
                    row[f"small_over_full_{metric}"] = (
                        small[metric] / full[metric] if full[metric] else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["patient_class", "patient_uid", "small_fov", "full_fov"])


def summarize(per_fov: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "significant_motif_count",
        *(
            ["absence_significant_motif_count"]
            if "absence_significant_motif_count" in per_fov.columns
            else []
        ),
    ]
    summary = (
        per_fov.groupby(["patient_class", "scale_label"], observed=True)
        .agg(
            n_fovs=("graph_file", "count"),
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
            mean_size=("size", "mean"),
            **{f"mean_{metric}": (metric, "mean") for metric in metric_cols},
            **{f"std_{metric}": (metric, "std") for metric in metric_cols},
        )
        .reset_index()
    )
    if pairs.empty:
        pair_summary = pd.DataFrame()
    else:
        pair_summary = pairs.agg(
            {
                "linear_scale_ratio": ["count", "mean", "std"],
                **{
                    f"small_minus_full_{metric}": ["mean", "std", "median"]
                    for metric in metric_cols
                },
                **{
                    f"small_over_full_{metric}": ["mean", "std", "median"]
                    for metric in metric_cols
                },
            }
        )
    return summary, pair_summary


def plot_pair_counts(pairs: pd.DataFrame, output_dir: Path, metric: str) -> Path | None:
    small_col = f"small_{metric}"
    full_col = f"full_{metric}"
    if pairs.empty or small_col not in pairs.columns:
        return None

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for idx, row in enumerate(pairs.itertuples(index=False)):
        x = [0, 1]
        y = [getattr(row, small_col), getattr(row, full_col)]
        ax.plot(x, y, color="#7a7a7a", alpha=0.45, linewidth=1)
        ax.scatter(x, y, s=28, color=["#4C78A8", "#F58518"], zorder=3)
    ax.set_xticks([0, 1], ["small FOV", "full FOV"])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Same-patient FOV motif counts ({len(pairs)} small/full pairs)")
    ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / f"same_patient_small_vs_full_{metric}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_fov = build_per_fov_counts(args)
    pairs = build_same_patient_pairs(per_fov)
    class_scale_summary, pair_summary = summarize(per_fov, pairs)

    per_fov.to_csv(args.output_dir / "fov_motif_counts_per_fov.csv", index=False)
    pairs.to_csv(args.output_dir / "same_patient_different_scale_pairs.csv", index=False)
    class_scale_summary.to_csv(args.output_dir / "fov_motif_counts_by_class_and_scale.csv", index=False)
    pair_summary.to_csv(args.output_dir / "same_patient_different_scale_pair_summary.csv")

    plot_paths = []
    for metric in ["significant_motif_count", "absence_significant_motif_count"]:
        path = plot_pair_counts(pairs, args.output_dir, metric)
        if path is not None:
            plot_paths.append(path)

    run_summary = {
        "scale_table": str(args.scale_table),
        "artifact": str(args.artifact),
        "absence_artifact": str(args.absence_artifact) if args.absence_artifact.exists() else None,
        "labels": args.labels,
        "small_scale_threshold": args.small_scale_threshold,
        "n_fovs": int(len(per_fov)),
        "n_patients_with_multiple_scales": int(pairs["patient_uid"].nunique()) if not pairs.empty else 0,
        "n_small_full_pairs": int(len(pairs)),
        "plots": [str(path) for path in plot_paths],
    }
    (args.output_dir / "fov_scale_motif_counts_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n"
    )

    print("Class/scale summary:")
    print(class_scale_summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nSame-patient small/full pairs:")
    if pairs.empty:
        print("No patients with both small and full FOV scales.")
    else:
        columns = [
            "patient_uid",
            "patient_class",
            "small_fov",
            "full_fov",
            "linear_scale_ratio",
            "small_significant_motif_count",
            "full_significant_motif_count",
            "small_minus_full_significant_motif_count",
        ]
        if "small_absence_significant_motif_count" in pairs.columns:
            columns.extend(
                [
                    "small_absence_significant_motif_count",
                    "full_absence_significant_motif_count",
                    "small_minus_full_absence_significant_motif_count",
                ]
            )
        print(pairs[columns].to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
