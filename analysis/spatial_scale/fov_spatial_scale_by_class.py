#!/usr/bin/env python3
"""Summarize melanoma FOV bounding-box scale by tissue-state class."""

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
import re
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = _REPO_ROOT
DEFAULT_GRAPH_DIR = REPO_ROOT / "melanoma_tutorials" / "pet_data" / "Melanoma"
DEFAULT_PATIENT_CLASS_CSV = DEFAULT_GRAPH_DIR / "patient_class.csv"
DEFAULT_RAW_CENTROIDS_CSV = Path("/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime" / "fov_spatial_scale_nn_np"
)
GRAPH_FILE_RE = re.compile(r"Patient_(\d+)_FOVFOV(\d+)\.txt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--patient-class-csv", type=Path, default=DEFAULT_PATIENT_CLASS_CSV)
    parser.add_argument("--raw-centroids-csv", type=Path, default=DEFAULT_RAW_CENTROIDS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs="+", default=["NN", "NP"])
    parser.add_argument("--patient-col", default="patient number")
    parser.add_argument("--fov-col", default="fov")
    parser.add_argument("--height-col", default="centroid-0")
    parser.add_argument("--width-col", default="centroid-1")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def load_local_fovs(graph_dir: Path, patient_class_csv: Path, labels: list[str]) -> pd.DataFrame:
    classes = pd.read_csv(patient_class_csv, header=None, names=["patient_uid", "patient_class"])
    classes["patient_number"] = classes["patient_uid"].str.extract(r"Melanoma(\d+)").astype(int)
    classes = classes[classes["patient_class"].isin(labels)].copy()
    patient_to_class = dict(zip(classes["patient_number"], classes["patient_class"]))

    rows = []
    for path in sorted(graph_dir.glob("Patient_*_FOVFOV*.txt")):
        match = GRAPH_FILE_RE.match(path.name)
        if not match:
            continue
        patient_number = int(match.group(1))
        fov_number = int(match.group(2))
        patient_class = patient_to_class.get(patient_number)
        if patient_class is None:
            continue
        rows.append(
            {
                "patient_number": patient_number,
                "patient_uid": f"Melanoma{patient_number}",
                "patient_class": patient_class,
                "fov": f"FOV{fov_number}",
                "fov_number": fov_number,
                "graph_file": path.name,
            }
        )
    return pd.DataFrame(rows)


def load_matching_centroids(args: argparse.Namespace, local_fovs: pd.DataFrame) -> pd.DataFrame:
    usecols = [args.patient_col, args.fov_col, args.height_col, args.width_col]
    patients = set(local_fovs["patient_number"])
    wanted = set(zip(local_fovs["patient_number"], local_fovs["fov"]))
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
        raise ValueError("No matching centroid rows found for the selected graph FOVs.")
    return pd.concat(chunks, ignore_index=True)


def build_fov_scale_table(args: argparse.Namespace, local_fovs: pd.DataFrame) -> pd.DataFrame:
    centroids = load_matching_centroids(args, local_fovs)
    bbox = (
        centroids.groupby([args.patient_col, args.fov_col], observed=True)
        .agg(
            min_height_coord=(args.height_col, "min"),
            max_height_coord=(args.height_col, "max"),
            min_width_coord=(args.width_col, "min"),
            max_width_coord=(args.width_col, "max"),
            cell_count=(args.height_col, "size"),
        )
        .reset_index()
    )
    bbox["height"] = bbox["max_height_coord"] - bbox["min_height_coord"]
    bbox["width"] = bbox["max_width_coord"] - bbox["min_width_coord"]
    bbox["size"] = bbox["width"] * bbox["height"]

    merged = local_fovs.merge(
        bbox,
        left_on=["patient_number", "fov"],
        right_on=[args.patient_col, args.fov_col],
        how="left",
    )
    return merged.drop(columns=[args.patient_col, args.fov_col]).sort_values(
        ["patient_class", "patient_number", "fov_number"]
    )


def summarize_by_class(fov_scale: pd.DataFrame) -> pd.DataFrame:
    measured = fov_scale.dropna(subset=["width", "height", "size"]).copy()
    return (
        measured.groupby("patient_class", observed=True)
        .agg(
            n_fovs=("graph_file", "count"),
            mean_width=("width", "mean"),
            std_width=("width", "std"),
            mean_height=("height", "mean"),
            std_height=("height", "std"),
            mean_size=("size", "mean"),
            std_size=("size", "std"),
            mean_cell_count=("cell_count", "mean"),
        )
        .reset_index()
    )


def plot_class_overlay(fov_scale: pd.DataFrame, patient_class: str, output_dir: Path) -> Path:
    class_fovs = fov_scale[
        (fov_scale["patient_class"] == patient_class) & fov_scale[["width", "height"]].notna().all(axis=1)
    ].copy()
    if class_fovs.empty:
        raise ValueError(f"No measured FOVs available for class {patient_class}.")

    max_width = float(class_fovs["width"].max())
    max_height = float(class_fovs["height"].max())
    center_x = max_width / 2
    center_y = max_height / 2

    rectangles = []
    for row in class_fovs.itertuples(index=False):
        width = float(row.width)
        height = float(row.height)
        rectangles.append(
            {
                "left": center_x - width / 2,
                "right": center_x + width / 2,
                "bottom": center_y - height / 2,
                "top": center_y + height / 2,
                "width": width,
                "height": height,
            }
        )

    x_edges = np.array(sorted({0.0, max_width, *[r["left"] for r in rectangles], *[r["right"] for r in rectangles]}))
    y_edges = np.array(sorted({0.0, max_height, *[r["bottom"] for r in rectangles], *[r["top"] for r in rectangles]}))
    x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2
    y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2
    counts = np.zeros((len(y_midpoints), len(x_midpoints)), dtype=int)

    for rect in rectangles:
        x_mask = (x_midpoints >= rect["left"]) & (x_midpoints <= rect["right"])
        y_mask = (y_midpoints >= rect["bottom"]) & (y_midpoints <= rect["top"])
        counts[np.ix_(y_mask, x_mask)] += 1

    max_overlap = int(counts.max())
    bounds = np.arange(0.5, max_overlap + 1.5, 1)
    cmap = plt.get_cmap("viridis", max_overlap).copy()
    cmap.set_bad("white")
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    masked_counts = np.ma.masked_where(counts == 0, counts)
    mesh = ax.pcolormesh(x_edges, y_edges, masked_counts, cmap=cmap, norm=norm, shading="flat")
    ax.add_patch(Rectangle((0, 0), max_width, max_height, fill=False, edgecolor="black", linewidth=2.2))
    for rect in rectangles:
        ax.add_patch(
            Rectangle(
                (rect["left"], rect["bottom"]),
                rect["width"],
                rect["height"],
                fill=False,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.45,
            )
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, max_width)
    ax.set_ylim(0, max_height)
    ax.set_xlabel("FOV width")
    ax.set_ylabel("FOV height")
    ax.set_title(
        f"{patient_class} FOV size overlay "
        f"(n={len(class_fovs)}, max={max_width:,.1f} x {max_height:,.1f})"
    )
    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(1, max_overlap + 1))
    cbar.set_label("Overlapping FOV rectangles")

    output_path = output_dir / f"fov_spatial_scale_overlay_{patient_class}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_fovs = load_local_fovs(args.graph_dir, args.patient_class_csv, args.labels)
    if local_fovs.empty:
        raise ValueError(f"No local graph FOVs found for labels: {args.labels}")

    fov_scale = build_fov_scale_table(args, local_fovs)
    summary = summarize_by_class(fov_scale)

    fov_scale.to_csv(args.output_dir / "fov_spatial_scale_per_fov.csv", index=False)
    summary.to_csv(args.output_dir / "fov_spatial_scale_summary_by_class.csv", index=False)
    plot_paths = [plot_class_overlay(fov_scale, label, args.output_dir) for label in args.labels]

    run_summary = {
        "graph_dir": str(args.graph_dir),
        "patient_class_csv": str(args.patient_class_csv),
        "raw_centroids_csv": str(args.raw_centroids_csv),
        "labels": args.labels,
        "n_local_fovs": int(len(local_fovs)),
        "n_measured_fovs": int(fov_scale[["width", "height", "size"]].notna().all(axis=1).sum()),
        "std_definition": "sample standard deviation (pandas/std ddof=1)",
        "coordinate_mapping": {
            "width": args.width_col,
            "height": args.height_col,
            "size": "width * height",
        },
        "plots": [str(path) for path in plot_paths],
        "plot_alignment": "FOV rectangles are centered within the largest class-specific FOV rectangle.",
    }
    (args.output_dir / "fov_spatial_scale_run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n"
    )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
