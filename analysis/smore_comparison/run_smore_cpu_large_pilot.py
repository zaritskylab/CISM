#!/usr/bin/env python3
"""Validate and run one of the five largest matched melanoma FOVs on CPU."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image


ROOT = _REPO_ROOT
OUTPUT = ROOT / "analysis/outputs/smore_matched"
RAW_IMAGES = Path("/groups/assafza_group/assafza/Datasets/Melanoma/raw_data")
N_FOVS = 5
IMAGE_SIZE = (2048, 2048)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--path-probability", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=28200)
    args = parser.parse_args()
    if not 0 <= args.task_id < N_FOVS:
        parser.error(f"--task-id must be in 0..{N_FOVS - 1}")
    if not 0 < args.path_probability <= 1:
        parser.error("--path-probability must be in (0, 1]")

    manifest = pd.read_csv(OUTPUT / "cohort_fovs.csv")
    selected = manifest.sort_values(["graph_vertices", "fov"], ascending=[False, True]).head(N_FOVS)
    if len(selected) != N_FOVS or selected.fov.nunique() != N_FOVS:
        raise ValueError("Could not select five distinct cohort FOVs")
    row = selected.iloc[args.task_id]
    fov = str(row.fov)
    with ZipFile(RAW_IMAGES / f"{fov}.zip") as archive:
        first_tiff = next(name for name in archive.namelist() if name.lower().endswith(".tif"))
        with Image.open(BytesIO(archive.read(first_tiff))) as image:
            dimensions = image.size
    if dimensions != IMAGE_SIZE:
        raise ValueError(f"{fov} has image size {dimensions}, expected {IMAGE_SIZE}")

    cells = pd.read_csv(row.smore_input, usecols=["Centroid_X", "Centroid_Y"])
    if len(cells) != int(row.graph_vertices):
        raise ValueError(f"{fov} cell count does not match the cohort manifest")
    if not ((cells >= 0) & (cells < IMAGE_SIZE[0])).all().all():
        raise ValueError(f"{fov} has cell centroids outside the 2048 x 2048 image")
    print(f"{fov}: patient {row.patient}, {len(cells)} cells, image {dimensions}, "
          f"path probability {args.path_probability:g}", flush=True)
    if args.validate_only:
        return

    subprocess.run([
        sys.executable, "-m", "analysis.smore_comparison.run_smore_matched_batch",
        "--fov", fov,
        "--backend", "cpu",
        "--graph-mode", "delaunay",
        "--path-probability", str(args.path_probability),
        "--max-motifs", "0",
        "--n-train", "10",
        "--n-score", "2",
        "--no-enrich",
        "--timeout", str(args.timeout),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
