"""Recalculate saved motif radius densities using circle area inside each image."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import math
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    PROJECT_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
    / "motif_density_analysis_report_notebook03_discriminative"
)
DEFAULT_IMAGE_DIR = Path("/groups/assafza_group/assafza/Datasets/Melanoma/raw_data")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "reports" / "motif_density_edge_corrected"
PIXEL_SIZE_MICRONS = 0.5
RADII = (25, 50, 100, 150, 200)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.spatial_density.generate_motif_density_complementary_report import build_figures, build_grouped_figures  # noqa: E402


def quadrant_area(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Area of a radius-r circle in the positive rectangle [0,a] x [0,b]."""
    a = np.clip(np.asarray(a, dtype=float), 0.0, radius)
    b = np.clip(np.asarray(b, dtype=float), 0.0, radius)
    x0 = np.sqrt(np.maximum(radius**2 - b**2, 0.0))
    flat_width = np.minimum(a, x0)

    def integral(x: np.ndarray) -> np.ndarray:
        return 0.5 * (
            x * np.sqrt(np.maximum(radius**2 - x**2, 0.0))
            + radius**2 * np.arcsin(np.clip(x / radius, 0.0, 1.0))
        )

    return b * flat_width + np.where(a > x0, integral(a) - integral(x0), 0.0)


def circle_image_area_px2(
    x: np.ndarray, y: np.ndarray, width: np.ndarray, height: np.ndarray, radius: float
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    width = np.asarray(width, dtype=float)
    height = np.asarray(height, dtype=float)
    if np.any((x < 0) | (y < 0) | (x > width) | (y > height)):
        raise ValueError("A density center lies outside its FOV image")
    return (
        quadrant_area(x, y, radius)
        + quadrant_area(width - x, y, radius)
        + quadrant_area(x, height - y, radius)
        + quadrant_area(width - x, height - y, radius)
    )


def image_dimensions(image_dir: Path, fovs: set[int]) -> pd.DataFrame:
    rows = []
    for fov in sorted(fovs):
        archive = image_dir / f"FOV{fov}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"Image archive missing: {archive}")
        with zipfile.ZipFile(archive) as zipped:
            tif = next((name for name in zipped.namelist() if name.lower().endswith(".tif")), None)
            if tif is None:
                raise ValueError(f"No TIFF found in {archive}")
            with zipped.open(tif) as image_file:
                width, height = Image.open(image_file).size
        rows.append({"FOV": fov, "image_width_px": width, "image_height_px": height})
    return pd.DataFrame(rows)


def correct_table(data: pd.DataFrame, dimensions: pd.DataFrame, pixel_size_microns: float) -> pd.DataFrame:
    corrected = data.merge(dimensions, on="FOV", how="left", validate="many_to_one")
    if corrected[["image_width_px", "image_height_px"]].isna().any().any():
        raise ValueError("Some FOVs have no image dimensions")
    x = corrected["motif_centroid_x"].to_numpy(dtype=float)
    y = corrected["motif_centroid_y"].to_numpy(dtype=float)
    width = corrected["image_width_px"].to_numpy(dtype=float)
    height = corrected["image_height_px"].to_numpy(dtype=float)
    for radius in RADII:
        metric = f"radius_{radius}_density"
        old_density = corrected[metric].to_numpy(dtype=float)
        full_area_um2 = math.pi * (radius * pixel_size_microns) ** 2
        unrounded_count = old_density * full_area_um2
        cell_count = np.rint(unrounded_count).astype(int)
        if np.max(np.abs(unrounded_count - cell_count)) > 1e-6:
            raise ValueError(f"Cannot recover integer cell counts for {metric}")
        area_um2 = circle_image_area_px2(x, y, width, height, radius) * pixel_size_microns**2
        if np.any(area_um2 <= 0) or np.any(area_um2 > full_area_um2 + 1e-7):
            raise ValueError(f"Invalid clipped circle area for {metric}")
        corrected[f"{metric}_full_circle"] = old_density
        corrected[f"radius_{radius}_cell_count"] = cell_count
        corrected[f"radius_{radius}_area_um2"] = area_um2
        corrected[metric] = cell_count / area_um2
    return corrected


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for i in range(len(values) - 1, -1, -1):
        running = min(running, values[order[i]] * len(values) / (i + 1))
        adjusted[order[i]] = running
    return pd.Series(adjusted, index=p_values.index)


def motif_summary(observed: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in RADII:
        metric = f"radius_{radius}_density"
        for null_type in ("random_point", "random_connected_cellset"):
            selected_null = nulls[nulls["null_type"] == null_type]
            for motif_id, obs in observed.groupby("motif_id", observed=True):
                pool = selected_null[selected_null["motif_id"] == motif_id]
                replicate = pool.groupby("sample_index", observed=True).agg(
                    density=(metric, "mean"), n_instances=("instance_id", "nunique")
                )
                replicate = replicate[replicate["n_instances"] == obs["instance_id"].nunique()]
                if replicate.empty:
                    continue
                observed_mean = float(obs[metric].mean())
                null_mean = float(replicate["density"].mean())
                null_deviation = np.abs(replicate["density"] - null_mean)
                p = (1 + int((null_deviation >= abs(observed_mean - null_mean)).sum())) / (len(replicate) + 1)
                rows.append({
                    "motif_id": motif_id,
                    "motif_association": obs["motif_association"].iloc[0],
                    "metric": metric,
                    "null_type": null_type,
                    "n_instances": len(obs),
                    "n_null_replicates": len(replicate),
                    "observed_mean": observed_mean,
                    "null_mean": null_mean,
                    "enrichment_delta": observed_mean - null_mean,
                    "enrichment_ratio": observed_mean / null_mean if null_mean else np.nan,
                    "permutation_p": p,
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_q"] = bh_adjust(result["permutation_p"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--micron-per-pixel", type=float, default=PIXEL_SIZE_MICRONS)
    args = parser.parse_args()
    if args.micron_per_pixel <= 0:
        parser.error("Pixel scale must be positive")
    observed = pd.read_csv(args.source_dir / "motif_density_instance_metrics.csv")
    nulls = pd.read_csv(args.source_dir / "motif_density_null_samples.csv")
    if observed.motif_id.nunique() != 19:
        raise ValueError(f"Expected 19 motifs, found {observed.motif_id.nunique()}")
    dimensions = image_dimensions(args.image_dir, set(observed.FOV.astype(int)) | set(nulls.FOV.astype(int)))
    observed = correct_table(observed, dimensions, args.micron_per_pixel)
    nulls = correct_table(nulls, dimensions, args.micron_per_pixel)
    summary = motif_summary(observed, nulls)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed.to_csv(args.output_dir / "motif_density_instance_metrics_edge_corrected.csv", index=False)
    nulls.to_csv(args.output_dir / "motif_density_null_samples_edge_corrected.csv", index=False)
    summary.to_csv(args.output_dir / "motif_density_summary_edge_corrected.csv", index=False)
    dimensions.to_csv(args.output_dir / "image_dimensions.csv", index=False)
    figure_dir = args.output_dir / "figures" / "instance_and_motif_no_points"
    report_pdf = args.output_dir / "motif_density_edge_corrected_nn_np_null_report_no_points.pdf"
    test_rows, _ = build_figures(
        observed, nulls, figure_dir, report_pdf, show_points=False,
        footer_note="Image-edge corrected circle area; 19 motifs; bounded to one FOV per motif.",
    )
    pd.DataFrame(test_rows).to_csv(args.output_dir / "motif_density_edge_corrected_nn_np_null_tests.csv", index=False)
    build_grouped_figures(
        observed, nulls, output_dir=args.output_dir, name_prefix="motif_density_edge_corrected",
        footer_note="Image-edge corrected; one FOV per patient in this bounded run. Tests are exploratory.",
    )
    for radius in (100, 200):
        metric = f"radius_{radius}_density"
        affected = int((observed[metric] > observed[f"{metric}_full_circle"] + 1e-12).sum())
        print(f"Radius {radius} px: {affected}/{len(observed)} observed instances changed")
    print(f"Wrote corrected motif-density results to {args.output_dir}")


if __name__ == "__main__":
    main()
