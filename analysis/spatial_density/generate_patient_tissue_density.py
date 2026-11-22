"""Summarize NN/NP tissue density from non-empty 50 x 50 micron patches."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELLS = Path("/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv")
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "reports" / "tissue_density_patient_50um"
COLORS = {"NN": "#5B7DB1", "NP": "#D88452"}


def build_tables(cells: pd.DataFrame, patch_pixels: int, image_pixels: int, micron_per_pixel: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"Group", "patient number", "fov", "centroid-0", "centroid-1"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"Missing cell columns: {sorted(missing)}")
    cells = cells[cells["Group"].isin(["NN", "NP"])].copy()
    if cells.empty:
        raise ValueError("No NN/NP cells found")
    if not (
        cells[["centroid-0", "centroid-1"]].ge(0).all().all()
        and cells[["centroid-0", "centroid-1"]].lt(image_pixels).all().all()
    ):
        raise ValueError("Cell coordinates extend outside the expected image rectangle")
    if cells.groupby("patient number")["Group"].nunique().max() != 1:
        raise ValueError("A patient has multiple tissue-state labels")

    cells["patch_x"] = np.floor(cells["centroid-0"] / patch_pixels).astype(int)
    cells["patch_y"] = np.floor(cells["centroid-1"] / patch_pixels).astype(int)
    keys = ["Group", "patient number", "fov"]
    patches = cells.groupby(keys + ["patch_x", "patch_y"], observed=True).size().reset_index(name="cell_count")
    width = np.minimum(patch_pixels, image_pixels - patches["patch_x"] * patch_pixels)
    height = np.minimum(patch_pixels, image_pixels - patches["patch_y"] * patch_pixels)
    patches["patch_area_um2"] = width * height * micron_per_pixel**2
    patches["density_cells_um2"] = patches["cell_count"] / patches["patch_area_um2"]

    fovs = patches.groupby(keys, observed=True).agg(
        density_cells_um2=("density_cells_um2", "mean"),
        n_nonempty_patches=("density_cells_um2", "size"),
        n_cells=("cell_count", "sum"),
    ).reset_index()
    patients = fovs.groupby(["Group", "patient number"], observed=True).agg(
        density_cells_um2=("density_cells_um2", "mean"),
        n_fovs=("fov", "nunique"),
        n_cells=("n_cells", "sum"),
    ).reset_index()
    return patches, fovs, patients


def tests(patients: pd.DataFrame) -> pd.DataFrame:
    nn = patients.loc[patients.Group == "NN", "density_cells_um2"].to_numpy()
    np_values = patients.loc[patients.Group == "NP", "density_cells_um2"].to_numpy()
    welch = ttest_ind(nn, np_values, equal_var=False)
    mw = mannwhitneyu(nn, np_values, alternative="two-sided")
    return pd.DataFrame([{
        "n_NN_patients": len(nn),
        "n_NP_patients": len(np_values),
        "NN_mean": float(nn.mean()),
        "NP_mean": float(np_values.mean()),
        "NN_median": float(np.median(nn)),
        "NP_median": float(np.median(np_values)),
        "mean_difference_NN_minus_NP": float(nn.mean() - np_values.mean()),
        "welch_t": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "mannwhitney_u": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
    }])


def plot(patients: pd.DataFrame, test_row: pd.Series, output_dir: Path, patch_microns: float) -> None:
    groups = [patients.loc[patients.Group == group, "density_cells_um2"].to_numpy() for group in ("NN", "NP")]
    fig, ax = plt.subplots(figsize=(6.2, 5.0), facecolor="white")
    bp = ax.boxplot(
        groups, tick_labels=["NN", "NP"], patch_artist=True, showfliers=False,
        widths=0.48, boxprops={"edgecolor": "#666666"},
        medianprops={"color": "#555555", "linewidth": 1.3},
        whiskerprops={"color": "#999999"}, capprops={"color": "#999999"},
    )
    for box, group in zip(bp["boxes"], ("NN", "NP")):
        box.set_facecolor(COLORS[group])
        box.set_alpha(0.92)
    ax.set_title(f"Patient-level tissue density | {patch_microns:g} x {patch_microns:g} micron patches")
    ax.set_ylabel("Cell density (cells / micron²)")
    ax.grid(axis="y", color="#D0D0D0", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    summary = (
        f"Patients: NN {int(test_row.n_NN_patients)}, NP {int(test_row.n_NP_patients)}   |   "
        f"Welch p = {test_row.welch_p:.3g}   |   Mann-Whitney p = {test_row.mannwhitney_p:.3g}"
    )
    fig.text(0.5, 0.035, summary, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_dir / "patient_tissue_density_50um.png", dpi=220)
    fig.savefig(output_dir / "patient_tissue_density_50um.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-csv", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-pixels", type=int, default=100)
    parser.add_argument("--image-pixels", type=int, default=2048)
    parser.add_argument("--micron-per-pixel", type=float, default=0.5)
    args = parser.parse_args()
    if args.patch_pixels <= 0 or args.image_pixels <= 0 or args.micron_per_pixel <= 0:
        parser.error("Patch, image, and pixel scale must be positive")
    cells = pd.read_csv(args.cells_csv, usecols=["Group", "patient number", "fov", "centroid-0", "centroid-1"])
    patches, fovs, patients = build_tables(cells, args.patch_pixels, args.image_pixels, args.micron_per_pixel)
    result = tests(patients)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patches.to_csv(args.output_dir / "nonempty_patch_density.csv", index=False)
    fovs.to_csv(args.output_dir / "fov_density.csv", index=False)
    patients.to_csv(args.output_dir / "patient_density.csv", index=False)
    result.to_csv(args.output_dir / "patient_density_tests.csv", index=False)
    plot(patients, result.iloc[0], args.output_dir, args.patch_pixels * args.micron_per_pixel)
    print(result.to_string(index=False))
    print(f"Wrote patient tissue-density figure and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
