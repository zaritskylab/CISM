"""Remake 100-pixel tissue and motif density panels at 0.39 micron/pixel."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


ROOT = Path(__file__).resolve().parents[2]
TISSUE_CSV = ROOT / "analysis/reports/tissue_density_patient_50um/nonempty_patch_density.csv"
MOTIF_DIR = ROOT / "analysis/reports/motif_density_edge_corrected"
OUTPUT_DIR = ROOT / "analysis/reports/density_100px_039um"
OLD_MICRONS_PER_PIXEL = 0.5
NEW_MICRONS_PER_PIXEL = 0.39
RADIUS_PIXELS = 100
COLORS = {"NN": "#5B7DB1", "NP": "#D88452", "NN_null": "#9DB6D8", "NP_null": "#E8A06F"}


def styled_boxplot(ax: plt.Axes, values: list[np.ndarray], positions: list[float], colors: list[str]) -> None:
    result = ax.boxplot(
        values, positions=positions, widths=0.66, patch_artist=True, showfliers=False,
        boxprops={"edgecolor": "#666666", "linewidth": 1.3},
        medianprops={"color": "#424242", "linewidth": 2},
        whiskerprops={"color": "#999999", "linewidth": 1.5},
        capprops={"color": "#999999", "linewidth": 1.5},
    )
    for box, color in zip(result["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.94)
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=17, length=5, width=1.1)
    ax.yaxis.set_major_locator(MultipleLocator(0.002))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def remake_tissue_panel() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    patches = pd.read_csv(TISSUE_CSV)
    expected = {"Group", "patient number", "fov", "cell_count", "patch_area_um2"}
    if not expected.issubset(patches.columns):
        raise ValueError(f"Missing tissue columns: {expected - set(patches.columns)}")
    if set(patches["Group"]) != {"NN", "NP"}:
        raise ValueError("Expected NN and NP patches only")
    patches["patch_area_039_um2"] = (
        patches["patch_area_um2"] / OLD_MICRONS_PER_PIXEL**2 * NEW_MICRONS_PER_PIXEL**2
    )
    patches["density_039_cells_um2"] = patches["cell_count"] / patches["patch_area_039_um2"]
    if not np.allclose(patches["density_039_cells_um2"], patches["density_cells_um2"] * (OLD_MICRONS_PER_PIXEL / NEW_MICRONS_PER_PIXEL) ** 2):
        raise ValueError("Recomputed tissue density does not match expected scale change")
    fovs = patches.groupby(["Group", "patient number", "fov"], observed=True)["density_039_cells_um2"].mean().reset_index()
    patients = fovs.groupby(["Group", "patient number"], observed=True)["density_039_cells_um2"].mean().reset_index()
    patients.to_csv(OUTPUT_DIR / "panel_A_right_patient_density_039um.csv", index=False)
    values = {group: patients.loc[patients.Group.eq(group), "density_039_cells_um2"].to_numpy() for group in ("NN", "NP")}
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    styled_boxplot(ax, [values["NN"], values["NP"]], [1, 2], [COLORS["NN"], COLORS["NP"]])
    ax.set_xticks([1, 2], ["NN", "NP"])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel("Cell density (cells / µm²)", fontsize=19, labelpad=12)
    ax.set_title("Tissue density | 100 px patches (39 × 39 µm)", fontsize=18, pad=16)
    fig.tight_layout(pad=1.5)
    save_figure(fig, "panel_A_right_patient_tissue_density_100px_039um")
    return patients, values


def remake_motif_panel() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    observed = pd.read_csv(MOTIF_DIR / "motif_density_instance_metrics_edge_corrected.csv")
    nulls = pd.read_csv(MOTIF_DIR / "motif_density_null_samples_edge_corrected.csv")
    nulls = nulls.loc[nulls.null_type.eq("random_point")].copy()
    if observed.motif_id.nunique() != 19 or observed.instance_id.nunique() != 56 or len(nulls) != 1120:
        raise ValueError("Unexpected motif or point-null sample count")
    metric = "radius_100_density_039_cells_um2"
    for frame in (observed, nulls):
        area_px2 = frame["radius_100_area_um2"] / OLD_MICRONS_PER_PIXEL**2
        frame[metric] = frame["radius_100_cell_count"] / (area_px2 * NEW_MICRONS_PER_PIXEL**2)
        if not np.allclose(frame[metric], frame["radius_100_density"] * (OLD_MICRONS_PER_PIXEL / NEW_MICRONS_PER_PIXEL) ** 2):
            raise ValueError("Recomputed motif density does not match expected scale change")
    motif_keys = ["motif_id", "motif_association"]
    observed_means = observed.groupby(motif_keys, observed=True)[metric].mean().rename("observed_density_039_cells_um2")
    null_means = nulls.groupby(motif_keys, observed=True)[metric].mean().rename("point_null_density_039_cells_um2")
    paired = observed_means.to_frame().join(null_means, how="inner").reset_index()
    if len(paired) != 19 or paired[["observed_density_039_cells_um2", "point_null_density_039_cells_um2"]].isna().any().any():
        raise ValueError("Point-null and observed motif means did not pair for all 19 motifs")
    paired.to_csv(OUTPUT_DIR / "panel_C_motif_means_039um.csv", index=False)
    values = {
        f"{group}_{kind}": paired.loc[paired.motif_association.eq(group), f"{kind}_density_039_cells_um2"].to_numpy()
        for group in ("NN", "NP") for kind in ("observed", "point_null")
    }
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    styled_boxplot(
        ax,
        [values["NN_observed"], values["NP_observed"], values["NN_point_null"], values["NP_point_null"]],
        [1, 2, 4, 5],
        [COLORS["NN"], COLORS["NP"], COLORS["NN_null"], COLORS["NP_null"]],
    )
    ax.axvline(3, color="#999999", linestyle="--", linewidth=1.3)
    ax.set_xticks([1, 2, 4, 5], ["NN", "NP", "NN", "NP"])
    ax.set_xlim(0.45, 5.55)
    ax.text(1.5, -0.105, "Motif-local", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=17)
    ax.text(4.5, -0.105, "Point null", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=17)
    ax.set_ylabel("Cell density (cells / µm²)", fontsize=19, labelpad=12)
    ax.set_title("Motif-local density | 100 px radius (39 µm)", fontsize=18, pad=16)
    fig.tight_layout(rect=(0, 0.065, 1, 1), pad=1.5)
    save_figure(fig, "panel_C_motif_point_null_density_100px_039um")
    return paired, values


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patients, tissue_values = remake_tissue_panel()
    motifs, motif_values = remake_motif_panel()
    tests = []
    for panel, comparison, a, b in [
        ("A-right", "NN vs NP patients", tissue_values["NN"], tissue_values["NP"]),
        ("C", "NN vs NP observed motifs", motif_values["NN_observed"], motif_values["NP_observed"]),
        ("C", "NN observed vs matched-group point null", motif_values["NN_observed"], motif_values["NN_point_null"]),
        ("C", "NP observed vs matched-group point null", motif_values["NP_observed"], motif_values["NP_point_null"]),
    ]:
        tests.append({
            "panel": panel, "comparison": comparison, "n_a": len(a), "n_b": len(b),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "welch_p": float(ttest_ind(a, b, equal_var=False).pvalue),
            "mannwhitney_p": float(mannwhitneyu(a, b, alternative="two-sided").pvalue),
        })
    pd.DataFrame(tests).to_csv(OUTPUT_DIR / "plot_comparison_tests_039um.csv", index=False)
    print(f"Saved 0.39 µm/pixel figures: {len(patients)} patients, {len(motifs)} motifs; scale factor {(OLD_MICRONS_PER_PIXEL / NEW_MICRONS_PER_PIXEL) ** 2:.6f}")
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
