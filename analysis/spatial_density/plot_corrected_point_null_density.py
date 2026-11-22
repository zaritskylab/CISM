"""Plot image-edge-corrected point-null motif-local density distributions."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "analysis" / "reports" / "motif_density_edge_corrected"
COLORS = {"NN": "#5B7DB1", "NP": "#D88452"}
RADII = {100: 50, 200: 100}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_DIR / "motif_density_null_samples_edge_corrected.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    data = data.loc[data["null_type"].eq("random_point") & data["motif_association"].isin(COLORS)].copy()
    if data.empty:
        raise ValueError("No NN/NP point-null samples found")
    if data["instance_id"].nunique() != 56 or data["motif_id"].nunique() != 19:
        raise ValueError("Expected 56 instances from 19 motifs")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=True, sharey=True)
    bins = np.arange(0, 0.0205, 0.00075)
    summary = []
    for ax, (radius_px, radius_um) in zip(axes, RADII.items()):
        metric = f"radius_{radius_px}_density"
        area = f"radius_{radius_px}_area_um2"
        count = f"radius_{radius_px}_cell_count"
        if data[[metric, area, count]].isna().any().any():
            raise ValueError(f"Missing corrected values for {radius_px} px")
        if not np.allclose(data[metric], data[count] / data[area], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Density is not cell count / clipped area for {radius_px} px")
        for group in ("NN", "NP"):
            group_data = data.loc[data["motif_association"].eq(group)]
            values = group_data[metric].to_numpy(dtype=float)
            if np.any(values < bins[0]) or np.any(values > bins[-1]):
                raise ValueError(f"Values extend outside plot bins for {group}, {radius_px} px")
            median = float(np.median(values))
            ax.hist(
                values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                histtype="stepfilled", alpha=0.38, color=COLORS[group],
                edgecolor=COLORS[group], linewidth=1.5,
                label=f"{group} (n={len(values)}, median={median:.4f})",
            )
            ax.axvline(median, color=COLORS[group], linewidth=1.5, linestyle="--")
            summary.append({
                "radius_pixels": radius_px,
                "radius_microns": radius_um,
                "motif_association": group,
                "n_point_null_samples": len(values),
                "n_motifs": group_data["motif_id"].nunique(),
                "n_instances": group_data["instance_id"].nunique(),
                "min_density": float(values.min()),
                "median_density": median,
                "mean_density": float(values.mean()),
                "max_density": float(values.max()),
                "n_zero_cell_circles": int(group_data[count].eq(0).sum()),
                "n_edge_clipped_circles": int((group_data[area] < np.pi * radius_um**2 - 1e-7).sum()),
            })
        ax.set_title(f"{radius_um} µm radius ({radius_px} px)", fontsize=13)
        ax.set_xlabel("Point-null local density (cells / µm²)")
        ax.grid(axis="y", color="#D8D8D8", alpha=0.65)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of samples per bin")
    axes[0].set_xlim(0, bins[-1])
    fig.suptitle("Point-null motif-local density | image-edge corrected", fontsize=15, y=0.98)
    fig.text(
        0.5, 0.015,
        "Each draw: cells within radius / (circle area inside its FOV image). "
        "Twenty draws per motif instance; draws are not independent biological replicates.",
        ha="center", fontsize=8.5, color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "point_null_density_distribution_edge_corrected"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame(summary).to_csv(args.output_dir / "point_null_density_distribution_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
