#!/usr/bin/env python3
"""Compare full-FOV and cropped-FOV cell counts, with and without area scaling."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = _REPO_ROOT
RUNTIME = ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
CROP_ROOT = RUNTIME / "crop_fov_scale_experiment"
OUTPUT = CROP_ROOT / "cell_count_scale_control"
FRACTIONS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def load_counts() -> pd.DataFrame:
    full = pd.read_csv(RUNTIME / "fov_spatial_scale_nn_np" / "fov_spatial_scale_per_fov.csv")
    rows = [
        {
            "patient_uid": row.patient_uid,
            "source_fov": f"FOV{row.fov_number}",
            "bootstrap_index": pd.NA,
            "retained_area": 1.0,
            "cells": int(row.cell_count),
        }
        for row in full.itertuples(index=False)
    ]
    full_sources = set(zip(full.patient_uid, ("FOV" + full.fov_number.astype(str))))

    for fraction in FRACTIONS[1:]:
        token = f"{round(fraction * 100):03d}"
        crop = pd.read_csv(CROP_ROOT / f"crop_{token}_metadata.csv")
        if crop["fraction_mode"].nunique() != 1 or crop["fraction_mode"].iloc[0] != "area":
            raise ValueError(f"Crop {token} was not prepared in area mode")
        if set(zip(crop.patient_uid, crop.source_fov)) != full_sources:
            raise ValueError(f"Crop {token} does not cover exactly the full-FOV cohort")
        if not crop.groupby(["patient_uid", "source_fov"]).size().eq(10).all():
            raise ValueError(f"Crop {token} does not have 10 replicates per source FOV")
        rows.extend(
            {
                "patient_uid": row.patient_uid,
                "source_fov": row.source_fov,
                "bootstrap_index": int(row.bootstrap_index),
                "retained_area": fraction,
                "cells": int(row.cells),
            }
            for row in crop.itertuples(index=False)
        )

    data = pd.DataFrame(rows)
    data["cells_per_full_fov_area"] = data["cells"] / data["retained_area"]
    return data


def make_plot(data: pd.DataFrame, column: str, title: str, ylabel: str, stem: str) -> None:
    groups = [data.loc[data.retained_area.eq(fraction), column].to_numpy() for fraction in FRACTIONS]
    labels = [f"{fraction:.0%}" for fraction in FRACTIONS]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    artists = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.62,
                         showfliers=False, medianprops={"color": "#17252c", "linewidth": 1.7})
    for index, box in enumerate(artists["boxes"]):
        box.set_facecolor("#356c82" if index == 0 else "#91b7a7")
        box.set_edgecolor("#2c4a50")
    for key in ("whiskers", "caps"):
        for line in artists[key]:
            line.set_color("#52646b")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Tissue area retained")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d7e1e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.21, "100%: 76 FOVs  |  Each crop size: 760 crops from the same 76 FOVs",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#52646b")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.25)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=240)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_counts()
    data.to_csv(OUTPUT / "full_and_cropped_fov_cell_counts.csv", index=False)
    make_plot(data, "cells", "Cells in full FOVs and cropped tissue", "Cells per FOV or crop",
              "cell_counts_raw")
    make_plot(data, "cells_per_full_fov_area", "Cells adjusted for retained tissue area",
              "Cells per full-FOV equivalent area", "cell_counts_area_normalized")
    print(data.groupby("retained_area", sort=False).agg(
        n=("cells", "size"), raw_median=("cells", "median"),
        normalized_median=("cells_per_full_fov_area", "median")
    ).to_string())
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
