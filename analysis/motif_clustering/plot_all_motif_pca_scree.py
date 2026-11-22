#!/usr/bin/env python3
"""Plot PCA variance for the unfiltered NN/NP patient-by-motif matrix."""

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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = ANALYSIS_DIR / "outputs" / "fanmod_patient_motif_clustering_NN_NP_tutorial_artifact_all_motifs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    matrix = pd.read_csv(args.input_dir / "patient_motif_matrix_cism_frequency.csv", index_col=0)
    if matrix.shape[0] != 38:
        raise ValueError(f"Expected 38 NN/NP patients, found {matrix.shape[0]}")
    if not np.isfinite(matrix.to_numpy()).all() or (matrix.to_numpy() < 0).any():
        raise ValueError("Motif frequencies must be nonnegative and finite")

    log_values = np.log1p(matrix.to_numpy(dtype=float) * 1_000_000)
    scaled = StandardScaler().fit_transform(log_values)
    pca = PCA(n_components=matrix.shape[0] - 1, svd_solver="full")
    pca.fit(scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pcs = np.arange(1, len(explained) + 1)
    crossing = int(np.searchsorted(cumulative, .8) + 1)
    nearest = int(np.argmin(np.abs(cumulative - .8)) + 1)

    output = args.input_dir / "pca_cutoff_review"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "PC": pcs,
        "variance_explained_pct": explained * 100,
        "cumulative_variance_pct": cumulative * 100,
    }).to_csv(output / "pca_variance.csv", index=False, float_format="%.8g")
    details = {
        "patients": int(matrix.shape[0]),
        "motifs": int(matrix.shape[1]),
        "transform": "log1p(frequency * 1_000_000)",
        "scaling": "StandardScaler per motif across patients",
        "pca_solver": "full",
        "max_nonzero_components": int(len(explained)),
        "first_at_least_80_pct": crossing,
        "variance_at_crossing_pct": float(cumulative[crossing - 1] * 100),
        "nearest_to_80_pct": nearest,
        "variance_at_nearest_pct": float(cumulative[nearest - 1] * 100),
        "variance_at_20_pct": float(cumulative[19] * 100),
        "zero_variance_motifs": int(np.count_nonzero(log_values.var(axis=0) == 0)),
    }
    (output / "pca_cutoff_summary.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11.7, 6.2))
    fig.subplots_adjust(left=.09, right=.90, top=.79, bottom=.17)
    fig.text(.09, .95, "PCA scree plot: all observed motifs", fontsize=18, weight="bold")
    fig.text(.09, .89, f"{matrix.shape[0]} NN/NP patients x {matrix.shape[1]:,} motifs | log1p(frequency x 10^6), then per-motif standardization", fontsize=10)
    fig.text(.09, .845, "Bars show each PC's contribution; the line shows cumulative motif variance explained.", fontsize=10)

    ax.bar(pcs, explained * 100, width=.78, color="#858E92", edgecolor="white", linewidth=.6, zorder=2)
    ax.set_xlim(.3, len(pcs) + .7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, len(pcs)])
    ax.set_xlabel("Number of principal components retained", labelpad=9)
    ax.set_ylabel("Individual variance explained (%)", labelpad=9)
    ax.grid(axis="y", color="#DDE3E5", linewidth=.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    cumulative_ax = ax.twinx()
    cumulative_ax.plot(pcs, cumulative * 100, color="#2F6C9E", linewidth=2.2, zorder=3)
    cumulative_ax.axhline(80, color="#D77A3D", linewidth=1.3, linestyle="--")
    cumulative_ax.axvline(crossing, color="#D77A3D", linewidth=1, linestyle=":")
    cumulative_ax.scatter([crossing], [cumulative[crossing - 1] * 100], s=58, color="#D77A3D", zorder=4)
    cumulative_ax.set_ylim(0, 105)
    cumulative_ax.set_ylabel("Cumulative variance explained (%)", labelpad=10, color="#2F6C9E")
    cumulative_ax.spines[["top", "left"]].set_visible(False)
    cumulative_ax.annotate(f"First >=80%: {crossing} PCs ({cumulative[crossing - 1] * 100:.1f}%)",
                           (crossing, cumulative[crossing - 1] * 100), xytext=(7, -24),
                           textcoords="offset points", color="#A75520", fontsize=10)
    fig.text(.09, .045, f"20 PCs: {cumulative[19] * 100:.1f}% | Nearest 80%: {nearest} PCs ({cumulative[nearest - 1] * 100:.1f}%).", fontsize=10)
    fig.savefig(output / "pca_scree.png", dpi=220)
    fig.savefig(output / "pca_scree.pdf")
    plt.close(fig)
    print(json.dumps(details, indent=2))
    print(output / "pca_scree.pdf")


if __name__ == "__main__":
    main()
