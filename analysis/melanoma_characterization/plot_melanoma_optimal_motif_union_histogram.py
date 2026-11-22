#!/usr/bin/env python3
"""Plot fold-union motif counts for optimized or fixed-0.46 melanoma shuffles."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import ast
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_OUTPUT_ROOT


def count_fold_union(path: Path) -> int:
    folds = pd.read_csv(path)
    if len(folds) != 38 or folds.test_patient_id.nunique() != 38:
        raise ValueError(f"Expected 38 distinct LOOCV folds in {path}")
    motif_ids = set()
    for value in folds.features:
        features = ast.literal_eval(value)
        if not isinstance(features, list):
            raise ValueError(f"Expected a motif list in {path}")
        motif_ids.update(features)
    return len(motif_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    parser.add_argument("--variant", choices=("optimal", "fixed"), default="optimal")
    parser.add_argument("--bin-width", type=int, default=None,
                        help="Motifs per histogram bin (default: 4 for fixed, 10 for optimal)")
    args = parser.parse_args()
    bin_width = args.bin_width if args.bin_width is not None else (4 if args.variant == "fixed" else 10)
    if bin_width < 1:
        parser.error("--bin-width must be positive")

    feature_file = f"{args.variant}_motif_space_features.csv"
    original_path = args.run_root / "original/from_artifact" / feature_file
    original_count = count_fold_union(original_path)
    records = []
    for seed_dir in sorted((args.run_root / "shuffled").glob("seed_*"),
                           key=lambda path: int(path.name.removeprefix("seed_"))):
        path = seed_dir / feature_file
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"seed": int(seed_dir.name.removeprefix("seed_")),
                        "fold_union_discriminative_motifs": count_fold_union(path)})
    counts = pd.DataFrame(records)
    if counts.empty or counts.seed.duplicated().any():
        raise ValueError("Expected at least one completed shuffle with unique seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output_dir / f"{args.variant}_fold_union_motif_counts.csv", index=False)

    max_count = int(max(counts.fold_union_discriminative_motifs.max(), original_count))
    bin_edges = np.arange(0, (max_count // bin_width + 2) * bin_width, bin_width)
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        ax.hist(counts.fold_union_discriminative_motifs, bins=bin_edges,
                color="#24558c", edgecolor="white", linewidth=0.6)
        ax.axvline(original_count, color="#df5948", linewidth=1.5,
                   label=f"Original = {original_count}")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlim=(0, bin_edges[-1]), xlabel="Fold-union discriminative motifs",
               ylabel="Number of shuffles",
               title=f"{'Optimized stringency' if args.variant == 'optimal' else 'Fixed stringency 0.46'} "
                     f"(n = {len(counts)} shuffles)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"{args.variant}_fold_union_motif_count_histogram.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    values = counts.fold_union_discriminative_motifs
    print(f"Shuffles: {len(values)}; original: {original_count}; "
          f"min: {values.min()}; median: {values.median():g}; "
          f"mean: {values.mean():.2f}; max: {values.max()}")


if __name__ == "__main__":
    main()
