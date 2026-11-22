#!/usr/bin/env python3
"""Patient-level label-permutation sanity check for the provisional SMORE AUC."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.smore_comparison.plot_smore_cism_pipeline_auc import OUTPUT, load_smore, run_cism_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    original = load_smore("provisional")
    cohort = original[["patient", "class"]].drop_duplicates().sort_values("patient")
    if len(cohort) != 38:
        raise ValueError("Expected 38 patients")
    rows = []
    for seed in range(args.permutations):
        rng = np.random.RandomState(seed + 4100)
        class_map = dict(zip(cohort.patient, rng.permutation(cohort["class"])))
        shuffled = original.copy()
        shuffled["class"] = shuffled.patient.map(class_map)
        result = run_cism_pipeline(shuffled, args.output_dir, thresholds=(0.30,))
        rows.append({"permutation_seed": seed + 4100,
                     "auc": float(result.auc_failed_equal.iloc[0]),
                     "failed_folds": int(result.failed_folds.iloc[0])})
        pd.DataFrame(rows).to_csv(args.output_dir / "smore_provisional_label_permutations.csv",
                                  index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
