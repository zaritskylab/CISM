#!/usr/bin/env python3
"""Re-evaluate SMORE at 0.35 with neutral scores for motif-absent patients."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse

import pandas as pd

from analysis.smore_comparison.evaluate_smore_matched import OUTPUT, evaluate, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="gdelaunay_p1_n0_train10_score2_gpu_noenrich")
    args = parser.parse_args()

    run_dir = OUTPUT / "runs" / args.run_tag
    patients = validate_manifest(pd.read_csv(OUTPUT / "cohort_fovs.csv"))
    features = pd.read_csv(run_dir / "smore_matched_patient_features.csv")
    summary, folds = evaluate(features, patients, (0.35,), max_per_class=30, max_failed=5)
    summary.to_csv(run_dir / "smore_auc_035_absent_neutral.csv", index=False)
    folds.to_csv(run_dir / "smore_fold_predictions_035_absent_neutral.csv", index=False)
    print(summary.to_string(index=False))
    print(folds.loc[folds.failed | folds.no_test_motifs,
                    ["patient", "true_class", "failed", "no_test_motifs",
                     "probability_NN_absent_neutral"]].to_string(index=False))


if __name__ == "__main__":
    main()
