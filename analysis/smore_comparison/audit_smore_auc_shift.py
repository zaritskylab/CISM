#!/usr/bin/env python3
"""Isolate cohort, motif-ID, and feature-mode effects at stringency 0.20."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


from pathlib import Path

import pandas as pd

from analysis.smore_comparison.compact_smore_results import (
    DEFAULT_METADATA_DIR,
    DEFAULT_SMORE_RESULTS,
    load_labeled_smore,
)
from analysis.smore_comparison.plot_smore_cism_pipeline_auc import run_cism_pipeline


OUTPUT = ANALYSIS_DIR / "outputs/stringency_clean"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full = load_labeled_smore(DEFAULT_SMORE_RESULTS, DEFAULT_METADATA_DIR)
    full = full.rename(columns={"patient number": "patient"})
    full["p_value"] = 0.01
    old_cohort = full[full.patient != 1].copy()
    cases = [
        ("original_setup_37", old_cohort, "motifNumber", "motif_and_composition", {"NN": 20, "NP": 17}),
        ("stable_seed_37", old_cohort, "seed", "motif_and_composition", {"NN": 20, "NP": 17}),
        ("stable_seed_38", full, "seed", "motif_and_composition", {"NN": 20, "NP": 18}),
        ("motif_only_38", full, "seed", "motif_only", {"NN": 20, "NP": 18}),
    ]
    records = []
    for name, data, identity, mode, counts in cases:
        print(f"AUDIT {name}", flush=True)
        result = run_cism_pipeline(data, OUTPUT, thresholds=(0.20,), identity=identity,
                                   feature_mode=mode, expected_class_counts=counts)
        row = result.iloc[0].to_dict() if not result.empty else {"stringency": 0.20}
        row.update({"case": name, "identity": identity, "feature_mode": mode,
                    "p_value_policy": "forced_0.01"})
        records.append(row)
        pd.DataFrame(records).to_csv(OUTPUT / "smore_auc_shift_audit.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
