#!/usr/bin/env python3
"""Reassess melanoma fixed-0.46 AUC using the CRC/TNBC permutation evaluator."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import numpy as np
import pandas as pd

from analysis.permutation_controls.crc_label_permutation import evaluate, prepare_data
from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_ARTIFACT, DEFAULT_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "fixed_046_reassessment_historical")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.seed_start < 0 or args.count < 1 or args.n_jobs < 1:
        parser.error("Seed start must be nonnegative; count and workers must be positive")

    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    patient_table = artifact["discriminator"].patient_class_df
    original = patient_table.loc[patient_table.patient_class.isin(["NN", "NP"]), "patient_class"].copy()
    original = original.map({"NN": "CLR", "NP": "DII"})
    original.index.name = "patient"
    if original.value_counts().to_dict() != {"CLR": 20, "DII": 18}:
        raise ValueError("Expected the 38-patient NN/NP melanoma cohort")
    dataset = artifact["cism"].motifs_dataset
    dataset = dataset.loc[dataset.Patient_uId.isin(original.index),
                          ["Patient_uId", "ID", "FOV", "p_value", "Count", "Freq", "nunique_colors"]].copy()
    significant, matrix = prepare_data(dataset, original.index)
    config = {"max_class_features": 30, "forest_seed": 0, "trees": 100,
              "rng_mode": "historical", "fixed_stringency": 0.46,
              "positive_class": "NN", "zero_features": "p_NN=0.5; all patients retained",
              "engine": "analysis/permutation_controls/crc_label_permutation.py:evaluate"}
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in [None, *range(args.seed_start, args.seed_start + args.count)]:
        name = "original" if seed is None else f"seed_{seed}"
        directory = args.output_root / name
        directory.mkdir(exist_ok=True)
        labels = original.copy()
        if seed is not None:
            labels.iloc[:] = np.random.RandomState(seed).permutation(original.to_numpy())
        digest = hashlib.sha256(labels.to_csv().encode()).hexdigest()
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["label_hash"] != digest or summary["config"] != config:
                raise ValueError(f"Incompatible saved run: {directory}")
        else:
            scores, predictions = evaluate(significant, matrix, labels, [0.46], config, args.n_jobs)
            if len(predictions) != len(labels) or predictions.patient.nunique() != len(labels):
                raise ValueError(f"Incomplete LOOCV predictions for {name}")
            if not (predictions.loc[predictions.no_features, "p_CLR"] == 0.5).all():
                raise ValueError(f"Empty folds not assigned p_NN=0.5 for {name}")
            summary = {"seed": seed, "fixed_auc": float(scores.auc.iloc[0]),
                       "no_feature_folds": int(scores.no_feature_folds.iloc[0]),
                       "n_patients": len(labels), "label_hash": digest, "config": config}
            labels.rename("class_code").to_csv(directory / "patient_labels.csv")
            predictions.assign(true_class=predictions.true_class.map({"CLR": "NN", "DII": "NP"}),
                               p_NN=predictions.p_CLR).drop(columns="p_CLR").to_csv(
                                   directory / "loocv_predictions.csv", index=False)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        records.append({k: v for k, v in summary.items() if k not in ("config", "label_hash")})
        print(f"{name}: AUC={summary['fixed_auc']:.4f}, empty folds={summary['no_feature_folds']}", flush=True)
    pd.DataFrame(records).to_csv(args.output_root / "all_fixed_046_aucs.csv", index=False)


if __name__ == "__main__":
    main()
