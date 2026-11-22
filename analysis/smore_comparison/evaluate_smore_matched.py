#!/usr/bin/env python3
"""Evaluate the complete, matched 38-patient SMORE discovery run."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


ROOT = _REPO_ROOT
OUTPUT = ROOT / "analysis/outputs/smore_matched"
RUN_TAG = "p0.001_n0_train50_score10"
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50)
LABELS = ("NN", "NP")


def canonical_consensus(value: str) -> str:
    parts = tuple(value.split("-"))
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(f"Expected a four-cell consensus, got {value!r}")
    reverse = parts[::-1]
    return "-".join(min(parts, reverse))


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"patient", "class", "fov"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest missing {required - set(manifest.columns)}")
    if len(manifest) != 76 or manifest.fov.nunique() != 76:
        raise ValueError("Expected exactly 76 distinct FOVs")
    patients = manifest[["patient", "class"]].drop_duplicates()
    if patients.patient.nunique() != 38:
        raise ValueError("Expected exactly 38 distinct patients")
    if patients.groupby("class").patient.nunique().to_dict() != {"NN": 20, "NP": 18}:
        raise ValueError("Expected 20 NN and 18 NP patients")
    if manifest.groupby("patient")["class"].nunique().max() != 1:
        raise ValueError("A patient has conflicting class labels")
    return patients.sort_values("patient")


def check_outputs(manifest: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        mat = run_dir / f"{row.fov}.mat"
        catalog = run_dir / f"{row.fov}_motifs.csv"
        metadata = run_dir / f"{row.fov}_metadata.csv"
        rows.append({"fov": row.fov, "mat": mat.is_file(),
                     "catalog": catalog.is_file(), "metadata": metadata.is_file()})
    return pd.DataFrame(rows)


def load_catalog(
    manifest: pd.DataFrame, run_dir: Path, path_probability: float = 0.001,
    n_train: int = 50, n_score: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalogs = []
    denominators = []
    for row in manifest.itertuples(index=False):
        catalog = pd.read_csv(run_dir / f"{row.fov}_motifs.csv")
        if len(catalog):
            if not (catalog.fov == row.fov).all():
                raise ValueError(f"FOV mismatch in {row.fov} catalog")
            if not np.isclose(catalog.path_probability.to_numpy(dtype=float), path_probability).all():
                raise ValueError(f"Unexpected sampling probability for {row.fov}")
            if not ((catalog.n_train == n_train) & (catalog.n_score == n_score)).all():
                raise ValueError(f"Unexpected SMORE parameters for {row.fov}")
        meta_file = run_dir / f"{row.fov}_metadata.csv"
        if meta_file.is_file():
            meta = pd.read_csv(meta_file)
            if len(meta) != 1 or meta.fov.iloc[0] != row.fov:
                raise ValueError(f"Invalid metadata for {row.fov}")
            denominator = int(meta.sampled_radial_paths.iloc[0])
        elif len(catalog):
            # The first submitted tasks predate the separate metadata file.
            denominator = int(catalog.sampled_radial_paths.iloc[0])
        else:
            raise ValueError(f"{row.fov} has no motifs and no path-count metadata")
        if denominator <= 0:
            raise ValueError(f"No sampled paths for {row.fov}")
        if len(catalog) and not (catalog.sampled_radial_paths == denominator).all():
            raise ValueError(f"Inconsistent path counts for {row.fov}")
        denominators.append({"patient": row.patient, "fov": row.fov,
                             "sampled_radial_paths": denominator})
        if len(catalog):
            catalog["patient"] = row.patient
            catalog["class"] = manifest.loc[manifest.fov == row.fov, "class"].iloc[0]
            catalogs.append(catalog)
    if not catalogs:
        raise ValueError("No SMORE motifs were discovered")
    return pd.concat(catalogs, ignore_index=True), pd.DataFrame(denominators)


def patient_features(catalog: pd.DataFrame, denominators: pd.DataFrame) -> pd.DataFrame:
    significant = catalog[catalog.p_value < 0.05].copy()
    if significant.empty:
        raise ValueError("No statistically significant SMORE motifs")
    significant["feature"] = significant.consensus_cell_types.map(canonical_consensus)
    counts = (significant.groupby(["patient", "feature"], observed=True)
              .observed_znic_count.sum().rename("observed_count"))
    patient_paths = denominators.groupby("patient").sampled_radial_paths.sum()
    features = counts.reset_index()
    features["frequency"] = features.observed_count.to_numpy() / features.patient.map(patient_paths).to_numpy()
    return features


def select_train_features(
    train: pd.DataFrame, labels: pd.Series, threshold: float, max_per_class: int
) -> list[str]:
    present = train[train.frequency > 0].merge(
        labels.rename("class"), left_on="patient", right_index=True, how="inner")
    if present.empty:
        return []
    summary = (present.groupby(["feature", "class"], observed=True)
               .agg(n_patients=("patient", "nunique"), mean_frequency=("frequency", "mean"))
               .reset_index())
    exclusive = summary.groupby("feature")["class"].nunique()
    summary = summary[summary.feature.isin(exclusive[exclusive == 1].index)].copy()
    class_sizes = labels.value_counts()
    summary["prevalence"] = summary.n_patients / summary["class"].map(class_sizes)
    summary = summary[summary.prevalence >= threshold]
    summary = summary.sort_values(
        ["class", "prevalence", "n_patients", "mean_frequency", "feature"],
        ascending=[True, False, False, False, True],
    )
    selected = []
    for label in LABELS:
        selected.extend(summary.loc[summary["class"] == label, "feature"].head(max_per_class))
    return selected


def evaluate(
    features: pd.DataFrame, patients: pd.DataFrame, thresholds: tuple[float, ...],
    max_per_class: int, max_failed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = patients.set_index("patient")["class"]
    pivot = features.pivot_table(index="patient", columns="feature", values="frequency", fill_value=0)
    pivot = pivot.reindex(index=labels.index, fill_value=0)
    fold_rows = []
    summary_rows = []
    for threshold in thresholds:
        current = []
        for patient, true_label in labels.items():
            train_labels = labels.drop(patient)
            train_features = features[features.patient.isin(train_labels.index)]
            selected = select_train_features(train_features, train_labels, threshold, max_per_class)
            failed = not selected
            no_test_motifs = bool(selected) and not pivot.loc[patient, selected].gt(0).any()
            probability = np.nan
            if not failed:
                model = RandomForestClassifier(random_state=0, n_jobs=1)
                model.fit(pivot.loc[train_labels.index, selected].to_numpy(), train_labels.to_numpy())
                probability = float(model.predict_proba(pivot.loc[[patient], selected].to_numpy())[0,
                                  list(model.classes_).index("NN")])
            current.append({"stringency": threshold, "patient": patient, "true_class": true_label,
                            "failed": failed, "no_test_motifs": no_test_motifs,
                            "selected_features": len(selected),
                            "probability_NN": probability})
        failed_count = sum(row["failed"] for row in current)
        for row in current:
            row["probability_NN_equal"] = 0.5 if row["failed"] else row["probability_NN"]
            row["probability_NN_absent_neutral"] = (
                0.5 if row["failed"] or row["no_test_motifs"] else row["probability_NN"]
            )
            # Deliberately pessimistic scoring policy, not a deployable classifier.
            row["probability_NN_false"] = (0.0 if row["true_class"] == "NN" else 1.0) if row["failed"] else row["probability_NN"]
        folds = pd.DataFrame(current)
        y = (folds.true_class == "NN").astype(int)
        summary_rows.append({
            "stringency": threshold,
            "auc_equal": roc_auc_score(y, folds.probability_NN_equal),
            "auc_absent_neutral": roc_auc_score(y, folds.probability_NN_absent_neutral),
            "auc_forced_false": roc_auc_score(y, folds.probability_NN_false),
            "auc_reportable": failed_count <= max_failed,
            "failed_folds": failed_count,
            "no_test_motifs": int(folds.no_test_motifs.sum()),
            "no_motifs_to_classify": int((folds.failed | folds.no_test_motifs).sum()),
            "valid_folds": len(folds) - failed_count,
            "mean_selected_features": folds.selected_features.mean(),
            "max_features_per_class": max_per_class,
            "feature_identity": "reverse-canonicalized significant SMORE 4-cell PWM consensus",
        })
        fold_rows.extend(current)
        if failed_count > max_failed:
            break
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--path-probability", type=float, default=0.001)
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-score", type=int, default=10)
    parser.add_argument("--status", action="store_true", help="Only report completed FOV outputs")
    parser.add_argument("--max-features-per-class", type=int, default=30)
    parser.add_argument("--max-failed-folds", type=int, default=5)
    args = parser.parse_args()
    manifest = pd.read_csv(args.output_dir / "cohort_fovs.csv")
    patients = validate_manifest(manifest)
    run_dir = args.output_dir / "runs" / args.run_tag
    status = check_outputs(manifest, run_dir)
    complete = status.mat & status.catalog
    print(f"Complete FOVs: {complete.sum()}/76; patients: {len(patients)} (20 NN, 18 NP)")
    if args.status:
        print(status.loc[~complete, "fov"].to_string(index=False))
        return
    if not complete.all():
        raise RuntimeError("SMORE extraction incomplete; AUC withheld. Use --status to inspect missing FOVs.")
    catalog, denominators = load_catalog(manifest, run_dir, args.path_probability,
                                         args.n_train, args.n_score)
    features = patient_features(catalog, denominators)
    summary, folds = evaluate(features, patients, THRESHOLDS,
                              args.max_features_per_class, args.max_failed_folds)
    summary.to_csv(run_dir / "smore_matched_auc.csv", index=False)
    folds.to_csv(run_dir / "smore_matched_fold_predictions.csv", index=False)
    features.to_csv(run_dir / "smore_matched_patient_features.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
