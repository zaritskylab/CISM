#!/usr/bin/env python3
"""Bootstrap imbalanced NN/NP cohorts and measure RF LOOCV AUC.

For each label, this script samples 10%, 20%, ..., 90% of that label's
patients, keeps the complete opposite label, and tunes hard-discriminative
patient-prevalence stringency with 10 Optuna trials and leakage-safe Random
Forest LOOCV. Motifs are re-extracted inside every LOOCV fold using only that
fold's training patients.
"""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm.auto import tqdm

SCRIPT_DIR = ANALYSIS_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.cism import DiscriminativeMotifs  # noqa: E402
from cism.motif_selection import build_patient_motif_matrix  # noqa: E402


DEFAULT_ARTIFACT_PATH = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def build_framework_random_forest(random_state: int) -> RandomForestClassifier:
    """Match the RandomForestClassifier used by the CISM notebook/framework path."""
    return RandomForestClassifier(random_state=random_state, n_jobs=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test how NN/NP patient imbalance affects discriminative-motif RF LOOCV AUC."
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Serialized CISM artifact from tutorial/notebook 02.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/JSON outputs will be saved.",
    )
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"], help="Labels to compare.")
    parser.add_argument(
        "--bootstrap-fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        help="Fractions of each sampled label to keep; sample size is rounded up.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=10,
        help="Number of bootstrap repeats per label/fraction. Change this for larger runs.",
    )
    parser.add_argument(
        "--n-optuna-trials",
        type=int,
        default=10,
        help="Number of Optuna stringency trials per bootstrap.",
    )
    parser.add_argument(
        "--stringency-range",
        nargs=2,
        type=float,
        default=(0.10, 0.95),
        metavar=("LOW", "HIGH"),
        help="Optuna search range for patient-prevalence stringency.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30,
        help="Maximum number of discriminative motifs used at each stringency.",
    )
    parser.add_argument(
        "--fanmod-p-value-threshold",
        type=float,
        default=0.05,
        help="Motif p-value threshold before discriminative motif discovery.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_patient_classes(discriminator: Any, labels: list[str]) -> pd.DataFrame:
    classes = discriminator.get_patients_class(labels)
    classes = classes[classes[DiscriminativeMotifs.PATIENT_CLASS].isin(labels)].copy()
    classes = classes.loc[~classes.index.duplicated(keep="first")]
    missing = sorted(set(labels) - set(classes[DiscriminativeMotifs.PATIENT_CLASS].unique()))
    if missing:
        raise ValueError(f"Missing labels in artifact patient classes: {missing}")
    return classes


def sample_patients(
    patient_classes: pd.DataFrame,
    sampled_label: str,
    other_label: str,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[list[Any], list[Any], list[Any]]:
    sampled_pool = patient_classes.index[
        patient_classes[DiscriminativeMotifs.PATIENT_CLASS] == sampled_label
    ].tolist()
    other_patients = patient_classes.index[
        patient_classes[DiscriminativeMotifs.PATIENT_CLASS] == other_label
    ].tolist()

    sample_size = max(1, int(math.ceil(len(sampled_pool) * fraction)))
    sampled_patients = rng.choice(sampled_pool, size=sample_size, replace=False).tolist()
    cohort_patients = sampled_patients + other_patients
    return sampled_patients, other_patients, cohort_patients


def discover_hard_discriminative_motifs(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    cohort_patients: list[Any],
    p_value_threshold: float,
) -> pd.DataFrame:
    """Return motifs present in exactly one class within the given cohort."""
    local_classes = patient_classes.loc[cohort_patients]
    local_motifs = motifs_df[motifs_df["Patient_uId"].isin(cohort_patients)].copy()
    local_motifs = local_motifs[local_motifs["p_value"] < p_value_threshold].copy()
    local_motifs[DiscriminativeMotifs.PATIENT_CLASS] = local_motifs["Patient_uId"].map(
        local_classes[DiscriminativeMotifs.PATIENT_CLASS]
    )

    motif_class_counts = (
        local_motifs.drop_duplicates(["ID", DiscriminativeMotifs.PATIENT_CLASS])
        .groupby("ID", observed=True)[DiscriminativeMotifs.PATIENT_CLASS]
        .nunique()
    )
    one_class_ids = motif_class_counts[motif_class_counts == 1].index
    one_class = local_motifs[local_motifs["ID"].isin(one_class_ids)].copy()

    if one_class.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                DiscriminativeMotifs.PATIENT_CLASS,
                "patient_count",
                "patient_percentage",
                "mean_freq",
                "total_count",
            ]
        )

    summary = (
        one_class.groupby(["ID", DiscriminativeMotifs.PATIENT_CLASS], observed=True)
        .agg(
            patient_count=("Patient_uId", "nunique"),
            mean_freq=("Freq", "mean"),
            total_count=("Count", "sum"),
        )
        .reset_index()
    )
    class_counts = local_classes[DiscriminativeMotifs.PATIENT_CLASS].value_counts()
    summary["patient_percentage"] = summary.apply(
        lambda row: row["patient_count"] / class_counts[row[DiscriminativeMotifs.PATIENT_CLASS]],
        axis=1,
    )
    return summary.sort_values(
        [DiscriminativeMotifs.PATIENT_CLASS, "patient_percentage", "patient_count", "mean_freq", "total_count"],
        ascending=False,
    )


def select_motifs_by_label(
    discriminative_summary: pd.DataFrame,
    labels: list[str],
    stringency: float,
    max_features: int,
) -> dict[str, list[Any]]:
    selected_by_label: dict[str, list[Any]] = {}
    for label in labels:
        label_summary = discriminative_summary[
            (discriminative_summary[DiscriminativeMotifs.PATIENT_CLASS] == label)
            & (discriminative_summary["patient_percentage"] >= stringency)
        ]
        selected_by_label[label] = label_summary["ID"].head(max_features).tolist()
    return selected_by_label


def evaluate_hard_discriminative_loocv(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    cohort_patients: list[Any],
    sampled_label: str,
    labels_to_compare: list[str],
    stringency: float,
    max_features: int,
    p_value_threshold: float,
    random_state: int,
) -> dict[str, float]:
    local_classes = patient_classes.loc[cohort_patients]
    labels = local_classes[DiscriminativeMotifs.PATIENT_CLASS]

    true_labels: list[str] = []
    pred_labels: list[str] = []
    positive_probabilities: list[float] = []
    selected_counts: list[int] = []
    selected_union_by_label: dict[str, set[Any]] = {label: set() for label in labels_to_compare}
    skipped_folds = 0

    for held_out_patient in cohort_patients:
        train_patients = [patient for patient in cohort_patients if patient != held_out_patient]
        y_train = labels.loc[train_patients]
        y_test = labels.loc[held_out_patient]

        if len(set(y_train)) < 2:
            skipped_folds += 1
            continue

        train_discriminative_summary = discover_hard_discriminative_motifs(
            motifs_df=motifs_df,
            patient_classes=patient_classes,
            cohort_patients=train_patients,
            p_value_threshold=p_value_threshold,
        )
        selected_by_label = select_motifs_by_label(
            discriminative_summary=train_discriminative_summary,
            labels=labels_to_compare,
            stringency=stringency,
            max_features=max_features,
        )
        for label, motif_ids in selected_by_label.items():
            selected_union_by_label[label].update(motif_ids)
        selected_motifs = list(
            dict.fromkeys(motif_id for motif_ids in selected_by_label.values() for motif_id in motif_ids)
        )
        selected_counts.append(len(selected_motifs))
        if not selected_motifs:
            skipped_folds += 1
            continue

        train_motifs = motifs_df[motifs_df["Patient_uId"].isin(train_patients)]
        test_motifs = motifs_df[motifs_df["Patient_uId"] == held_out_patient]
        x_train = build_patient_motif_matrix(train_motifs, patient_ids=train_patients).reindex(
            columns=selected_motifs, fill_value=0.0
        )
        x_test = build_patient_motif_matrix(test_motifs, patient_ids=[held_out_patient]).reindex(
            columns=selected_motifs, fill_value=0.0
        )

        clf = build_framework_random_forest(random_state=random_state)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)[0]
        prob = clf.predict_proba(x_test)[0]
        class_index = np.where(clf.classes_ == sampled_label)[0]
        if len(class_index) == 0:
            skipped_folds += 1
            continue

        true_labels.append(y_test)
        pred_labels.append(pred)
        positive_probabilities.append(float(prob[class_index[0]]))

    metrics = {
        "roc_auc": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "valid_folds": len(true_labels),
        "skipped_folds": skipped_folds,
        "mean_selected_motifs": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "median_selected_motifs": float(np.median(selected_counts)) if selected_counts else 0.0,
        "min_selected_motifs": int(np.min(selected_counts)) if selected_counts else 0,
        "max_selected_motifs": int(np.max(selected_counts)) if selected_counts else 0,
    }
    for label in labels_to_compare:
        metrics[f"fold_union_discriminative_motifs_{label}"] = len(selected_union_by_label[label])
    if true_labels:
        metrics["accuracy"] = float(accuracy_score(true_labels, pred_labels))
    if len(set(true_labels)) == 2:
        binary_true = [1 if label == sampled_label else 0 for label in true_labels]
        metrics["roc_auc"] = float(roc_auc_score(binary_true, positive_probabilities))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(true_labels, pred_labels))
    return metrics


def tune_stringency(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    cohort_patients: list[Any],
    sampled_label: str,
    labels_to_compare: list[str],
    stringency_range: tuple[float, float],
    n_trials: int,
    max_features: int,
    p_value_threshold: float,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for per-bootstrap stringency tuning. "
            "Install it with 'pip install optuna'."
        ) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    low, high = float(stringency_range[0]), float(stringency_range[1])
    if low < 0 or high <= low or high > 1:
        raise ValueError("--stringency-range must be an increasing range within [0, 1].")
    if n_trials <= 0:
        raise ValueError("--n-optuna-trials must be positive.")

    rows: list[dict[str, Any]] = []

    def objective(trial):
        stringency = trial.suggest_float("stringency", low, high)
        metrics = evaluate_hard_discriminative_loocv(
            motifs_df=motifs_df,
            patient_classes=patient_classes,
            cohort_patients=cohort_patients,
            sampled_label=sampled_label,
            labels_to_compare=labels_to_compare,
            stringency=float(stringency),
            max_features=max_features,
            p_value_threshold=p_value_threshold,
            random_state=random_state,
        )
        row = {"trial_number": trial.number, "stringency": float(stringency), **metrics}
        rows.append(row)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        if pd.isna(metrics["roc_auc"]):
            return 0.0
        return float(metrics["roc_auc"])

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = pd.DataFrame(rows)
    if trials.empty:
        raise ValueError("No Optuna stringency trials were completed.")

    trials["optuna_value"] = trials["roc_auc"].fillna(0.0)
    valid = trials.dropna(subset=["roc_auc"]).copy()
    if valid.empty:
        best = trials.sort_values(["optuna_value", "mean_selected_motifs"], ascending=False).iloc[0].to_dict()
    else:
        best = valid.sort_values(
            ["roc_auc", "balanced_accuracy", "mean_selected_motifs"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
    best["optuna_best_value"] = float(study.best_value)
    return best, trials


def confidence_interval(values: pd.Series) -> pd.Series:
    clean = values.dropna()
    if clean.empty:
        return pd.Series({"mean_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan})
    return pd.Series(
        {
            "mean_auc": float(clean.mean()),
            "ci_low": float(clean.quantile(0.025)),
            "ci_high": float(clean.quantile(0.975)),
        }
    )


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.artifact_path)
    discriminator = artifact["discriminator"]
    patient_classes = load_patient_classes(discriminator, args.labels)
    motifs_df = discriminator.cism.motifs_dataset

    rng = np.random.default_rng(args.random_state)
    experiment_rows = []
    optuna_trial_rows = []

    pairs = [(args.labels[0], args.labels[1]), (args.labels[1], args.labels[0])]
    iterator = [
        (sampled_label, other_label, fraction, bootstrap_index)
        for sampled_label, other_label in pairs
        for fraction in args.bootstrap_fractions
        for bootstrap_index in range(args.n_bootstraps)
    ]

    for sampled_label, other_label, fraction, bootstrap_index in tqdm(
        iterator,
        desc="Imbalance bootstraps",
        disable=args.no_progress,
    ):
        sampled_patients, other_patients, cohort_patients = sample_patients(
            patient_classes=patient_classes,
            sampled_label=sampled_label,
            other_label=other_label,
            fraction=fraction,
            rng=rng,
        )
        discriminative_summary = discover_hard_discriminative_motifs(
            motifs_df=motifs_df,
            patient_classes=patient_classes,
            cohort_patients=cohort_patients,
            p_value_threshold=args.fanmod_p_value_threshold,
        )
        best, optuna_trials = tune_stringency(
            motifs_df=motifs_df,
            patient_classes=patient_classes,
            cohort_patients=cohort_patients,
            sampled_label=sampled_label,
            labels_to_compare=list(args.labels),
            stringency_range=tuple(args.stringency_range),
            n_trials=args.n_optuna_trials,
            max_features=args.max_features,
            p_value_threshold=args.fanmod_p_value_threshold,
            random_state=args.random_state + bootstrap_index,
        )
        optimal_by_label = select_motifs_by_label(
            discriminative_summary=discriminative_summary,
            labels=list(args.labels),
            stringency=float(best["stringency"]),
            max_features=args.max_features,
        )
        optimal_discriminative_by_label = {
            label: int(
                (
                    (discriminative_summary[DiscriminativeMotifs.PATIENT_CLASS] == label)
                    & (discriminative_summary["patient_percentage"] >= float(best["stringency"]))
                ).sum()
            )
            for label in args.labels
        }
        optimal_discriminative_motifs = int(sum(optimal_discriminative_by_label.values()))
        optimal_selected_motifs_on_full_bootstrap = int(
            len({motif_id for motif_ids in optimal_by_label.values() for motif_id in motif_ids})
        )

        base_row = {
            "sampled_label": sampled_label,
            "other_label": other_label,
            "bootstrap_fraction": float(fraction),
            "bootstrap_percent": int(round(100 * fraction)),
            "bootstrap_index": bootstrap_index,
            "sampled_n": len(sampled_patients),
            "other_n": len(other_patients),
            "cohort_n": len(cohort_patients),
            "n_discriminative_motifs": int(discriminative_summary.shape[0]),
            "sampled_patients": "|".join(map(str, sampled_patients)),
        }
        for label in args.labels:
            base_row[f"n_discriminative_motifs_{label}"] = int(
                (discriminative_summary[DiscriminativeMotifs.PATIENT_CLASS] == label).sum()
            )
            base_row[f"optimal_full_bootstrap_discriminative_motifs_{label}"] = (
                optimal_discriminative_by_label[label]
            )
            base_row[f"optimal_full_bootstrap_selected_motifs_{label}"] = len(optimal_by_label[label])
        experiment_rows.append(
            {
                **base_row,
                "optimal_stringency": best["stringency"],
                "optimal_auc": best["roc_auc"],
                "optimal_accuracy": best["accuracy"],
                "optimal_balanced_accuracy": best["balanced_accuracy"],
                "optimal_mean_selected_motifs": best["mean_selected_motifs"],
                "optimal_median_selected_motifs": best["median_selected_motifs"],
                "optimal_min_selected_motifs": best["min_selected_motifs"],
                "optimal_max_selected_motifs": best["max_selected_motifs"],
                "optimal_full_bootstrap_discriminative_motifs": optimal_discriminative_motifs,
                "optimal_full_bootstrap_selected_motifs": optimal_selected_motifs_on_full_bootstrap,
                **{
                    f"optimal_fold_union_discriminative_motifs_{label}": best[
                        f"fold_union_discriminative_motifs_{label}"
                    ]
                    for label in args.labels
                },
                "optuna_best_value": best["optuna_best_value"],
                "valid_folds": best["valid_folds"],
                "skipped_folds": best["skipped_folds"],
            }
        )
        for _, trial_row in optuna_trials.iterrows():
            optuna_trial_rows.append({**base_row, **trial_row.to_dict()})

    experiments = pd.DataFrame(experiment_rows)
    optuna_trials = pd.DataFrame(optuna_trial_rows)
    summary = (
        experiments.groupby(
            ["sampled_label", "other_label", "bootstrap_fraction", "bootstrap_percent"],
            observed=True,
        )["optimal_auc"]
        .apply(confidence_interval)
        .unstack()
        .reset_index()
    )
    optimal_stringency_summary = (
        experiments.groupby(
            ["sampled_label", "other_label", "bootstrap_fraction", "bootstrap_percent"],
            observed=True,
        )
        .agg(
            mean_optimal_stringency=("optimal_stringency", "mean"),
            median_optimal_stringency=("optimal_stringency", "median"),
            mean_selected_motifs=("optimal_mean_selected_motifs", "mean"),
            mean_discriminative_motifs=("n_discriminative_motifs", "mean"),
            mean_optimal_full_bootstrap_discriminative_motifs=(
                "optimal_full_bootstrap_discriminative_motifs",
                "mean",
            ),
            n_bootstraps=("bootstrap_index", "count"),
        )
        .reset_index()
    )
    summary = summary.merge(
        optimal_stringency_summary,
        on=["sampled_label", "other_label", "bootstrap_fraction", "bootstrap_percent"],
        how="left",
    )

    experiments.to_csv(output_dir / "imbalanced_bootstrap_results.csv", index=False)
    optuna_trials.to_csv(output_dir / "stringency_optuna_trials.csv", index=False)
    optuna_trials.to_csv(output_dir / "stringency_sweep_results.csv", index=False)
    summary.to_csv(output_dir / "imbalanced_bootstrap_auc_ci.csv", index=False)
    save_json(vars(args), output_dir / "imbalanced_bootstrap_config.json")

    print(f"Saved per-bootstrap results: {output_dir / 'imbalanced_bootstrap_results.csv'}")
    print(f"Saved stringency Optuna trials: {output_dir / 'stringency_optuna_trials.csv'}")
    print(f"Saved AUC confidence intervals: {output_dir / 'imbalanced_bootstrap_auc_ci.csv'}")


if __name__ == "__main__":
    main()
