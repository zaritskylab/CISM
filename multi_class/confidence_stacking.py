from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from cism import build_patient_motif_matrix
from cism.cism import DiscriminativeMotifs

from .config import MAX_CLASS_FEATURES, P_VALUE_THRESHOLD, RANDOM_STATE, STRINGENCY_GRID
from .evaluate import compute_multiclass_metrics
from .motif_panels import (
    Comparison,
    _classes_for_comparison,
    make_comparisons,
    select_full_comparison_motifs,
    tune_and_select_panel,
)


@dataclass(frozen=True)
class ScoringFold:
    comparison: str
    fold_type: str
    train_patients: tuple[str, ...]
    score_patients: tuple[str, ...]


def make_confidence_scoring_folds(
    patient_classes: pd.DataFrame,
    comparison: Comparison,
) -> list[ScoringFold]:
    """Create folds without synthetic scores or not-applicable defaults."""
    comparison_classes = _classes_for_comparison(patient_classes, comparison)
    comparison_patients = tuple(comparison_classes.index.astype(str))
    folds = [
        ScoringFold(
            comparison=comparison.name,
            fold_type="binary_loocv",
            train_patients=tuple(patient for patient in comparison_patients if patient != held_out),
            score_patients=(held_out,),
        )
        for held_out in comparison_patients
    ]

    if comparison.kind == "one_vs_one_union":
        excluded = tuple(
            patient_classes.index[
                ~patient_classes[DiscriminativeMotifs.PATIENT_CLASS].isin(comparison.source_labels)
            ].astype(str)
        )
        if excluded:
            folds.append(
                ScoringFold(
                    comparison=comparison.name,
                    fold_type="full_pair_to_excluded_class",
                    train_patients=comparison_patients,
                    score_patients=excluded,
                )
            )
    return folds


def run_confidence_stack(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    labels: tuple[str, ...] | list[str],
    method: str,
    output_dir: Path,
    *,
    force_tuning: bool = False,
    force_evaluation: bool = False,
    max_class_features: int = MAX_CLASS_FEATURES,
    p_value_threshold: float = P_VALUE_THRESHOLD,
    random_state: int = RANDOM_STATE,
    max_patients_per_class: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "loocv_predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and not force_evaluation:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    labels = tuple(labels)
    local_classes = patient_classes[patient_classes[DiscriminativeMotifs.PATIENT_CLASS].isin(labels)].copy()
    if max_patients_per_class is not None:
        local_classes = (
            local_classes.groupby(DiscriminativeMotifs.PATIENT_CLASS, group_keys=False, observed=True)
            .head(max_patients_per_class)
            .copy()
        )
    local_classes.index = local_classes.index.astype(str)

    source_method = _source_method(method)
    source_dir = output_dir.parent / source_method
    tune_and_select_panel(
        motifs_df,
        local_classes,
        labels,
        source_method,
        source_dir,
        force_tuning=force_tuning,
        max_class_features=max_class_features,
        p_value_threshold=p_value_threshold,
        random_state=random_state,
        max_patients_per_class=None,
    )
    tuning = pd.read_csv(source_dir / "comparison_tuning.csv").set_index("comparison")
    comparisons = make_comparisons(labels, source_method)

    features, fold_details = build_binary_confidence_features(
        motifs_df,
        local_classes,
        comparisons,
        tuning,
        max_class_features=max_class_features,
        p_value_threshold=p_value_threshold,
        random_state=random_state,
    )
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing binary confidence scores are not allowed: {missing}")

    metrics, predictions, coefficients = evaluate_confidence_meta_model(
        features,
        local_classes,
        labels,
        random_state=random_state,
    )
    metrics.update(
        {
            "method": method,
            "source_method": source_method,
            "n_patients": int(len(predictions)),
            "n_confidence_features": int(features.shape[1]),
            "n_motifs": 0,
            "labels": list(labels),
            "fallback_fold_count": int(fold_details["used_stringency_fallback"].sum()),
            "fully_nested": False,
            "stringency_source": f"reused from {source_method}/comparison_tuning.csv",
        }
    )

    confusion = pd.DataFrame(
        confusion_matrix(predictions["true_class"], predictions["predicted_class"], labels=list(labels)),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    importance = (
        coefficients.groupby("feature", observed=True)["abs_coefficient"]
        .agg(mean_importance="mean", max_importance="max")
        .reset_index()
        .rename(columns={"feature": "ID"})
        .sort_values("mean_importance", ascending=False)
    )

    features.rename_axis("patient_id").reset_index().to_csv(output_dir / "confidence_features.csv", index=False)
    fold_details.to_csv(output_dir / "base_classifier_folds.csv", index=False)
    predictions.to_csv(predictions_path, index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv")
    coefficients.to_csv(output_dir / "meta_coefficients.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "ID": features.columns,
            "comparison": [column.removeprefix("confidence_") for column in features.columns],
            "source_method": source_method,
        }
    ).to_csv(output_dir / "selected_motifs.csv", index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stacking_summary.json").write_text(
        json.dumps(
            {
                "method": method,
                "source_method": source_method,
                "confidence_features": list(features.columns),
                "base_score_protocol": (
                    "one-vs-all binary LOOCV"
                    if source_method == "one_vs_all_union"
                    else "pair-member binary LOOCV plus full-pair scoring of excluded-class patients"
                ),
                "missing_score_policy": "error; no default values are inserted",
                "meta_classifier": "L2-regularized multinomial logistic regression",
                "evaluation_protocol": "cross-fitted base scores followed by meta-classifier LOOCV",
                "fully_nested": False,
                "nesting_note": (
                    "Comparison stringencies are reused from the prior full-cohort binary analyses. "
                    "A fully nested performance estimate would reselect stringency inside every outer fold."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def build_binary_confidence_features(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    comparisons: list[Comparison],
    tuning: pd.DataFrame,
    *,
    max_class_features: int,
    p_value_threshold: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_ids = list(patient_classes.index.astype(str))
    features = pd.DataFrame(index=patient_ids, dtype=float)
    detail_rows: list[dict[str, Any]] = []

    for comparison in tqdm(comparisons, desc="Binary confidence features", unit="comparison", leave=False):
        requested_stringency = float(tuning.loc[comparison.name, "stringency"])
        column = f"confidence_{comparison.name}_{comparison.positive_label}"
        features[column] = np.nan
        comparison_classes = _classes_for_comparison(patient_classes, comparison)

        for fold in tqdm(
            make_confidence_scoring_folds(patient_classes, comparison),
            desc=f"{comparison.name}: confidence folds",
            unit="fold",
            leave=False,
        ):
            train_classes = comparison_classes.loc[list(fold.train_patients)]
            score_ids = list(fold.score_patients)
            selected, used_stringency = _select_with_training_only_fallback(
                motifs_df[motifs_df["Patient_uId"].isin(fold.train_patients)],
                train_classes,
                comparison,
                requested_stringency,
                max_class_features=max_class_features,
                p_value_threshold=p_value_threshold,
            )
            motif_ids = selected["ID"].drop_duplicates().tolist()
            probabilities = _fit_binary_and_score(
                motifs_df,
                train_classes,
                score_ids,
                motif_ids,
                comparison.positive_label,
                random_state=random_state,
            )
            features.loc[score_ids, column] = probabilities
            for patient_id, probability in zip(score_ids, probabilities):
                detail_rows.append(
                    {
                        "comparison": comparison.name,
                        "positive_label": comparison.positive_label,
                        "patient_id": patient_id,
                        "patient_class": patient_classes.loc[patient_id, DiscriminativeMotifs.PATIENT_CLASS],
                        "fold_type": fold.fold_type,
                        "train_patient_count": len(fold.train_patients),
                        "requested_stringency": requested_stringency,
                        "used_stringency": used_stringency,
                        "used_stringency_fallback": used_stringency != requested_stringency,
                        "selected_motif_count": len(motif_ids),
                        "confidence": probability,
                    }
                )
    return features, pd.DataFrame(detail_rows)


def evaluate_confidence_meta_model(
    features: pd.DataFrame,
    patient_classes: pd.DataFrame,
    labels: tuple[str, ...],
    *,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = patient_classes.loc[features.index, DiscriminativeMotifs.PATIENT_CLASS]
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for held_out in tqdm(features.index, desc="Confidence-stack LOOCV", unit="patient", leave=False):
        train_ids = features.index[features.index != held_out]
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
        )
        clf.fit(features.loc[train_ids], y.loc[train_ids])
        probabilities = clf.predict_proba(features.loc[[held_out]])[0]
        predicted = clf.classes_[int(np.argmax(probabilities))]
        row: dict[str, Any] = {
            "patient_id": held_out,
            "true_class": y.loc[held_out],
            "predicted_class": predicted,
        }
        for label in labels:
            class_index = np.where(clf.classes_ == label)[0]
            row[f"prob_{label}"] = float(probabilities[class_index[0]]) if len(class_index) else 0.0
        prediction_rows.append(row)
        for class_index, label in enumerate(clf.classes_):
            for feature_index, feature in enumerate(features.columns):
                coefficient = float(clf.coef_[class_index, feature_index])
                coefficient_rows.append(
                    {
                        "fold_patient_id": held_out,
                        "class": label,
                        "feature": feature,
                        "coefficient": coefficient,
                        "abs_coefficient": abs(coefficient),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    return compute_multiclass_metrics(predictions, labels), predictions, pd.DataFrame(coefficient_rows)


def _select_with_training_only_fallback(
    training_motifs: pd.DataFrame,
    training_classes: pd.DataFrame,
    comparison: Comparison,
    requested_stringency: float,
    *,
    max_class_features: int,
    p_value_threshold: float,
) -> tuple[pd.DataFrame, float]:
    candidates = [requested_stringency]
    candidates.extend(sorted((value for value in STRINGENCY_GRID if value < requested_stringency), reverse=True))
    candidates.append(0.0)
    for stringency in dict.fromkeys(candidates):
        selected = select_full_comparison_motifs(
            training_motifs,
            training_classes,
            comparison=comparison,
            stringency=float(stringency),
            max_class_features=max_class_features,
            p_value_threshold=p_value_threshold,
        )
        if not selected.empty:
            return selected, float(stringency)
    raise ValueError(
        f"No discriminative motifs were available for {comparison.name}, even at stringency 0. "
        "The confidence score cannot be generated without inserting a synthetic default."
    )


def _fit_binary_and_score(
    motifs_df: pd.DataFrame,
    training_classes: pd.DataFrame,
    score_ids: list[str],
    motif_ids: list[Any],
    positive_label: str,
    *,
    random_state: int,
) -> np.ndarray:
    train_ids = list(training_classes.index.astype(str))
    x_train = build_patient_motif_matrix(
        motifs_df[motifs_df["Patient_uId"].isin(train_ids)], patient_ids=train_ids
    ).reindex(columns=motif_ids, fill_value=0.0)
    x_score = build_patient_motif_matrix(
        motifs_df[motifs_df["Patient_uId"].isin(score_ids)], patient_ids=score_ids
    ).reindex(columns=motif_ids, fill_value=0.0)
    y_train = training_classes.loc[train_ids, DiscriminativeMotifs.PATIENT_CLASS]
    if y_train.nunique() != 2:
        raise ValueError(f"Binary confidence model requires two training classes; found {sorted(y_train.unique())}")

    clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    clf.fit(x_train, y_train)
    class_index = np.where(clf.classes_ == positive_label)[0]
    if len(class_index) != 1:
        raise ValueError(f"Positive label {positive_label!r} is absent from the fitted binary classifier")
    return clf.predict_proba(x_score)[:, class_index[0]].astype(float)


def _source_method(method: str) -> str:
    if method == "one_vs_all_confidence_stack":
        return "one_vs_all_union"
    if method == "one_vs_one_third_class_confidence_stack":
        return "one_vs_one_union"
    raise ValueError(f"Unsupported confidence-stacking method: {method}")
