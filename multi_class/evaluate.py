from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from cism import build_patient_motif_matrix
from cism.cism import DiscriminativeMotifs

from .config import RANDOM_STATE


def evaluate_multiclass_panel(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    selected_motifs: pd.DataFrame,
    labels: tuple[str, ...] | list[str],
    output_dir: Path,
    *,
    force_evaluation: bool = False,
    random_state: int = RANDOM_STATE,
    max_patients_per_class: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "loocv_predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and not force_evaluation:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    local_classes = patient_classes[patient_classes[DiscriminativeMotifs.PATIENT_CLASS].isin(labels)].copy()
    if max_patients_per_class is not None:
        local_classes = (
            local_classes.groupby(DiscriminativeMotifs.PATIENT_CLASS, group_keys=False, observed=True)
            .head(max_patients_per_class)
            .copy()
        )
    motif_ids = selected_motifs["ID"].drop_duplicates().tolist()
    if not motif_ids:
        raise ValueError("Cannot evaluate an empty motif panel.")

    patient_ids = list(local_classes.index.astype(str))
    all_motifs = motifs_df[motifs_df["Patient_uId"].isin(patient_ids)]
    matrix = build_patient_motif_matrix(all_motifs, patient_ids=patient_ids).reindex(columns=motif_ids, fill_value=0.0)
    y = local_classes.loc[patient_ids, DiscriminativeMotifs.PATIENT_CLASS]

    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    labels = tuple(labels)

    for held_out in tqdm(patient_ids, desc="Multi-class RF LOOCV", unit="patient", leave=False):
        train_patients = [patient for patient in patient_ids if patient != held_out]
        x_train = matrix.loc[train_patients]
        x_test = matrix.loc[[held_out]]
        y_train = y.loc[train_patients]
        if y_train.nunique() < 2:
            continue

        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)[0]
        probabilities = clf.predict_proba(x_test)[0]
        row: dict[str, Any] = {
            "patient_id": held_out,
            "true_class": y.loc[held_out],
            "predicted_class": pred,
        }
        for label in labels:
            class_index = np.where(clf.classes_ == label)[0]
            row[f"prob_{label}"] = float(probabilities[class_index[0]]) if len(class_index) else 0.0
        prediction_rows.append(row)
        importance_rows.append(
            pd.DataFrame(
                {
                    "ID": motif_ids,
                    "fold_patient_id": held_out,
                    "importance": clf.feature_importances_,
                }
            )
        )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError("No valid LOOCV folds were evaluated.")

    metrics = compute_multiclass_metrics(predictions, labels)
    metrics["n_patients"] = int(len(predictions))
    metrics["n_motifs"] = int(len(motif_ids))
    metrics["labels"] = list(labels)

    confusion = pd.DataFrame(
        confusion_matrix(predictions["true_class"], predictions["predicted_class"], labels=list(labels)),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    feature_importance = summarize_feature_importance(importance_rows, selected_motifs)

    predictions.to_csv(predictions_path, index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv")
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def compute_multiclass_metrics(predictions: pd.DataFrame, labels: tuple[str, ...]) -> dict[str, Any]:
    y_true = predictions["true_class"]
    y_pred = predictions["predicted_class"]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, labels=list(labels), average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
    }
    probability_columns = [f"prob_{label}" for label in labels]
    y_score = predictions[probability_columns].to_numpy()
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, y_score, labels=list(labels), multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    for index, label in enumerate(labels):
        binary_true = (y_true == label).astype(int)
        try:
            metrics[f"roc_auc_ovr_{label}"] = float(roc_auc_score(binary_true, y_score[:, index]))
        except ValueError:
            metrics[f"roc_auc_ovr_{label}"] = float("nan")
    return metrics


def summarize_feature_importance(
    importance_rows: list[pd.DataFrame],
    selected_motifs: pd.DataFrame,
) -> pd.DataFrame:
    importances = pd.concat(importance_rows, ignore_index=True)
    summary = (
        importances.groupby("ID", observed=True)
        .agg(
            mean_importance=("importance", "mean"),
            median_importance=("importance", "median"),
            max_importance=("importance", "max"),
            nonzero_fold_count=("importance", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    merged = summary.merge(selected_motifs, on="ID", how="left")
    return merged.sort_values(["mean_importance", "max_importance"], ascending=False).reset_index(drop=True)
