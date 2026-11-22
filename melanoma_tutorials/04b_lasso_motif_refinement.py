#!/usr/bin/env python3
"""Refine tutorial 03b soft motifs with L1 logistic regression.

This downstream stage starts from ``03b_soft_motif_selection.py`` outputs,
uses leakage-safe leave-one-patient-out validation to identify top motifs with
stable non-zero LASSO coefficients, and then tries lower-ranked soft motifs as
fixed-size replacements.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:
    sns = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.cism import DiscriminativeMotifs  # noqa: E402
from cism.motif_selection import build_patient_motif_matrix  # noqa: E402


if sns is not None:
    sns.set_theme(style="whitegrid")
else:
    plt.style.use("ggplot")


DEFAULT_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
SCORE_COLUMNS = [
    "final_score",
    "additive_score",
    "effect_score",
    "abundance_score",
    "prevalence_score",
    "confidence_score",
    "dispersion_desirability_score",
    "stability",
    "stability_gate",
    "direction",
    "logfc_group1_vs_group0",
    "auc",
    "p_value",
    "q_value",
]


@dataclass
class PanelEvaluation:
    panel_name: str
    motif_ids: list[Any]
    results: pd.DataFrame
    metrics: dict[str, float]
    coefficient_summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    default_runtime = SCRIPT_DIR / "melanoma_tutorials" / "runtime"
    parser = argparse.ArgumentParser(
        description="Prune and refill 03b soft motifs with LOOCV L1 logistic regression."
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=default_runtime / "serialized" / "Melanoma_cism_ready.pkl",
        help="Path to the serialized tutorial 02 artifact.",
    )
    parser.add_argument(
        "--soft-motif-dir",
        type=Path,
        default=default_runtime / "soft_motif_selection",
        help="Directory containing 03b soft motif selection outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_runtime / "lasso_motif_refinement",
        help="Directory where 04b outputs will be saved.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="Number of lower-ranked motifs to try after the original top panel.",
    )
    parser.add_argument(
        "--coefficient-frequency-threshold",
        type=float,
        default=0.25,
        help="Minimum LOOCV non-zero coefficient frequency required to keep a top motif.",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=DEFAULT_C_GRID,
        help="Candidate inverse regularization strengths for L1 logistic regression.",
    )
    parser.add_argument(
        "--acceptance-tolerance",
        type=float,
        default=0.0,
        help="Allowed ROC AUC drop relative to the original panel when accepting replacements.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--skip-asymmetric-stringency",
        action="store_true",
        help="Skip the dominant/minority stringency experiment.",
    )
    parser.add_argument(
        "--asymmetric-max-ranked-motifs",
        type=int,
        default=2000,
        help="Number of ranked 03b motifs to consider for the asymmetric stringency experiment.",
    )
    parser.add_argument(
        "--dominant-min-grid",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        help="Dominant-group minimum patient-prevalence thresholds to tune.",
    )
    parser.add_argument(
        "--minority-max-grid",
        nargs="+",
        type=float,
        default=[0.00, 0.25, 0.50, 0.70, 0.80, 0.85, 0.90],
        help="Minority-group maximum patient-prevalence thresholds to tune.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else str(value)
        for key, value in data.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved image: {path}")


def save_table_image(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    max_rows: int = 20,
    max_cols: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_df = df.head(max_rows).iloc[:, :max_cols].copy()
    if table_df.empty:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.axis("off")
        ax.set_title(title, pad=12, fontsize=13)
        ax.text(0.5, 0.5, "No rows", ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved image: {path}")
        return
    table_df = table_df.reset_index() if table_df.index.name or not isinstance(table_df.index, pd.RangeIndex) else table_df
    table_df = table_df.map(format_cell) if hasattr(table_df, "map") else table_df.applymap(format_cell)

    fig_width = max(8, min(24, 1.6 * len(table_df.columns)))
    fig_height = max(2.5, min(18, 0.45 * (len(table_df) + 2)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=13)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {path}")


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 50 else f"{text[:47]}..."


def load_soft_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if "labels" not in config or len(config["labels"]) != 2:
        raise ValueError(f"{path} must contain exactly two labels.")
    if "top_k" not in config:
        raise ValueError(f"{path} must contain top_k.")
    return config


def read_ranked_motifs(soft_motif_dir: Path) -> tuple[pd.DataFrame, list[Any]]:
    scores = pd.read_csv(soft_motif_dir / "soft_motif_scores.csv")
    selected = pd.read_csv(soft_motif_dir / "selected_soft_motif_ids.csv")["ID"].tolist()
    if "ID" not in scores.columns:
        raise ValueError("soft_motif_scores.csv must include an ID column.")
    scores = scores.drop_duplicates("ID").reset_index(drop=True)
    scores["soft_rank"] = np.arange(1, scores.shape[0] + 1)
    return scores, selected


def build_restricted_matrix(
    artifact_path: Path,
    labels: list[str],
    motif_ids: list[Any],
) -> tuple[pd.DataFrame, pd.Series]:
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)

    discriminator = artifact["discriminator"]
    patient_classes = discriminator.get_patients_class(labels)
    patient_classes = patient_classes[patient_classes[DiscriminativeMotifs.PATIENT_CLASS].isin(labels)].copy()
    patient_classes = patient_classes.loc[~patient_classes.index.duplicated(keep="first")]
    motif_set = set(motif_ids)
    motif_rows = discriminator.cism.motifs_dataset[
        discriminator.cism.motifs_dataset["Patient_uId"].isin(patient_classes.index)
        & discriminator.cism.motifs_dataset["ID"].isin(motif_set)
    ].copy()
    matrix = build_patient_motif_matrix(motif_rows, patient_ids=patient_classes.index)
    matrix = matrix.reindex(index=patient_classes.index, columns=motif_ids, fill_value=0.0)
    labels_series = patient_classes[DiscriminativeMotifs.PATIENT_CLASS].reindex(matrix.index)
    return matrix, labels_series


def motif_ids_from_scores(scores: pd.DataFrame, max_ranked_motifs: int) -> list[Any]:
    if max_ranked_motifs <= 0:
        raise ValueError("--asymmetric-max-ranked-motifs must be positive.")
    return scores.head(max_ranked_motifs)["ID"].tolist()


def tune_lasso_c(
    matrix: pd.DataFrame,
    labels: pd.Series,
    motif_ids: list[Any],
    label_order: list[str],
    c_grid: list[float],
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for c_value in c_grid:
        evaluation = evaluate_panel(
            matrix=matrix,
            labels=labels,
            motif_ids=motif_ids,
            label_order=label_order,
            c_value=c_value,
            random_state=random_state,
            panel_name=f"lasso_C_{c_value:g}",
        )
        rows.append(
            {
                "C": c_value,
                "roc_auc": evaluation.metrics["roc_auc"],
                "accuracy": evaluation.metrics["accuracy"],
                "balanced_accuracy": evaluation.metrics["balanced_accuracy"],
                "mean_nonzero_motifs": float(evaluation.results["nonzero_feature_count"].mean()),
                "median_nonzero_motifs": float(evaluation.results["nonzero_feature_count"].median()),
            }
        )
    tuning = pd.DataFrame(rows)
    return tuning.sort_values(
        ["roc_auc", "mean_nonzero_motifs", "C"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def evaluate_panel(
    matrix: pd.DataFrame,
    labels: pd.Series,
    motif_ids: list[Any],
    label_order: list[str],
    c_value: float,
    random_state: int,
    panel_name: str,
) -> PanelEvaluation:
    if not motif_ids:
        raise ValueError(f"Panel '{panel_name}' has no motif IDs.")

    positive_label = label_order[0]
    negative_label = label_order[1]
    x = matrix.reindex(columns=motif_ids, fill_value=0.0)
    y = labels.reindex(x.index)
    rows = []
    coefficient_rows = []

    for patient_id in x.index:
        train_index = x.index.difference([patient_id], sort=False)
        x_train = x.loc[train_index]
        x_test = x.loc[[patient_id]]
        y_train = y.loc[train_index]
        y_test = y.loc[patient_id]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )
        clf.set_params(C=float(c_value))
        clf.fit(x_train_scaled, y_train)

        pred = clf.predict(x_test_scaled)[0]
        prob = clf.predict_proba(x_test_scaled)[0]
        positive_index = int(np.where(clf.classes_ == positive_label)[0][0])
        positive_probability = float(prob[positive_index])
        coefficients = clf.coef_[0].astype(float)
        nonzero_count = int(np.count_nonzero(np.abs(coefficients) > 0.0))

        for motif_id, coef in zip(motif_ids, coefficients):
            coefficient_rows.append(
                {
                    "held_out_patient": patient_id,
                    "ID": motif_id,
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "nonzero": bool(abs(coef) > 0.0),
                }
            )

        rows.append(
            {
                "patient_id": patient_id,
                "class": y_test,
                "pred_class": pred,
                "positive_probability": positive_probability,
                "TP": int(pred == y_test and y_test == positive_label),
                "TN": int(pred == y_test and y_test == negative_label),
                "FN": int(pred == negative_label and y_test == positive_label),
                "FP": int(pred == positive_label and y_test == negative_label),
                "cFeatures": len(motif_ids),
                "nonzero_feature_count": nonzero_count,
            }
        )

    results = pd.DataFrame(rows).set_index("patient_id")
    coefficient_df = pd.DataFrame(coefficient_rows)
    coefficient_summary = summarize_coefficients(coefficient_df, n_folds=len(x.index))
    metrics = compute_metrics(results, label_order)
    return PanelEvaluation(
        panel_name=panel_name,
        motif_ids=list(motif_ids),
        results=results,
        metrics=metrics,
        coefficient_summary=coefficient_summary,
    )


def summarize_coefficients(coefficient_df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    if coefficient_df.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                "selected_fold_count",
                "coefficient_frequency",
                "mean_coefficient",
                "mean_abs_coefficient",
                "median_abs_coefficient",
                "max_abs_coefficient",
            ]
        )
    summary = coefficient_df.groupby("ID", observed=True).agg(
        selected_fold_count=("nonzero", "sum"),
        mean_coefficient=("coefficient", "mean"),
        mean_abs_coefficient=("abs_coefficient", "mean"),
        median_abs_coefficient=("abs_coefficient", "median"),
        max_abs_coefficient=("abs_coefficient", "max"),
    )
    summary["coefficient_frequency"] = summary["selected_fold_count"] / max(1, n_folds)
    summary = summary.sort_values(
        ["coefficient_frequency", "mean_abs_coefficient"],
        ascending=False,
    ).reset_index()
    summary["lasso_importance_rank"] = np.arange(1, summary.shape[0] + 1)
    return summary


def compute_metrics(results: pd.DataFrame, label_order: list[str]) -> dict[str, float]:
    positive_label = label_order[0]
    y_true = results["class"].tolist()
    y_pred = results["pred_class"].tolist()
    y_binary = [1 if label == positive_label else 0 for label in y_true]
    pred_binary = [1 if label == positive_label else 0 for label in y_pred]
    probabilities = results["positive_probability"].tolist()

    roc_auc = np.nan
    if len(set(y_binary)) == 2:
        roc_auc = float(roc_auc_score(y_binary, probabilities))

    return {
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_binary, pred_binary, zero_division=0)),
        "recall": float(recall_score(y_binary, pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_binary, pred_binary, zero_division=0)),
        "n_patients": float(len(results)),
        "mean_nonzero_motifs": float(results["nonzero_feature_count"].mean()),
        "median_nonzero_motifs": float(results["nonzero_feature_count"].median()),
    }


def compute_class_prevalence(matrix: pd.DataFrame, labels: pd.Series, label_order: list[str]) -> pd.DataFrame:
    rows = []
    for motif_id in matrix.columns:
        row = {"ID": motif_id}
        prevalences = {}
        means = {}
        for label in label_order:
            values = matrix.loc[labels == label, motif_id]
            prevalence = float((values > 0).mean()) if len(values) else 0.0
            mean_value = float(values.mean()) if len(values) else 0.0
            row[f"{label}_prevalence"] = prevalence
            row[f"{label}_mean_frequency"] = mean_value
            prevalences[label] = prevalence
            means[label] = mean_value

        dominant_label = max(label_order, key=lambda label: (prevalences[label], means[label]))
        minority_label = label_order[1] if dominant_label == label_order[0] else label_order[0]
        row["dominant_group"] = dominant_label
        row["minority_group"] = minority_label
        row["dominant_prevalence"] = prevalences[dominant_label]
        row["minority_prevalence"] = prevalences[minority_label]
        row["prevalence_gap"] = row["dominant_prevalence"] - row["minority_prevalence"]
        row["dominant_mean_frequency"] = means[dominant_label]
        row["minority_mean_frequency"] = means[minority_label]
        row["frequency_gap"] = row["dominant_mean_frequency"] - row["minority_mean_frequency"]
        rows.append(row)
    return pd.DataFrame(rows)


def select_asymmetric_motifs(
    matrix: pd.DataFrame,
    labels: pd.Series,
    label_order: list[str],
    dominant_min: float,
    minority_max: float,
    top_k: int,
) -> tuple[list[Any], pd.DataFrame]:
    prevalence = compute_class_prevalence(matrix, labels, label_order)
    selected = prevalence[
        (prevalence["dominant_prevalence"] >= dominant_min)
        & (prevalence["minority_prevalence"] <= minority_max)
        & (prevalence["prevalence_gap"] > 0)
    ].copy()
    selected = selected.sort_values(
        ["prevalence_gap", "dominant_prevalence", "frequency_gap"],
        ascending=False,
    ).head(top_k)
    return selected["ID"].tolist(), selected


def evaluate_asymmetric_stringency(
    matrix: pd.DataFrame,
    labels: pd.Series,
    label_order: list[str],
    dominant_min: float,
    minority_max: float,
    top_k: int,
    random_state: int,
    panel_name: str,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, list[Any]]]:
    positive_label = label_order[0]
    negative_label = label_order[1]
    rows = []
    selected_by_patient: dict[str, list[Any]] = {}

    for patient_id in matrix.index:
        train_index = matrix.index.difference([patient_id], sort=False)
        x_train_all = matrix.loc[train_index]
        y_train = labels.loc[train_index]
        x_test_all = matrix.loc[[patient_id]]
        y_test = labels.loc[patient_id]

        selected, _ = select_asymmetric_motifs(
            matrix=x_train_all,
            labels=y_train,
            label_order=label_order,
            dominant_min=dominant_min,
            minority_max=minority_max,
            top_k=top_k,
        )
        selected_by_patient[str(patient_id)] = selected

        if not selected:
            rows.append(
                {
                    "patient_id": patient_id,
                    "class": y_test,
                    "pred_class": None,
                    "positive_probability": np.nan,
                    "TP": 0,
                    "TN": 0,
                    "FN": 0,
                    "FP": 0,
                    "cFeatures": 0,
                    "valid_fold": False,
                }
            )
            continue

        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        clf.fit(x_train_all.reindex(columns=selected, fill_value=0.0), y_train)
        x_test = x_test_all.reindex(columns=selected, fill_value=0.0)
        pred = clf.predict(x_test)[0]
        prob = clf.predict_proba(x_test)[0]
        positive_index = int(np.where(clf.classes_ == positive_label)[0][0])
        rows.append(
            {
                "patient_id": patient_id,
                "class": y_test,
                "pred_class": pred,
                "positive_probability": float(prob[positive_index]),
                "TP": int(pred == y_test and y_test == positive_label),
                "TN": int(pred == y_test and y_test == negative_label),
                "FN": int(pred == negative_label and y_test == positive_label),
                "FP": int(pred == positive_label and y_test == negative_label),
                "cFeatures": len(selected),
                "valid_fold": True,
            }
        )

    results = pd.DataFrame(rows).set_index("patient_id")
    metrics = compute_asymmetric_metrics(results, label_order)
    metrics["panel"] = panel_name
    metrics["dominant_min"] = float(dominant_min)
    metrics["minority_max"] = float(minority_max)
    return results, metrics, selected_by_patient


def compute_asymmetric_metrics(results: pd.DataFrame, label_order: list[str]) -> dict[str, float]:
    valid = results[results["valid_fold"]].copy()
    if valid.empty:
        return {
            "roc_auc": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "valid_folds": 0.0,
            "empty_folds": float((~results["valid_fold"]).sum()),
            "mean_selected_motifs": 0.0,
            "median_selected_motifs": 0.0,
        }

    positive_label = label_order[0]
    y_true = valid["class"].tolist()
    y_pred = valid["pred_class"].tolist()
    y_binary = [1 if label == positive_label else 0 for label in y_true]
    pred_binary = [1 if label == positive_label else 0 for label in y_pred]
    probabilities = valid["positive_probability"].tolist()
    roc_auc = np.nan
    if len(set(y_binary)) == 2:
        roc_auc = float(roc_auc_score(y_binary, probabilities))

    return {
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) == 2 else np.nan,
        "precision": float(precision_score(y_binary, pred_binary, zero_division=0)),
        "recall": float(recall_score(y_binary, pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_binary, pred_binary, zero_division=0)),
        "valid_folds": float(valid.shape[0]),
        "empty_folds": float((~results["valid_fold"]).sum()),
        "mean_selected_motifs": float(results["cFeatures"].mean()),
        "median_selected_motifs": float(results["cFeatures"].median()),
    }


def run_asymmetric_stringency_experiment(
    matrix: pd.DataFrame,
    labels: pd.Series,
    label_order: list[str],
    top_k: int,
    scores: pd.DataFrame,
    output_dir: Path,
    image_dir: Path,
    dominant_min_grid: list[float],
    minority_max_grid: list[float],
    random_state: int,
    show_progress: bool,
) -> None:
    threshold_pairs = [
        (float(dominant_min), float(minority_max))
        for dominant_min in dominant_min_grid
        for minority_max in minority_max_grid
        if 0.0 <= minority_max <= dominant_min <= 1.0
    ]
    if not threshold_pairs:
        raise ValueError("Asymmetric stringency grids produced no valid threshold pairs.")

    tuning_rows = []
    progress = tqdm(
        threshold_pairs,
        desc="Tuning asymmetric stringency",
        leave=False,
        disable=not show_progress,
    )
    for dominant_min, minority_max in progress:
        _, metrics, _ = evaluate_asymmetric_stringency(
            matrix=matrix,
            labels=labels,
            label_order=label_order,
            dominant_min=dominant_min,
            minority_max=minority_max,
            top_k=top_k,
            random_state=random_state,
            panel_name="asymmetric_stringency_trial",
        )
        tuning_rows.append(metrics)

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df = tuning_df.sort_values(
        ["roc_auc", "valid_folds", "mean_selected_motifs", "dominant_min", "minority_max"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    tuning_df.to_csv(output_dir / "asymmetric_stringency_tuning.csv", index=False)
    best = tuning_df.iloc[0]
    best_dominant_min = float(best["dominant_min"])
    best_minority_max = float(best["minority_max"])

    final_results, final_metrics, selected_by_patient = evaluate_asymmetric_stringency(
        matrix=matrix,
        labels=labels,
        label_order=label_order,
        dominant_min=best_dominant_min,
        minority_max=best_minority_max,
        top_k=top_k,
        random_state=random_state,
        panel_name="asymmetric_stringency_best",
    )
    final_results.to_csv(output_dir / "asymmetric_stringency_loocv_results.csv")
    pd.Series(final_metrics, name="value").to_csv(output_dir / "asymmetric_stringency_metrics.csv")

    final_ids, selected_table = select_asymmetric_motifs(
        matrix=matrix,
        labels=labels,
        label_order=label_order,
        dominant_min=best_dominant_min,
        minority_max=best_minority_max,
        top_k=top_k,
    )
    selected_table = attach_soft_scores(selected_table, scores)
    selected_table.insert(1, "asymmetric_rank", np.arange(1, selected_table.shape[0] + 1))
    selected_table.to_csv(output_dir / "asymmetric_stringency_selected_motifs.csv", index=False)

    frequency_counts: dict[Any, int] = {}
    for motif_ids in selected_by_patient.values():
        for motif_id in motif_ids:
            frequency_counts[motif_id] = frequency_counts.get(motif_id, 0) + 1
    frequency_df = (
        pd.Series(frequency_counts, name="selected_fold_count")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "ID"})
    )
    frequency_df["selection_frequency"] = frequency_df["selected_fold_count"] / max(1, len(selected_by_patient))
    frequency_df = attach_soft_scores(frequency_df, scores)
    frequency_df.to_csv(output_dir / "asymmetric_stringency_loocv_selection_frequency.csv", index=False)

    save_json(
        {
            "best_dominant_min": best_dominant_min,
            "best_minority_max": best_minority_max,
            "top_k": top_k,
            "dominant_min_grid": dominant_min_grid,
            "minority_max_grid": minority_max_grid,
            "valid_threshold_pairs": len(threshold_pairs),
            "selected_motif_count": len(final_ids),
        },
        output_dir / "asymmetric_stringency_config.json",
    )

    save_table_image(
        tuning_df.head(20),
        image_dir / "asymmetric_stringency_tuning_top20.png",
        title="Asymmetric stringency tuning",
        max_cols=12,
    )
    save_table_image(
        selected_table,
        image_dir / "asymmetric_stringency_selected_motifs.png",
        title="Asymmetric stringency selected motifs",
        max_rows=40,
        max_cols=12,
    )
    plot_asymmetric_tuning(tuning_df, image_dir / "asymmetric_stringency_tuning.png")
    plot_asymmetric_selected(selected_table, image_dir / "asymmetric_stringency_selected_motifs_bar.png")
    plot_asymmetric_roc(final_results, label_order, image_dir / "asymmetric_stringency_roc_curve.png")


def plot_asymmetric_tuning(tuning_df: pd.DataFrame, path: Path) -> None:
    pivot = tuning_df.pivot_table(
        index="dominant_min",
        columns="minority_max",
        values="roc_auc",
        aggfunc="max",
    ).sort_index(ascending=False)
    plt.figure(figsize=(9, 6))
    if sns is not None:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    else:
        image = plt.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xlabel("Minority-group maximum prevalence")
    plt.ylabel("Dominant-group minimum prevalence")
    plt.title("Asymmetric stringency LOOCV ROC AUC")
    save_current_figure(path)


def plot_asymmetric_selected(selected_table: pd.DataFrame, path: Path) -> None:
    plot_df = selected_table.copy()
    if plot_df.empty:
        plot_df = pd.DataFrame({"ID": ["none"], "prevalence_gap": [0.0]})
    plot_df["ID"] = plot_df["ID"].astype(str)
    plt.figure(figsize=(10, max(4, 0.35 * len(plot_df) + 2)))
    if sns is not None:
        sns.barplot(data=plot_df, y="ID", x="prevalence_gap", hue="dominant_group", dodge=False)
    else:
        plt.barh(np.arange(len(plot_df)), plot_df["prevalence_gap"], color="#4C78A8")
        plt.yticks(np.arange(len(plot_df)), plot_df["ID"])
        plt.gca().invert_yaxis()
    plt.xlabel("Dominant minus minority prevalence")
    plt.ylabel("Motif ID")
    plt.title("Asymmetric stringency selected motifs")
    save_current_figure(path)


def plot_asymmetric_roc(results: pd.DataFrame, label_order: list[str], path: Path) -> None:
    valid = results[results["valid_fold"]].copy()
    if valid.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No valid folds", ha="center", va="center")
        plt.axis("off")
        save_current_figure(path)
        return
    positive_label = label_order[0]
    y_true = [1 if label == positive_label else 0 for label in valid["class"]]
    probabilities = valid["positive_probability"].to_numpy(dtype=float)
    auc_value = roc_auc_score(y_true, probabilities) if len(set(y_true)) == 2 else np.nan
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4C78A8", label=f"ROC AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="#999999", linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Asymmetric stringency LOOCV ROC curve")
    plt.legend(loc="lower right")
    save_current_figure(path)


def attach_soft_scores(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    available = ["ID", "soft_rank"] + [column for column in SCORE_COLUMNS if column in scores.columns]
    return df.merge(scores[available], on="ID", how="left")


def prepare_lower_candidates(
    scores: pd.DataFrame,
    original_panel: list[Any],
    max_candidates: int,
) -> pd.DataFrame:
    original_set = set(original_panel)
    candidates = scores[~scores["ID"].isin(original_set)].head(max_candidates).copy()
    candidates["candidate_order"] = np.arange(1, candidates.shape[0] + 1)
    return candidates


def refine_with_replacements(
    matrix: pd.DataFrame,
    labels: pd.Series,
    label_order: list[str],
    scores: pd.DataFrame,
    kept_panel: list[Any],
    original_panel: list[Any],
    baseline_roc_auc: float,
    c_value: float,
    random_state: int,
    max_candidates: int,
    acceptance_tolerance: float,
    show_progress: bool,
) -> tuple[list[Any], pd.DataFrame, pd.DataFrame]:
    target_size = len(original_panel)
    current_panel = list(kept_panel)
    lower_candidates = prepare_lower_candidates(scores, original_panel, max_candidates)
    trial_rows = []
    progress = tqdm(
        lower_candidates.iterrows(),
        total=lower_candidates.shape[0],
        desc="Trying lower-ranked motifs",
        leave=False,
        disable=not show_progress,
    )
    for _, candidate in progress:
        if len(current_panel) >= target_size:
            break
        candidate_id = candidate["ID"]
        trial_panel = current_panel + [candidate_id]
        evaluation = evaluate_panel(
            matrix=matrix,
            labels=labels,
            motif_ids=trial_panel,
            label_order=label_order,
            c_value=c_value,
            random_state=random_state,
            panel_name=f"replacement_{candidate_id}",
        )
        trial_roc_auc = evaluation.metrics["roc_auc"]
        accepted = bool(trial_roc_auc + acceptance_tolerance >= baseline_roc_auc)
        if accepted:
            current_panel = trial_panel
        trial_rows.append(
            {
                "ID": candidate_id,
                "candidate_order": int(candidate["candidate_order"]),
                "soft_rank": int(candidate["soft_rank"]),
                "accepted": accepted,
                "panel_size_before": len(trial_panel) - 1,
                "panel_size_after": len(current_panel),
                "trial_roc_auc": trial_roc_auc,
                "baseline_roc_auc": baseline_roc_auc,
                "accuracy": evaluation.metrics["accuracy"],
                "balanced_accuracy": evaluation.metrics["balanced_accuracy"],
                "mean_nonzero_motifs": evaluation.metrics["mean_nonzero_motifs"],
            }
        )
    return current_panel, lower_candidates, pd.DataFrame(trial_rows)


def metrics_table(evaluations: list[PanelEvaluation]) -> pd.DataFrame:
    rows = []
    for evaluation in evaluations:
        row = {"panel": evaluation.panel_name, "motif_count": len(evaluation.motif_ids)}
        row.update(evaluation.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def save_panel_outputs(
    output_dir: Path,
    image_dir: Path,
    scores: pd.DataFrame,
    original_panel: list[Any],
    lasso_importance: pd.DataFrame,
    selected_top: pd.DataFrame,
    removed_top: pd.DataFrame,
    lower_candidates: pd.DataFrame,
    replacement_trials: pd.DataFrame,
    final_panel: list[Any],
    final_evaluation: PanelEvaluation,
    comparison_metrics: pd.DataFrame,
    label_order: list[str],
) -> None:
    lasso_importance.to_csv(output_dir / "lasso_feature_importance.csv", index=False)
    selected_top.to_csv(output_dir / "lasso_selected_top_motifs.csv", index=False)
    removed_top.to_csv(output_dir / "lasso_removed_top_motifs.csv", index=False)
    lower_candidates.to_csv(output_dir / "candidate_lower_ranked_motifs.csv", index=False)
    replacement_trials.to_csv(output_dir / "replacement_trials.csv", index=False)

    final_panel_df = pd.DataFrame(
        {
            "ID": final_panel,
            "final_panel_rank": np.arange(1, len(final_panel) + 1),
            "source": ["original_top" if motif_id in set(original_panel) else "lower_ranked_replacement" for motif_id in final_panel],
        }
    )
    final_panel_df = attach_soft_scores(final_panel_df, scores)
    final_panel_df.to_csv(output_dir / "final_refined_motif_panel.csv", index=False)

    final_evaluation.results.to_csv(output_dir / "refined_panel_loocv_results.csv")
    pd.Series(final_evaluation.metrics, name="value").to_csv(output_dir / "refined_panel_metrics.csv")
    comparison_metrics.to_csv(output_dir / "panel_comparison_metrics.csv", index=False)

    save_table_image(
        lasso_importance.head(20),
        image_dir / "lasso_feature_importance_top20.png",
        title="LASSO motif feature importance",
        max_cols=12,
    )
    save_table_image(
        removed_top,
        image_dir / "lasso_removed_top_motifs.png",
        title="Top motifs removed by LASSO",
        max_cols=12,
    )
    save_table_image(
        replacement_trials,
        image_dir / "replacement_trials.png",
        title="Lower-ranked motif replacement trials",
        max_rows=30,
        max_cols=12,
    )
    save_table_image(
        final_panel_df,
        image_dir / "final_refined_motif_panel.png",
        title="Final refined motif panel",
        max_rows=40,
        max_cols=12,
    )
    save_table_image(
        comparison_metrics,
        image_dir / "panel_comparison_metrics.png",
        title="Original vs pruned vs refined panels",
        max_rows=10,
        max_cols=12,
    )

    plot_importance(lasso_importance.head(25), image_dir / "lasso_feature_importance.png")
    plot_removed_motifs(removed_top, image_dir / "lasso_removed_top_motifs_bar.png")
    plot_replacement_trials(replacement_trials, image_dir / "replacement_decisions.png")
    plot_final_panel(final_panel_df, image_dir / "final_refined_motif_panel_bar.png")
    plot_confusion(final_evaluation.results, label_order, image_dir / "refined_panel_confusion_matrix.png")
    plot_roc(final_evaluation.results, label_order, image_dir / "refined_panel_roc_curve.png")


def plot_importance(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    plot_df["ID"] = plot_df["ID"].astype(str)
    plt.figure(figsize=(9, 7))
    if sns is not None:
        sns.barplot(data=plot_df, y="ID", x="mean_abs_coefficient", color="#4C78A8")
    else:
        plt.barh(np.arange(len(plot_df)), plot_df["mean_abs_coefficient"], color="#4C78A8")
        plt.yticks(np.arange(len(plot_df)), plot_df["ID"])
        plt.gca().invert_yaxis()
    plt.xlabel("Mean absolute LASSO coefficient")
    plt.ylabel("Motif ID")
    plt.title("LASSO motif feature importance")
    save_current_figure(path)


def plot_removed_motifs(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    if plot_df.empty:
        plot_df = pd.DataFrame({"ID": ["none"], "coefficient_frequency": [0.0]})
    plot_df["ID"] = plot_df["ID"].astype(str)
    plt.figure(figsize=(9, max(3, 0.35 * len(plot_df) + 2)))
    if sns is not None:
        sns.barplot(data=plot_df, y="ID", x="coefficient_frequency", color="#E45756")
    else:
        plt.barh(np.arange(len(plot_df)), plot_df["coefficient_frequency"], color="#E45756")
        plt.yticks(np.arange(len(plot_df)), plot_df["ID"])
        plt.gca().invert_yaxis()
    plt.xlabel("Non-zero coefficient frequency")
    plt.ylabel("Motif ID")
    plt.title("Top motifs removed by LASSO")
    save_current_figure(path)


def plot_replacement_trials(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No replacement trials were needed", ha="center", va="center")
        plt.axis("off")
        save_current_figure(path)
        return
    plot_df = df.copy()
    plot_df["ID"] = plot_df["ID"].astype(str)
    colors = np.where(plot_df["accepted"], "#59A14F", "#E45756")
    plt.figure(figsize=(11, 5))
    plt.bar(np.arange(len(plot_df)), plot_df["trial_roc_auc"], color=colors)
    plt.axhline(plot_df["baseline_roc_auc"].iloc[0], color="#333333", linestyle="--", label="Original panel ROC AUC")
    plt.xticks(np.arange(len(plot_df)), plot_df["ID"], rotation=45, ha="right")
    plt.ylabel("LOOCV ROC AUC")
    plt.xlabel("Candidate motif ID")
    plt.title("Lower-ranked motif replacement decisions")
    plt.legend(loc="best")
    save_current_figure(path)


def plot_final_panel(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    plot_df["ID"] = plot_df["ID"].astype(str)
    colors = plot_df["source"].map({"original_top": "#4C78A8", "lower_ranked_replacement": "#59A14F"}).fillna("#4C78A8")
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(plot_df)), plot_df["final_score"].fillna(0.0), color=colors)
    plt.xticks(np.arange(len(plot_df)), plot_df["ID"], rotation=45, ha="right")
    plt.ylabel("03b final soft motif score")
    plt.xlabel("Motif ID")
    plt.title("Final refined motif panel")
    save_current_figure(path)


def plot_confusion(results: pd.DataFrame, label_order: list[str], path: Path) -> None:
    matrix = confusion_matrix(results["class"], results["pred_class"], labels=label_order)
    confusion = pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in label_order],
        columns=[f"Predicted {label}" for label in label_order],
    )
    confusion.to_csv(path.parent.parent / "refined_panel_confusion_matrix.csv")
    plt.figure(figsize=(5, 4))
    if sns is not None:
        sns.heatmap(confusion, annot=True, fmt=".0f", cmap="Blues", cbar=False)
    else:
        image = plt.imshow(confusion.to_numpy(dtype=float), cmap="Blues")
        for row in range(confusion.shape[0]):
            for col in range(confusion.shape[1]):
                plt.text(col, row, str(confusion.iloc[row, col]), ha="center", va="center")
        plt.xticks(np.arange(len(confusion.columns)), confusion.columns, rotation=30, ha="right")
        plt.yticks(np.arange(len(confusion.index)), confusion.index)
        plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.title("Refined panel LOOCV confusion matrix")
    save_current_figure(path)


def plot_roc(results: pd.DataFrame, label_order: list[str], path: Path) -> None:
    positive_label = label_order[0]
    y_true = [1 if label == positive_label else 0 for label in results["class"]]
    probabilities = results["positive_probability"].to_numpy(dtype=float)
    auc_value = roc_auc_score(y_true, probabilities) if len(set(y_true)) == 2 else np.nan
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4C78A8", label=f"ROC AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="#999999", linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Refined panel LOOCV ROC curve")
    plt.legend(loc="lower right")
    save_current_figure(path)


def main() -> None:
    args = parse_args()
    if args.max_candidates < 0:
        raise ValueError("--max-candidates must be non-negative.")
    if args.coefficient_frequency_threshold < 0 or args.coefficient_frequency_threshold > 1:
        raise ValueError("--coefficient-frequency-threshold must be between 0 and 1.")
    if any(c <= 0 for c in args.c_grid):
        raise ValueError("--c-grid values must be positive.")
    if any(value < 0 or value > 1 for value in args.dominant_min_grid + args.minority_max_grid):
        raise ValueError("--dominant-min-grid and --minority-max-grid values must be between 0 and 1.")

    soft_motif_dir = args.soft_motif_dir.resolve()
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    config = load_soft_config(soft_motif_dir / "soft_motif_selection_config.json")
    labels = list(config["labels"])
    top_k = int(config["top_k"])
    scores, original_panel = read_ranked_motifs(soft_motif_dir)
    original_panel = original_panel[:top_k]
    candidate_ids = scores[~scores["ID"].isin(set(original_panel))].head(args.max_candidates)["ID"].tolist()
    asymmetric_ids = [] if args.skip_asymmetric_stringency else motif_ids_from_scores(
        scores,
        args.asymmetric_max_ranked_motifs,
    )
    matrix_ids = list(dict.fromkeys(original_panel + candidate_ids + asymmetric_ids))

    print(f"Serialized artifact: {args.artifact_path.resolve()}")
    print(f"03b output directory: {soft_motif_dir}")
    print(f"04b output directory: {output_dir}")
    print(f"Labels: {labels}")
    print(f"Original panel size: {len(original_panel)}")
    print(f"Lower-ranked candidates to inspect: {len(candidate_ids)}")
    if not args.skip_asymmetric_stringency:
        print(f"Asymmetric stringency ranked motif pool: {len(asymmetric_ids)}")

    matrix, patient_labels = build_restricted_matrix(args.artifact_path.resolve(), labels, matrix_ids)
    print(f"Patient-by-motif matrix: {matrix.shape[0]} patients x {matrix.shape[1]} motifs")

    tuning = tune_lasso_c(
        matrix=matrix,
        labels=patient_labels,
        motif_ids=original_panel,
        label_order=labels,
        c_grid=list(args.c_grid),
        random_state=args.random_state,
    )
    tuning.to_csv(output_dir / "lasso_c_tuning.csv", index=False)
    best_c = float(tuning.iloc[0]["C"])
    print(f"Selected LASSO C: {best_c:g}")

    original_evaluation = evaluate_panel(
        matrix=matrix,
        labels=patient_labels,
        motif_ids=original_panel,
        label_order=labels,
        c_value=best_c,
        random_state=args.random_state,
        panel_name="original_03b_panel",
    )
    baseline_roc_auc = original_evaluation.metrics["roc_auc"]
    lasso_importance = attach_soft_scores(original_evaluation.coefficient_summary, scores)

    selected_top = lasso_importance[
        lasso_importance["coefficient_frequency"] >= args.coefficient_frequency_threshold
    ].copy()
    if selected_top.empty:
        selected_top = lasso_importance.head(1).copy()
        selected_top["retention_note"] = "kept_highest_lasso_importance_because_threshold_removed_all_motifs"
    else:
        selected_top["retention_note"] = "met_coefficient_frequency_threshold"

    kept_panel = [motif_id for motif_id in original_panel if motif_id in set(selected_top["ID"])]
    removed_top = lasso_importance[~lasso_importance["ID"].isin(set(kept_panel))].copy()
    removed_top["removal_reason"] = "below_coefficient_frequency_threshold"

    pruned_evaluation = evaluate_panel(
        matrix=matrix,
        labels=patient_labels,
        motif_ids=kept_panel,
        label_order=labels,
        c_value=best_c,
        random_state=args.random_state,
        panel_name="lasso_pruned_panel",
    )

    final_panel, lower_candidates, replacement_trials = refine_with_replacements(
        matrix=matrix,
        labels=patient_labels,
        label_order=labels,
        scores=scores,
        kept_panel=kept_panel,
        original_panel=original_panel,
        baseline_roc_auc=baseline_roc_auc,
        c_value=best_c,
        random_state=args.random_state,
        max_candidates=args.max_candidates,
        acceptance_tolerance=args.acceptance_tolerance,
        show_progress=not args.no_progress,
    )

    final_evaluation = evaluate_panel(
        matrix=matrix,
        labels=patient_labels,
        motif_ids=final_panel,
        label_order=labels,
        c_value=best_c,
        random_state=args.random_state,
        panel_name="final_refined_panel",
    )
    comparison_metrics = metrics_table([original_evaluation, pruned_evaluation, final_evaluation])

    save_json(
        {
            "labels": labels,
            "top_k": top_k,
            "best_c": best_c,
            "c_grid": list(args.c_grid),
            "coefficient_frequency_threshold": args.coefficient_frequency_threshold,
            "max_candidates": args.max_candidates,
            "acceptance_tolerance": args.acceptance_tolerance,
            "random_state": args.random_state,
            "baseline_original_panel_roc_auc": baseline_roc_auc,
            "final_panel_size": len(final_panel),
        },
        output_dir / "lasso_motif_refinement_config.json",
    )

    save_panel_outputs(
        output_dir=output_dir,
        image_dir=image_dir,
        scores=scores,
        original_panel=original_panel,
        lasso_importance=lasso_importance,
        selected_top=selected_top,
        removed_top=removed_top,
        lower_candidates=lower_candidates,
        replacement_trials=replacement_trials,
        final_panel=final_panel,
        final_evaluation=final_evaluation,
        comparison_metrics=comparison_metrics,
        label_order=labels,
    )

    if not args.skip_asymmetric_stringency:
        run_asymmetric_stringency_experiment(
            matrix=matrix.reindex(columns=asymmetric_ids, fill_value=0.0),
            labels=patient_labels,
            label_order=labels,
            top_k=top_k,
            scores=scores,
            output_dir=output_dir,
            image_dir=image_dir,
            dominant_min_grid=list(args.dominant_min_grid),
            minority_max_grid=list(args.minority_max_grid),
            random_state=args.random_state,
            show_progress=not args.no_progress,
        )

    print("Panel comparison:")
    print(comparison_metrics.to_string(index=False))
    print(f"Final panel motif count: {len(final_panel)}")


if __name__ == "__main__":
    main()
