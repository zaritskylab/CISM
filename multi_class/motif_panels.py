from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from cism import build_patient_motif_matrix
from cism.cism import DiscriminativeMotifs

from .config import MAX_CLASS_FEATURES, OTHER_LABEL, P_VALUE_THRESHOLD, RANDOM_STATE, STRINGENCY_GRID


@dataclass(frozen=True)
class Comparison:
    name: str
    labels: tuple[str, str]
    source_labels: tuple[str, ...]
    positive_label: str
    kind: str


def make_comparisons(labels: tuple[str, ...] | list[str], method: str) -> list[Comparison]:
    labels = tuple(labels)
    if method in {"one_vs_all_union", "one_vs_all_positive_union"}:
        return [
            Comparison(
                name=f"{label}_vs_all",
                labels=(label, OTHER_LABEL),
                source_labels=labels,
                positive_label=label,
                kind=method,
            )
            for label in labels
        ]
    if method == "one_vs_one_union":
        return [
            Comparison(
                name=f"{left}_vs_{right}",
                labels=(left, right),
                source_labels=(left, right),
                positive_label=left,
                kind=method,
            )
            for left, right in itertools.combinations(labels, 2)
        ]
    raise ValueError(f"Unsupported motif panel method: {method}")


def tune_and_select_panel(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    labels: tuple[str, ...] | list[str],
    method: str,
    output_dir: Path,
    *,
    force_tuning: bool = False,
    max_class_features: int = MAX_CLASS_FEATURES,
    stringency_grid: tuple[float, ...] = STRINGENCY_GRID,
    p_value_threshold: float = P_VALUE_THRESHOLD,
    random_state: int = RANDOM_STATE,
    max_patients_per_class: int | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_motifs.csv"
    tuning_path = output_dir / "comparison_tuning.csv"
    if selected_path.exists() and tuning_path.exists() and not force_tuning:
        return pd.read_csv(selected_path)

    local_classes = patient_classes[patient_classes[DiscriminativeMotifs.PATIENT_CLASS].isin(labels)].copy()
    if max_patients_per_class is not None:
        local_classes = (
            local_classes.groupby(DiscriminativeMotifs.PATIENT_CLASS, group_keys=False, observed=True)
            .head(max_patients_per_class)
            .copy()
        )

    if method == "one_vs_all_positive_union":
        return select_positive_subset_from_one_vs_all(
            motifs_df,
            local_classes,
            tuple(labels),
            output_dir,
            force_tuning=force_tuning,
            max_class_features=max_class_features,
            stringency_grid=stringency_grid,
            p_value_threshold=p_value_threshold,
            random_state=random_state,
            max_patients_per_class=max_patients_per_class,
        )

    comparisons = make_comparisons(tuple(labels), method)
    all_selected: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for comparison in tqdm(
        comparisons,
        desc=f"{method}: binary comparisons",
        unit="comparison",
        leave=False,
    ):
        comparison_classes = _classes_for_comparison(local_classes, comparison)
        comparison_motifs = motifs_df[motifs_df["Patient_uId"].isin(comparison_classes.index)].copy()
        best, trials = tune_comparison_stringency(
            comparison_motifs,
            comparison_classes,
            comparison=comparison,
            stringency_grid=stringency_grid,
            max_class_features=max_class_features,
            p_value_threshold=p_value_threshold,
            random_state=random_state,
        )
        tuning_rows.append(best)
        trial_rows.extend(trials)
        selected = select_full_comparison_motifs(
            comparison_motifs,
            comparison_classes,
            comparison=comparison,
            stringency=float(best["stringency"]),
            max_class_features=max_class_features,
            p_value_threshold=p_value_threshold,
        )
        all_selected.append(selected)

    tuning_df = pd.DataFrame(tuning_rows)
    trials_df = pd.DataFrame(trial_rows)
    selected_df = merge_selected_motifs(all_selected)
    if selected_df.empty:
        raise ValueError(f"No motifs selected for method {method}. Try lowering stringency or p-value thresholds.")

    tuning_df.to_csv(tuning_path, index=False)
    trials_df.to_csv(output_dir / "comparison_stringency_trials.csv", index=False)
    selected_df.to_csv(selected_path, index=False)
    (output_dir / "panel_summary.json").write_text(
        json.dumps(
            {
                "method": method,
                "labels": list(labels),
                "comparisons": [comparison.name for comparison in comparisons],
                "selected_motif_count": int(selected_df["ID"].nunique()),
                "max_class_features": int(max_class_features),
                "stringency_grid": list(stringency_grid),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected_df


def select_positive_subset_from_one_vs_all(
    motifs_df: pd.DataFrame,
    local_classes: pd.DataFrame,
    labels: tuple[str, ...],
    output_dir: Path,
    *,
    force_tuning: bool,
    max_class_features: int,
    stringency_grid: tuple[float, ...],
    p_value_threshold: float,
    random_state: int,
    max_patients_per_class: int | None,
) -> pd.DataFrame:
    """Reuse normal one-vs-all selections, then keep only positive-class motifs."""
    base_output_dir = output_dir.parent / "one_vs_all_union"
    base_selected = tune_and_select_panel(
        motifs_df,
        local_classes,
        labels,
        "one_vs_all_union",
        base_output_dir,
        force_tuning=False,
        max_class_features=max_class_features,
        stringency_grid=stringency_grid,
        p_value_threshold=p_value_threshold,
        random_state=random_state,
        max_patients_per_class=max_patients_per_class,
    )
    base_tuning_path = base_output_dir / "comparison_tuning.csv"
    if not base_tuning_path.exists():
        raise FileNotFoundError(f"Expected base one-vs-all tuning output missing: {base_tuning_path}")

    base_tuning = pd.read_csv(base_tuning_path)
    comparisons = {comparison.name: comparison for comparison in make_comparisons(labels, "one_vs_all_union")}
    all_selected: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []

    for _, tuning_row in tqdm(
        base_tuning.iterrows(),
        total=len(base_tuning),
        desc="one_vs_all_positive_union: filter base comparisons",
        unit="comparison",
        leave=False,
    ):
        comparison = comparisons[str(tuning_row["comparison"])]
        comparison_classes = _classes_for_comparison(local_classes, comparison)
        comparison_motifs = motifs_df[motifs_df["Patient_uId"].isin(comparison_classes.index)].copy()
        selected = select_full_comparison_motifs(
            comparison_motifs,
            comparison_classes,
            comparison=comparison,
            stringency=float(tuning_row["stringency"]),
            max_class_features=max_class_features,
            p_value_threshold=p_value_threshold,
        )
        selected = selected[selected["comparison_label"] == comparison.positive_label].copy()
        if not selected.empty:
            selected["comparison"] = comparison.name
            selected["source_method"] = "one_vs_all_union"
            all_selected.append(selected)

        row = tuning_row.to_dict()
        row["method"] = "one_vs_all_positive_union"
        row["source_method"] = "one_vs_all_union"
        row["selection_filter"] = "positive_label_after_base_selection"
        tuning_rows.append(row)

    selected_df = merge_selected_motifs(all_selected)
    if selected_df.empty:
        raise ValueError(
            "No positive-class motifs remained after filtering the base one-vs-all selections. "
            "The base selected panel contained "
            f"{base_selected['ID'].nunique() if 'ID' in base_selected else 0} motifs."
        )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(output_dir / "comparison_tuning.csv", index=False)
    base_trials_path = base_output_dir / "comparison_stringency_trials.csv"
    if base_trials_path.exists():
        trials_df = pd.read_csv(base_trials_path)
        trials_df["method"] = "one_vs_all_positive_union"
        trials_df["source_method"] = "one_vs_all_union"
        trials_df["selection_filter"] = "positive_label_after_base_selection"
    else:
        trials_df = pd.DataFrame()
    trials_df.to_csv(output_dir / "comparison_stringency_trials.csv", index=False)
    selected_df.to_csv(output_dir / "selected_motifs.csv", index=False)
    (output_dir / "panel_summary.json").write_text(
        json.dumps(
            {
                "method": "one_vs_all_positive_union",
                "source_method": "one_vs_all_union",
                "selection_filter": "positive_label_after_base_selection",
                "labels": list(labels),
                "comparisons": [comparison.name for comparison in comparisons.values()],
                "selected_motif_count": int(selected_df["ID"].nunique()),
                "base_selected_motif_count": int(base_selected["ID"].nunique()),
                "max_class_features": int(max_class_features),
                "stringency_grid": list(stringency_grid),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected_df


def tune_comparison_stringency(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    *,
    comparison: Comparison,
    stringency_grid: tuple[float, ...],
    max_class_features: int,
    p_value_threshold: float,
    random_state: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for stringency in tqdm(
        stringency_grid,
        desc=f"{comparison.name}: stringency",
        unit="threshold",
        leave=False,
    ):
        metrics = evaluate_binary_loocv(
            motifs_df,
            patient_classes,
            comparison=comparison,
            stringency=stringency,
            max_class_features=max_class_features,
            p_value_threshold=p_value_threshold,
            random_state=random_state,
        )
        rows.append(
            {
                "comparison": comparison.name,
                "method": comparison.kind,
                "positive_label": comparison.positive_label,
                "stringency": float(stringency),
                **metrics,
            }
        )
    trials = pd.DataFrame(rows)
    sortable = trials.copy()
    sortable["auc_sort"] = sortable["roc_auc"].fillna(-1.0)
    sortable["balanced_accuracy_sort"] = sortable["balanced_accuracy"].fillna(-1.0)
    best = sortable.sort_values(
        ["auc_sort", "balanced_accuracy_sort", "mean_selected_motifs"],
        ascending=[False, False, False],
    ).iloc[0]
    best_dict = trials.loc[best.name].to_dict()
    best_dict["best_metric"] = "roc_auc"
    return best_dict, rows


def evaluate_binary_loocv(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    *,
    comparison: Comparison,
    stringency: float,
    max_class_features: int,
    p_value_threshold: float,
    random_state: int,
) -> dict[str, Any]:
    patient_ids = list(patient_classes.index.astype(str))
    true_labels: list[str] = []
    pred_labels: list[str] = []
    positive_probabilities: list[float] = []
    selected_counts: list[int] = []
    skipped_folds = 0

    for held_out in tqdm(
        patient_ids,
        desc=f"{comparison.name}: binary LOOCV",
        unit="patient",
        leave=False,
    ):
        train_patients = [patient for patient in patient_ids if patient != held_out]
        y_train = patient_classes.loc[train_patients, DiscriminativeMotifs.PATIENT_CLASS]
        if y_train.nunique() < 2:
            skipped_folds += 1
            continue

        selected = select_full_comparison_motifs(
            motifs_df[motifs_df["Patient_uId"].isin(train_patients)],
            patient_classes.loc[train_patients],
            comparison=comparison,
            stringency=stringency,
            max_class_features=max_class_features,
            p_value_threshold=p_value_threshold,
        )
        selected_motifs = selected["ID"].drop_duplicates().tolist()
        selected_counts.append(len(selected_motifs))
        if not selected_motifs:
            skipped_folds += 1
            continue

        x_train = build_patient_motif_matrix(
            motifs_df[motifs_df["Patient_uId"].isin(train_patients)],
            patient_ids=train_patients,
        ).reindex(columns=selected_motifs, fill_value=0.0)
        x_test = build_patient_motif_matrix(
            motifs_df[motifs_df["Patient_uId"] == held_out],
            patient_ids=[held_out],
        ).reindex(columns=selected_motifs, fill_value=0.0)

        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)[0]
        prob = clf.predict_proba(x_test)[0]
        class_index = np.where(clf.classes_ == comparison.positive_label)[0]
        if len(class_index) == 0:
            skipped_folds += 1
            continue
        true_labels.append(patient_classes.loc[held_out, DiscriminativeMotifs.PATIENT_CLASS])
        pred_labels.append(pred)
        positive_probabilities.append(float(prob[class_index[0]]))

    metrics: dict[str, Any] = {
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
    if true_labels:
        metrics["accuracy"] = float(accuracy_score(true_labels, pred_labels))
    if len(set(true_labels)) == 2:
        y_binary = [1 if label == comparison.positive_label else 0 for label in true_labels]
        metrics["roc_auc"] = float(roc_auc_score(y_binary, positive_probabilities))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(true_labels, pred_labels))
    return metrics


def select_full_comparison_motifs(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    *,
    comparison: Comparison,
    stringency: float,
    max_class_features: int,
    p_value_threshold: float,
) -> pd.DataFrame:
    summary = discover_hard_discriminative_motifs(motifs_df, patient_classes, p_value_threshold)
    selected_parts = []
    for label in _selection_labels_for_comparison(comparison):
        label_rows = summary[
            (summary[DiscriminativeMotifs.PATIENT_CLASS] == label)
            & (summary["patient_percentage"] >= stringency)
        ].head(max_class_features)
        if not label_rows.empty:
            selected_parts.append(label_rows)
    if not selected_parts:
        return pd.DataFrame(columns=["ID", "comparison", "comparison_label", "patient_percentage", "mean_freq", "total_count"])
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["comparison"] = comparison.name
    selected["comparison_label"] = selected[DiscriminativeMotifs.PATIENT_CLASS]
    selected["source_labels"] = ",".join(comparison.source_labels)
    return selected


def discover_hard_discriminative_motifs(
    motifs_df: pd.DataFrame,
    patient_classes: pd.DataFrame,
    p_value_threshold: float,
) -> pd.DataFrame:
    if motifs_df.empty or patient_classes.empty:
        return pd.DataFrame()
    local = motifs_df[motifs_df["p_value"] < p_value_threshold].copy()
    local = local[local["Patient_uId"].isin(patient_classes.index)].copy()
    if local.empty:
        return pd.DataFrame()
    local[DiscriminativeMotifs.PATIENT_CLASS] = local["Patient_uId"].map(
        patient_classes[DiscriminativeMotifs.PATIENT_CLASS].to_dict()
    )
    label_counts = patient_classes[DiscriminativeMotifs.PATIENT_CLASS].value_counts()
    motif_label_counts = local.drop_duplicates(["ID", DiscriminativeMotifs.PATIENT_CLASS]).groupby("ID", observed=True)[
        DiscriminativeMotifs.PATIENT_CLASS
    ].nunique()
    exclusive_ids = motif_label_counts[motif_label_counts == 1].index
    local = local[local["ID"].isin(exclusive_ids)].copy()
    if local.empty:
        return pd.DataFrame()
    grouped = (
        local.groupby(["ID", DiscriminativeMotifs.PATIENT_CLASS], observed=True)
        .agg(
            patient_count=("Patient_uId", "nunique"),
            mean_freq=("Freq", "mean"),
            median_freq=("Freq", "median"),
            total_count=("Count", "sum"),
        )
        .reset_index()
    )
    grouped["patient_percentage"] = grouped.apply(
        lambda row: row["patient_count"] / label_counts.loc[row[DiscriminativeMotifs.PATIENT_CLASS]],
        axis=1,
    )
    return grouped.sort_values(
        [DiscriminativeMotifs.PATIENT_CLASS, "patient_percentage", "patient_count", "mean_freq", "total_count"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)


def merge_selected_motifs(selected_tables: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [table for table in selected_tables if table is not None and not table.empty]
    if not non_empty:
        return pd.DataFrame(columns=["ID", "comparison_provenance", "comparison_label_provenance"])
    selected = pd.concat(non_empty, ignore_index=True)
    provenance = (
        selected.groupby("ID", observed=True)
        .agg(
            comparison_provenance=("comparison", lambda values: ";".join(sorted(set(map(str, values))))),
            comparison_label_provenance=("comparison_label", lambda values: ";".join(sorted(set(map(str, values))))),
            max_patient_percentage=("patient_percentage", "max"),
            max_mean_freq=("mean_freq", "max"),
            total_count_across_selected_rows=("total_count", "sum"),
        )
        .reset_index()
    )
    return provenance.sort_values(["max_patient_percentage", "max_mean_freq", "total_count_across_selected_rows"], ascending=False)


def _classes_for_comparison(patient_classes: pd.DataFrame, comparison: Comparison) -> pd.DataFrame:
    local = patient_classes[patient_classes[DiscriminativeMotifs.PATIENT_CLASS].isin(comparison.source_labels)].copy()
    if comparison.kind in {"one_vs_all_union", "one_vs_all_positive_union"}:
        local[DiscriminativeMotifs.PATIENT_CLASS] = np.where(
            local[DiscriminativeMotifs.PATIENT_CLASS] == comparison.positive_label,
            comparison.positive_label,
            OTHER_LABEL,
        )
    return local


def _selection_labels_for_comparison(comparison: Comparison) -> tuple[str, ...]:
    return comparison.labels
