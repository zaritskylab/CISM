#!/usr/bin/env python3
"""Build clean CISM4 and SMORE stringency CSVs on a shared 0.1 grid."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
import pickle
import sys
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut

SCRIPT_DIR = ANALYSIS_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.smore_comparison.compact_smore_results import (  # noqa: E402
    DEFAULT_METADATA_DIR,
    DEFAULT_SMORE_RESULTS,
    load_labeled_smore,
    motif_stringency,
)
from analysis.cohort_imbalance.imbalanced_bootstrap_stringency import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    build_framework_random_forest,
    discover_hard_discriminative_motifs,
    load_patient_classes,
    select_motifs_by_label,
)
from cism.cism import DiscriminativeMotifs  # noqa: E402
from cism.motif_selection import build_patient_motif_matrix  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "outputs" / "stringency_clean"
GRID = [round(x / 10, 1) for x in range(0, 11)]


def load_artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def format_grid(value: float) -> float:
    return round(float(value), 1)


def build_cism4_counts(
    artifact: dict,
    labels: list[str],
    grid: list[float],
    p_value_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    discriminator = artifact["discriminator"]
    patient_classes = load_patient_classes(discriminator, labels)
    motifs_df = discriminator.cism.motifs_dataset
    cohort_patients = patient_classes.index.tolist()
    summary = discover_hard_discriminative_motifs(
        motifs_df=motifs_df,
        patient_classes=patient_classes,
        cohort_patients=cohort_patients,
        p_value_threshold=p_value_threshold,
    )

    rows = []
    for threshold in grid:
        row = {"method": "CISM4", "stringency": threshold}
        total = 0
        for label in labels:
            count = int(
                (
                    (summary[DiscriminativeMotifs.PATIENT_CLASS] == label)
                    & (summary["patient_percentage"] >= threshold)
                ).sum()
            )
            row[f"{label}_motif_count"] = count
            total += count
        row["total_motif_count"] = total
        rows.append(row)

    motif_table = summary.rename(
        columns={
            DiscriminativeMotifs.PATIENT_CLASS: "motif_label",
            "patient_percentage": "stringency",
            "patient_count": "n_patients",
        }
    ).copy()
    motif_table.insert(0, "method", "CISM4")
    return pd.DataFrame(rows), motif_table


def build_cism4_auc(
    artifact: dict,
    labels: list[str],
    grid: list[float],
    max_features: int,
    p_value_threshold: float,
    random_state: int,
) -> pd.DataFrame:
    discriminator = artifact["discriminator"]
    patient_classes = load_patient_classes(discriminator, labels)
    motifs_df = discriminator.cism.motifs_dataset
    cohort_patients = patient_classes.index.tolist()
    class_series = patient_classes.loc[cohort_patients, DiscriminativeMotifs.PATIENT_CLASS]
    patient_matrix = build_patient_motif_matrix(
        motifs_df[motifs_df["Patient_uId"].isin(cohort_patients)],
        patient_ids=cohort_patients,
    )

    fold_cache = []
    for held_out_patient in cohort_patients:
        train_patients = [patient for patient in cohort_patients if patient != held_out_patient]
        train_summary = discover_hard_discriminative_motifs(
            motifs_df=motifs_df,
            patient_classes=patient_classes,
            cohort_patients=train_patients,
            p_value_threshold=p_value_threshold,
        )
        fold_cache.append((held_out_patient, train_patients, train_summary))

    rows = []
    for threshold in grid:
        y_true_labels: list[str] = []
        y_pred_labels: list[str] = []
        y_true_binary: list[int] = []
        y_prob: list[float] = []
        selected_counts: list[int] = []
        skipped = 0
        for held_out_patient, train_patients, train_summary in fold_cache:
            y_train = class_series.loc[train_patients]
            if y_train.nunique() < 2:
                skipped += 1
                continue
            selected_by_label = select_motifs_by_label(
                discriminative_summary=train_summary,
                labels=labels,
                stringency=threshold,
                max_features=max_features,
            )
            selected = list(
                dict.fromkeys(motif_id for motif_ids in selected_by_label.values() for motif_id in motif_ids)
            )
            selected_counts.append(len(selected))
            if not selected:
                skipped += 1
                continue

            x_train = patient_matrix.loc[train_patients].reindex(columns=selected, fill_value=0.0)
            x_test = patient_matrix.loc[[held_out_patient]].reindex(columns=selected, fill_value=0.0)
            clf = build_framework_random_forest(random_state=random_state)
            clf.fit(x_train.values, y_train.values)
            pred = clf.predict(x_test.values)[0]
            prob = clf.predict_proba(x_test.values)[0]
            pos_idx = list(clf.classes_).index(labels[0])

            true_label = class_series.loc[held_out_patient]
            y_true_labels.append(true_label)
            y_pred_labels.append(pred)
            y_true_binary.append(int(true_label == labels[0]))
            y_prob.append(float(prob[pos_idx]))

        auc = roc_auc_score(y_true_binary, y_prob) if len(set(y_true_binary)) == 2 else np.nan
        rows.append(
            {
                "method": "CISM4",
                "stringency": threshold,
                "auc": auc,
                "accuracy": accuracy_score(y_true_labels, y_pred_labels) if y_true_labels else np.nan,
                "balanced_accuracy": balanced_accuracy_score(y_true_labels, y_pred_labels)
                if len(set(y_true_labels)) == 2
                else np.nan,
                "valid_folds": len(y_true_labels),
                "skipped_folds": skipped,
                "mean_selected_motifs": float(np.mean(selected_counts)) if selected_counts else 0.0,
                "max_features_per_class": max_features,
                "positive_label": labels[0],
            }
        )
    return pd.DataFrame(rows)


def smore_counts_on_grid(exclusive: pd.DataFrame, labels: list[str], grid: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in grid:
        kept = exclusive[exclusive["stringency"] >= threshold]
        counts = kept["exclusive_to"].value_counts()
        row = {"method": "SMORE", "stringency": threshold}
        total = 0
        for label in labels:
            count = int(counts.get(label, 0))
            row[f"{label}_motif_count"] = count
            total += count
        row["total_motif_count"] = total
        rows.append(row)
    return pd.DataFrame(rows)


def _smore_train_summary(train_labeled: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    class_sizes = (
        train_labeled[["patient number", "class"]]
        .drop_duplicates()
        .groupby("class", observed=True)["patient number"]
        .nunique()
    )
    counts = (
        train_labeled.groupby(["seed", "class"], observed=True)["patient number"]
        .nunique()
        .reset_index(name="n_patients")
    )
    means = (
        train_labeled.groupby(["seed", "class"], observed=True)["frequency"]
        .mean()
        .reset_index(name="mean_frequency")
    )
    summary = counts.merge(means, on=["seed", "class"], how="left")
    present_classes = summary.groupby("seed", observed=True)["class"].nunique()
    one_class_seeds = present_classes[present_classes == 1].index
    summary = summary[summary["seed"].isin(one_class_seeds)].copy()
    summary["patient_percentage"] = summary.apply(
        lambda row: row["n_patients"] / class_sizes[row["class"]],
        axis=1,
    )
    summary = summary[summary["class"].isin(labels)].copy()
    return summary.sort_values(
        ["class", "patient_percentage", "n_patients", "mean_frequency"],
        ascending=[True, False, False, False],
    )


def _select_smore_motifs(summary: pd.DataFrame, labels: list[str], threshold: float, max_features: int) -> list[str]:
    selected: list[str] = []
    for label in labels:
        label_summary = summary[
            (summary["class"] == label) & (summary["patient_percentage"] >= threshold)
        ].head(max_features)
        selected.extend(label_summary["seed"].tolist())
    return list(dict.fromkeys(selected))


def build_smore_auc(
    labeled: pd.DataFrame,
    labels: list[str],
    grid: list[float],
    max_features: int,
    random_state: int,
) -> pd.DataFrame:
    agg = (
        labeled.groupby(["patient number", "seed"], observed=True)["frequency"]
        .mean()
        .reset_index(name="motif_score")
    )
    pivot = agg.pivot(index="patient number", columns="seed", values="motif_score").fillna(0.0)
    patient_class = (
        labeled[["patient number", "class"]]
        .drop_duplicates()
        .set_index("patient number")["class"]
    )
    patients = pivot.index.intersection(patient_class.index)
    pivot = pivot.loc[patients]
    patient_class = patient_class.loc[patients]
    fold_cache = []
    for train_idx, test_idx in LeaveOneOut().split(patients):
        train_patients = patients[train_idx]
        test_patient = patients[test_idx[0]]
        train_labeled = labeled[labeled["patient number"].isin(train_patients)]
        fold_cache.append((train_patients, test_patient, _smore_train_summary(train_labeled, labels)))

    rows = []
    for threshold in grid:
        y_true: list[int] = []
        y_pred: list[int] = []
        y_prob: list[float] = []
        selected_counts: list[int] = []
        skipped = 0
        for train_patients, test_patient, summary in fold_cache:
            selected = _select_smore_motifs(summary, labels, threshold, max_features)
            selected_counts.append(len(selected))
            if not selected:
                skipped += 1
                continue

            y_train_labels = patient_class.loc[train_patients]
            if y_train_labels.nunique() < 2:
                skipped += 1
                continue
            x_train = pivot.loc[train_patients, selected]
            x_test = pivot.loc[[test_patient], selected]

            clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            clf.fit(x_train.values, y_train_labels.values)
            prob = clf.predict_proba(x_test.values)[0]
            pos_idx = list(clf.classes_).index(labels[0])
            pred = clf.predict(x_test.values)[0]

            y_true.append(int(patient_class.loc[test_patient] == labels[0]))
            y_pred.append(int(pred == labels[0]))
            y_prob.append(float(prob[pos_idx]))

        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else np.nan
        rows.append(
            {
                "method": "SMORE",
                "stringency": threshold,
                "auc": auc,
                "accuracy": accuracy_score(y_true, y_pred) if y_true else np.nan,
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
                if len(set(y_true)) == 2
                else np.nan,
                "valid_folds": len(y_true),
                "skipped_folds": skipped,
                "mean_selected_motifs": float(np.mean(selected_counts)) if selected_counts else 0.0,
                "max_features_per_class": max_features,
                "positive_label": labels[0],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--smore-results", type=Path, default=DEFAULT_SMORE_RESULTS)
    parser.add_argument("--smore-metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    parser.add_argument("--max-features", type=int, default=30)
    parser.add_argument("--p-value-threshold", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(args.labels)

    artifact = load_artifact(args.artifact_path)
    cism_counts, cism_motifs = build_cism4_counts(
        artifact=artifact,
        labels=labels,
        grid=GRID,
        p_value_threshold=args.p_value_threshold,
    )
    cism_auc = build_cism4_auc(
        artifact=artifact,
        labels=labels,
        grid=GRID,
        max_features=args.max_features,
        p_value_threshold=args.p_value_threshold,
        random_state=args.random_state,
    )

    smore_labeled = load_labeled_smore(args.smore_results, args.smore_metadata_dir)
    smore_motifs = motif_stringency(smore_labeled).rename(
        columns={
            "motif": "ID",
            "exclusive_to": "motif_label",
            "total_patients": "n_patients",
        }
    )
    smore_motifs.insert(0, "method", "SMORE")
    smore_counts = smore_counts_on_grid(smore_motifs.rename(columns={"motif_label": "exclusive_to"}), labels, GRID)
    smore_auc = build_smore_auc(
        labeled=smore_labeled,
        labels=labels,
        grid=GRID,
        max_features=args.max_features,
        random_state=args.random_state,
    )

    cism_counts.to_csv(args.output_dir / "cism4_melanoma_motif_count_vs_stringency_grid.csv", index=False)
    cism_auc.to_csv(args.output_dir / "cism4_melanoma_auc_vs_stringency_grid.csv", index=False)
    cism_motifs.to_csv(args.output_dir / "cism4_melanoma_discriminative_motifs.csv", index=False)

    smore_counts.to_csv(args.output_dir / "smore_motif_count_vs_stringency_grid.csv", index=False)
    smore_auc.to_csv(args.output_dir / "smore_auc_vs_stringency_grid.csv", index=False)
    smore_motifs.to_csv(args.output_dir / "smore_discriminative_motifs_grid_source.csv", index=False)

    pd.concat([cism_counts, smore_counts], ignore_index=True).to_csv(
        args.output_dir / "combined_motif_count_vs_stringency_grid.csv",
        index=False,
    )
    pd.concat([cism_auc, smore_auc], ignore_index=True).to_csv(
        args.output_dir / "combined_auc_vs_stringency_grid.csv",
        index=False,
    )

    print(f"Wrote clean stringency CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
