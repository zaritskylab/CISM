#!/usr/bin/env python3
"""Evaluate CRC classification from patient-level cell-type frequencies."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut


ROOT = _REPO_ROOT
DEFAULT_DATA = Path("/groups/assafza_group/assafza/Datasets/CRC")
DEFAULT_OUTPUT = ROOT / "analysis/outputs/crc_singleton_classifier"


def load_patient_features(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    cells = pd.read_csv(data_dir / "CRC_cells.csv", usecols=["fov", "class"])
    patient = cells["fov"].str.extract(r"^P(\d+)_", expand=False)
    if patient.isna().any():
        bad = cells.loc[patient.isna(), "fov"].drop_duplicates().head().tolist()
        raise ValueError(f"Could not extract patient IDs from FOVs: {bad}")
    cells["Patient"] = patient.astype(int)

    counts = cells.groupby(["Patient", "class"], observed=True).size().unstack(fill_value=0)
    frequencies = counts.div(counts.sum(axis=1), axis=0).sort_index()
    frequencies.columns.name = "cell_type"

    labels = pd.read_csv(data_dir / "CRC_patient_class.csv")[["Patient", "Group"]].dropna()
    labels["Patient"] = labels["Patient"].astype(int)
    labels["Group"] = pd.to_numeric(labels["Group"], errors="raise").astype(int)
    if labels["Patient"].duplicated().any():
        raise ValueError("Patient labels are not unique")
    labels = labels.set_index("Patient")["Group"].map({1: "CLR", 2: "DII"}).sort_index()
    if labels.isna().any() or set(labels.index) != set(frequencies.index):
        raise ValueError("Cell table and true-label table contain different patient cohorts")
    return frequencies.loc[labels.index], labels


def evaluate(features: pd.DataFrame, labels: pd.Series, seeds: list[int], trees: int):
    predictions = []
    patient_ids = features.index.to_numpy()
    x = features.to_numpy()
    y = labels.to_numpy()
    splitter = LeaveOneOut()
    for seed in seeds:
        for train, test in splitter.split(x):
            model = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=1)
            model.fit(x[train], y[train])
            clr_index = int(np.flatnonzero(model.classes_ == "CLR")[0])
            predictions.append(
                {
                    "seed": seed,
                    "Patient": int(patient_ids[test][0]),
                    "true_class": str(y[test][0]),
                    "predicted_class": str(model.predict(x[test])[0]),
                    "p_CLR": float(model.predict_proba(x[test])[0, clr_index]),
                }
            )
    prediction_table = pd.DataFrame(predictions)
    metrics = []
    for seed, table in prediction_table.groupby("seed", sort=True):
        metrics.append(
            {
                "seed": int(seed),
                "roc_auc": float(roc_auc_score(table.true_class == "CLR", table.p_CLR)),
                "accuracy": float(accuracy_score(table.true_class, table.predicted_class)),
            }
        )
    return prediction_table, pd.DataFrame(metrics)


def plot_roc(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    grid = np.linspace(0, 1, 201)
    interpolated = []
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for _, table in predictions.groupby("seed", sort=True):
        fpr, tpr, _ = roc_curve(table.true_class == "CLR", table.p_CLR)
        ax.plot(fpr, tpr, color="#7da7d9", alpha=.28, linewidth=1)
        interpolated.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(interpolated, axis=0)
    ax.plot(grid, mean_tpr, color="#2166ac", linewidth=2.4,
            label=f"Singleton RF: AUC {metrics.roc_auc.mean():.3f} ± {metrics.roc_auc.std(ddof=0):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"crc_singleton_loocv_roc.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()
    if args.trees < 1 or not args.seeds:
        raise ValueError("At least one seed and one tree are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, labels = load_patient_features(args.data_dir)
    predictions, metrics = evaluate(features, labels, args.seeds, args.trees)
    features.to_csv(args.output_dir / "patient_cell_type_frequencies.csv")
    labels.rename("patient_class").to_csv(args.output_dir / "true_patient_labels.csv")
    predictions.to_csv(args.output_dir / "loocv_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_by_seed.csv", index=False)
    plot_roc(predictions, metrics, args.output_dir)

    summary = {
        "analysis": "Random forest on patient-level cell-type frequencies",
        "labels": "Original CRC labels; no permutation or shuffling",
        "validation": "Leave one patient out",
        "patients": int(len(labels)),
        "class_counts": labels.value_counts().to_dict(),
        "features": int(features.shape[1]),
        "seeds": args.seeds,
        "trees": args.trees,
        "positive_class": "CLR",
        "auc_input": "Held-out CLR probabilities",
        "roc_auc_mean": float(metrics.roc_auc.mean()),
        "roc_auc_sd_population": float(metrics.roc_auc.std(ddof=0)),
        "roc_auc_min": float(metrics.roc_auc.min()),
        "roc_auc_max": float(metrics.roc_auc.max()),
        "accuracy_mean": float(metrics.accuracy.mean()),
        "accuracy_sd_population": float(metrics.accuracy.std(ddof=0)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
