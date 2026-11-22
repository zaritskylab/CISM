#!/usr/bin/env python3
"""Recompute LOOCV fold-union motif-count sweeps for nearby permutation seeds."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import ast
import copy
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cism.cism import DiscriminativeMotifs


ROOT = _REPO_ROOT
RUNS = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/cism_label_permutation_test"
ARTIFACT = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/serialized/Melanoma_cism_ready.pkl"


def patient_labels(run_dir: Path) -> pd.Series:
    filename = "patient_class_original.csv" if run_dir.name == "from_artifact" else "patient_class_permuted.csv"
    frame = pd.read_csv(run_dir / filename, header=None, names=["patient", "patient_class"])
    return frame.set_index("patient").patient_class


def fold_union_curve(
    discriminator,
    motif_rows: pd.DataFrame,
    patients: list[str],
    labels: pd.Series,
    feature_conf,
    thresholds: np.ndarray,
    saved_features: pd.DataFrame,
    optimum: float,
) -> list[dict]:
    unions = {float(threshold): set() for threshold in thresholds}
    matched_folds = 0
    for held_out in patients:
        train = motif_rows.loc[motif_rows.Patient_uId != held_out].drop(
            columns=["patient_class"], errors="ignore").copy()
        train["patient_class"] = pd.Categorical(train.Patient_uId.map(labels))
        candidate_rows, _ = discriminator._extract_discriminative(
            train, DiscriminativeMotifs.PATIENT_CLASS, "ID", list(discriminator.common_cells.keys()))
        for threshold in thresholds:
            feature_conf.shared_percentage = float(threshold)
            _, motif_ids, _ = discriminator._extract_features(candidate_rows, feature_conf)
            unions[float(threshold)].update(motif_ids)
            if np.isclose(threshold, optimum, rtol=0, atol=1e-12):
                saved = set(ast.literal_eval(saved_features.loc[held_out, "features"]))
                matched_folds += int(set(motif_ids) == saved)
    if matched_folds != len(patients):
        print(f"Tie-sensitive fold selections matched: {matched_folds}/{len(patients)}", flush=True)
    saved_union = set().union(*(set(ast.literal_eval(value)) for value in saved_features.features))
    reconstructed_union = next(ids for threshold, ids in unions.items()
                               if np.isclose(threshold, optimum, rtol=0, atol=1e-12))
    if reconstructed_union != saved_union:
        raise ValueError("Fold-union IDs at the saved optimum differ from the stored selection")
    return [
        {"stringency": threshold, "n_fold_union_motifs": len(ids),
         "selected_motif_ids": ";".join(map(str, sorted(ids)))}
        for threshold, ids in sorted(unions.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--nearest", type=int, default=10)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()
    if not 0 < args.step <= 0.1:
        parser.error("--step must be in (0, 0.1]")

    report = args.runs_dir / "report"
    nearest = pd.read_csv(report / f"nearest_{args.nearest}_optimal_stringency_shuffles.csv")
    original_dir = args.runs_dir / "original/from_artifact"
    run_dirs = [("original", original_dir)] + [
        (f"seed_{int(seed)}", args.runs_dir / "shuffled" / f"seed_{int(seed)}")
        for seed in nearest.seed
    ]
    grid = np.round(np.arange(0, 0.651, args.step), 10)

    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    discriminator = artifact["discriminator"]
    base_conf = artifact.get("feature_conf", artifact.get("best_feature_conf"))
    patients = discriminator.get_patients_class(["NN", "NP"]).index.tolist()
    motifs = discriminator.cism.motifs_dataset
    motif_rows = motifs.loc[motifs.Patient_uId.isin(patients) & (motifs.p_value < 0.05)].copy()
    curves = []
    for run_name, run_dir in run_dirs:
        optimum = float(pd.read_csv(run_dir / "trial_summary.csv").optimal_stringency.iloc[0])
        thresholds = np.unique(np.r_[grid, optimum])
        saved = pd.read_csv(run_dir / "optimal_motif_space_features.csv").set_index("test_patient_id")
        if set(saved.index) != set(patients):
            raise ValueError(f"Patient mismatch: {run_dir}")
        labels = patient_labels(run_dir)
        if labels.loc[patients].isna().any():
            raise ValueError(f"Missing labels: {run_dir}")
        feature_conf = copy.copy(base_conf)
        rows = fold_union_curve(discriminator, motif_rows, patients, labels,
                                feature_conf, thresholds, saved, optimum)
        saved_count = int(pd.read_csv(run_dir / "trial_summary.csv").n_selected_unique_motifs.iloc[0])
        optimum_rows = [row for row in rows if np.isclose(row["stringency"], optimum, rtol=0, atol=1e-12)]
        if len(optimum_rows) != 1 or optimum_rows[0]["n_fold_union_motifs"] != saved_count:
            raise ValueError(f"Optimum count does not match saved run: {run_dir}")
        for row in rows:
            row["run"] = run_name
            row["optimal_stringency"] = optimum
            row["at_saved_optimum"] = bool(np.isclose(row["stringency"], optimum, rtol=0, atol=1e-12))
        curves.extend(rows)
        print(f"{run_name}: {len(rows)} thresholds; optimum {optimum:.6f} -> {saved_count} motifs", flush=True)

    data = pd.DataFrame(curves)
    data.to_csv(report / f"nearest_{args.nearest}_fold_union_motif_sweeps.csv", index=False)
    original_point = data.loc[(data.run == "original") & data.at_saved_optimum].iloc[0]
    fig, axes = plt.subplots(5, 2, figsize=(12.5, 16), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, seed in zip(axes.flat, nearest.seed):
        name = f"seed_{int(seed)}"
        subset = data.loc[data.run == name].sort_values("stringency")
        optimum = subset.loc[subset.at_saved_optimum].iloc[0]
        ax.plot(subset.stringency, subset.n_fold_union_motifs, color="#e09b42",
                linewidth=1.8, marker="o", markersize=2.6, label="Shuffle NN + NP")
        ax.scatter([optimum.stringency], [optimum.n_fold_union_motifs],
                   color="#e09b42", edgecolor="#744613", s=65, zorder=4,
                   label="Shuffle optimum")
        ax.scatter([original_point.stringency], [original_point.n_fold_union_motifs],
                   color="#b54435", edgecolor="#60251f", s=85, marker="D", zorder=5,
                   label="Unpermuted optimum")
        ax.set_title(f"Seed {int(seed)} | optimum: {int(optimum.n_fold_union_motifs)} motifs")
        ax.grid(color="#e1e5e9", alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Stringency")
    for ax in axes[:, 0]:
        ax.set_ylabel("Discriminative motifs (NN + NP fold union)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(report / f"nearest_{args.nearest}_fold_union_motif_sweeps.{extension}", dpi=250)
    plt.close(fig)

    for seed in nearest.seed:
        name = f"seed_{int(seed)}"
        subset = data.loc[data.run == name].sort_values("stringency")
        optimum = subset.loc[subset.at_saved_optimum].iloc[0]
        fig, ax = plt.subplots(figsize=(7.5, 5.1), constrained_layout=True)
        ax.plot(subset.stringency, subset.n_fold_union_motifs, color="#e09b42",
                linewidth=2, marker="o", markersize=3, label="Shuffle NN + NP")
        ax.scatter([optimum.stringency], [optimum.n_fold_union_motifs],
                   color="#e09b42", edgecolor="#744613", s=100, zorder=4,
                   label="Shuffle optimum")
        ax.scatter([original_point.stringency], [original_point.n_fold_union_motifs],
                   color="#b54435", edgecolor="#60251f", marker="D", s=110,
                   zorder=5, label="Unpermuted optimum")
        ax.set(title=f"Seed {int(seed)}: motifs vs stringency",
               xlabel="Stringency", ylabel="Discriminative motifs (NN + NP fold union)",
               xlim=(-0.01, 0.66), ylim=(0, None))
        ax.grid(color="#e1e5e9", alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(report / f"seed_{int(seed)}_fold_union_motifs_vs_stringency.png", dpi=250)
        plt.close(fig)
    print(f"Saved curves and figures to {report}")


if __name__ == "__main__":
    main()
