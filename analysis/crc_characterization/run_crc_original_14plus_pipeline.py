#!/usr/bin/env python3
"""Run FANMOD/CISM and patient-level CRC analyses on the complete 14+U dataset."""

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
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-original-14plus-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from cism import CISM, DiscriminativeFeatureKey, HardDiscriminativeFC, TissueStateDiscriminativeMotifs
from cism.cism import GetMotifsResult
from cism.helpers import create_weight_matrix_from_motifs
from pairwise import GraphReader, PairwiseAnalysis


def loocv_rf(features: pd.DataFrame, labels: pd.Series, seeds=range(5)) -> pd.DataFrame:
    rows = []
    x, y = features.to_numpy(), labels.loc[features.index].to_numpy()
    ids = features.index.to_numpy()
    for seed in seeds:
        predictions = []
        for train, test in LeaveOneOut().split(x):
            model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
            model.fit(x[train], y[train])
            clr_idx = int(np.flatnonzero(model.classes_ == "CLR")[0])
            predictions.append((int(ids[test][0]), str(y[test][0]), float(model.predict_proba(x[test])[0, clr_idx])))
        table = pd.DataFrame(predictions, columns=["Patient", "true_class", "p_CLR"])
        table["seed"] = seed
        table["roc_auc"] = roc_auc_score(table.true_class == "CLR", table.p_CLR)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--motif-size", type=int, choices=[3, 4], required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()

    output = args.output_root / f"motif_{args.motif_size}"
    fanmod_output = output / "fanmod_output"
    fanmod_cache = output / "fanmod_cache"
    analysis = output / "analysis_outputs"
    plots = analysis / "plots"
    serialized = output / "serialized"
    for path in [fanmod_output, fanmod_cache, analysis, plots, serialized]:
        path.mkdir(parents=True, exist_ok=True)

    color_map = pd.read_csv(args.dataset_dir / "node_color_map.csv").sort_values("color_id")
    cell_types = dict(zip(color_map.color_id.astype(int), color_map.node_color.astype(str)))
    if len(cell_types) != 15 or "Unidentified" not in cell_types.values():
        raise ValueError(f"Expected 14 groups plus Unidentified, found {cell_types}")
    labels = ["CLR", "DII"]
    class_map = {"CLR": "CLR", "DII": "DII"}

    cism = CISM(
        fanmod_exe="LocalFANMOD",
        fanmod_path=str(args.project_root / "FANMOD_binaries"),
        network_dataset_root_path=str(args.dataset_dir.parent),
        fanmod_output_root_path=str(fanmod_output),
        fanmod_cache_root_path=str(fanmod_cache),
        motif_size=args.motif_size,
        iterations=args.iterations,
    )
    cism.add_dataset(
        dataset_folder=args.dataset_dir.name,
        dataset_type="Disease", dataset_name="CRC",
        force_run_fanmod=True, force_parse=True,
        n_jobs=args.n_jobs, prefer="threads", require_patient_class=True,
    )
    if cism.motifs_dataset is None or cism.motifs_dataset.empty:
        raise RuntimeError("FANMOD completed without parsed motifs")

    discriminator = TissueStateDiscriminativeMotifs(
        cism=cism,
        tissue_state_csv_path=str(args.dataset_dir / "patient_class.csv"),
        tissue_state_to_string=class_map,
        common_cells_type=cell_types,
    )
    discover = discriminator.discover(
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        classes=labels,
    )
    discriminative = discover.get_discriminative_motifs().copy()
    discriminative.to_csv(analysis / "discriminative_motifs.csv", index=False)

    thresholds = np.round(np.arange(0.0, 0.951, 0.05), 2)
    sweep_rows = []
    count_rows = []
    for threshold in thresholds:
        for label in labels:
            count_rows.append({
                "stringency": float(threshold), "patient_class": label,
                "discriminative_motifs": int(discriminative.loc[
                    (discriminative.patient_class == label) &
                    (discriminative.patient_percentage > threshold), "ID"
                ].nunique()),
            })
        config = HardDiscriminativeFC(
            labels=labels,
            extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
            use_cells_type_composition=False, use_motifs=True,
            shared_percentage=float(threshold), max_class_features=30,
        )
        try:
            result = discriminator.analyze_motifs(
                feature_conf=config, exclude_patients=[], n_jobs=args.n_jobs,
                prefer="processes", random_state=np.random.RandomState(0),
            )
            sweep_rows.append({
                "stringency": float(threshold), "roc_auc": float(result.get_roc_auc_score()),
                "failed_folds": int((result.results.cFeatures == 0).sum()),
                "mean_selected_features": float(result.results.cFeatures.mean()), "status": "ok",
            })
            result.results.to_csv(analysis / f"loocv_stringency_{threshold:.2f}.csv")
        except Exception as exc:
            sweep_rows.append({"stringency": float(threshold), "roc_auc": np.nan,
                               "failed_folds": np.nan, "mean_selected_features": 0,
                               "status": f"{type(exc).__name__}: {exc}"})

    sweep = pd.DataFrame(sweep_rows)
    counts = pd.DataFrame(count_rows)
    sweep.to_csv(analysis / "auc_by_stringency.csv", index=False)
    counts.to_csv(analysis / "motif_count_vs_stringency.csv", index=False)
    valid = sweep.dropna(subset=["roc_auc"])
    if valid.empty:
        raise RuntimeError("No stringency produced a valid LOOCV AUC")
    best = valid.sort_values(["roc_auc", "stringency"], ascending=[False, False]).iloc[0]
    best_threshold = float(best.stringency)

    best_config = HardDiscriminativeFC(
        labels=labels,
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        use_cells_type_composition=False, use_motifs=True,
        shared_percentage=best_threshold, max_class_features=30,
    )
    motif_space = discriminator.get_features(
        feature_conf=best_config, exclude_patients=[], n_jobs=args.n_jobs, prefer="processes"
    )
    motif_space.to_csv(analysis / "selected_motif_space_features.csv", index=False)
    motif_result = GetMotifsResult(motif_space_features=motif_space, discriminator=discriminator, classes=labels)
    motif_weights = motif_result.get_motifs_mean_weight()
    motif_weights.to_csv(analysis / "selected_motif_mean_weight.csv")

    full_graph = GraphReader.get_graphs(
        full_graph_df=pd.DataFrame(), raw_data_folder=str(args.dataset_dir.parent),
        raw_data_folder_type=f"/{args.dataset_dir.name}", disease="CRC", cells_type=cell_types,
    )
    pairwise = PairwiseAnalysis(
        tissue_state_csv_path=str(args.dataset_dir / "patient_class.csv"),
        tissue_state_to_string=class_map,
    )
    motifs_by_class = motif_result.get_motifs_by_class()
    for label in labels:
        pairwise.get_pairwise_counter(full_graph, label, cell_types, normalize=True).to_csv(
            analysis / f"general_pairwise_{label}.csv"
        )
        create_weight_matrix_from_motifs(
            discriminator=discriminator, cells_type=cell_types,
            cell_identity_to_motif_hash=motif_result.cell_identity_to_motif_hash,
            motifs_ids=motifs_by_class[label], motifs_weight=motif_weights,
        ).to_csv(analysis / f"motif_induced_pairwise_{label}.csv")

    mapped_cells = pd.read_csv(args.dataset_dir / "mapped_cells.csv.gz", usecols=["patient_id", "cell_type"])
    singleton = mapped_cells.groupby(["patient_id", "cell_type"], observed=True).size().unstack(fill_value=0)
    singleton = singleton.div(singleton.sum(axis=1), axis=0)
    patient_labels = pd.read_csv(args.dataset_dir / "patient_class.csv", header=None, names=["uid", "class"])
    patient_labels.index = patient_labels.uid.str.replace("CRC", "", regex=False).astype(int)
    singleton_predictions = loocv_rf(singleton, patient_labels["class"])
    singleton_predictions.to_csv(analysis / "singleton_loocv_predictions.csv", index=False)

    pair_features = {}
    triangle = np.triu_indices(len(cell_types), k=0)
    for patient_uid, group in full_graph.groupby("Patient_uId", observed=True):
        matrix = group.pairwise_count.sum()
        matrix = matrix / matrix.sum()
        pair_features[int(str(patient_uid).replace("CRC", ""))] = matrix[triangle]
    pair_features = pd.DataFrame.from_dict(pair_features, orient="index").sort_index()
    pair_predictions = loocv_rf(pair_features, patient_labels["class"])
    pair_predictions.to_csv(analysis / "pairwise_loocv_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(sweep.stringency, sweep.roc_auc, marker="o", color="#3569a8")
    ax.axvline(best_threshold, color="#c73e3a", linestyle="--")
    ax.set(xlabel="Discrimination stringency", ylabel="LOOCV ROC AUC", ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(plots / "auc_vs_stringency.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for label, color in [("CLR", "#347fba"), ("DII", "#c74d4d")]:
        part = counts[counts.patient_class == label]
        ax.plot(part.stringency, part.discriminative_motifs, marker="o", label=label, color=color)
    ax.set_yscale("log"); ax.set(xlabel="Discrimination stringency", ylabel="# discriminative motifs")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(plots / "motifs_vs_stringency.png", dpi=300); plt.close(fig)

    raw_cism_path = serialized / f"cism_CRC_original_14plusU_{args.motif_size}.pkl"
    with raw_cism_path.open("wb") as handle:
        pickle.dump(cism, handle)
    bundle_path = serialized / f"crc_original_14plusU_{args.motif_size}_pipeline.pkl"
    with bundle_path.open("wb") as handle:
        pickle.dump({"cism": cism, "discriminator": discriminator, "best_feature_conf": best_config,
                     "common_cells_type": cell_types, "labels": labels}, handle)

    summary = {
        "motif_size": args.motif_size, "iterations": args.iterations,
        "patients": int(singleton.shape[0]), "fovs": int(len(full_graph)),
        "cell_types": len(cell_types), "includes_unidentified": "Unidentified" in cell_types.values(),
        "motifs_dataset_rows": int(len(cism.motifs_dataset)),
        "discriminative_motif_ids": int(discriminative.ID.nunique()),
        "optimal_stringency": best_threshold, "optimal_auc": float(best.roc_auc),
        "fixed_0.55_auc": (None if sweep.loc[sweep.stringency == .55, "roc_auc"].isna().all()
                           else float(sweep.loc[sweep.stringency == .55, "roc_auc"].iloc[0])),
        "singleton_auc_mean": float(singleton_predictions.groupby("seed").roc_auc.first().mean()),
        "pairwise_auc_mean": float(pair_predictions.groupby("seed").roc_auc.first().mean()),
        "raw_cism_artifact": str(raw_cism_path), "pipeline_artifact": str(bundle_path),
    }
    (analysis / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
