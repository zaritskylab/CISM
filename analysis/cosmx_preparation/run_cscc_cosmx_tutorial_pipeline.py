#!/usr/bin/env python3
"""Run the cSCC CosMx dataset through tutorial pipeline parts 1-3.

By default this script performs only the pre-run approval checks. Pass
``--approved`` after reviewing the prepared dataset to run FANMOD/CISM.
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
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cism import (
    CISM,
    DiscriminativeFeatureKey,
    HardDiscriminativeFC,
    TissueStateDiscriminativeMotifs,
    rank_motifs_by_stringency_count,
    validate_network_dataset_directory,
)
from cism.cism import GetMotifsResult
from cism.helpers import create_weight_matrix_from_motifs
from pairwise import GraphReader, PairwiseAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Run FANMOD/CISM after the prepared dataset has been approved.",
    )
    parser.add_argument(
        "--dataset-folder",
        default="cSCC_CosMx_baseline_reconstructed",
        help="Folder under data/ containing CISM graph files.",
    )
    parser.add_argument(
        "--prep-summary-dir",
        default="tutorial_runtime/cscc_cosmx_baseline_reconstructed",
        help="Preparation summary directory containing node_color_map.csv.",
    )
    parser.add_argument(
        "--runtime-name",
        default="cscc_cosmx_pipeline",
        help="Runtime subdirectory under tutorial_runtime for FANMOD/CISM outputs.",
    )
    parser.add_argument("--dataset-name", default="cSCC")
    parser.add_argument("--dataset-type", default="Disease")
    parser.add_argument("--motif-size", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--tuning-trials", type=int, default=3)
    parser.add_argument("--force-run-fanmod", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-parse", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_common_cells(color_map_path: Path) -> dict[int, str]:
    color_map = pd.read_csv(color_map_path)
    return {
        int(row["color_id"]): str(row["node_color"])
        for _, row in color_map.sort_values("color_id").iterrows()
    }


def write_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)


def main() -> None:
    args = parse_args()

    project_root = _REPO_ROOT
    data_root = project_root / "data"
    dataset_dir = data_root / args.dataset_folder
    prep_summary_dir = project_root / args.prep_summary_dir
    runtime_root = project_root / "tutorial_runtime" / args.runtime_name
    fanmod_output_root = runtime_root / "fanmod_output"
    fanmod_cache_root = runtime_root / "fanmod_cache"
    serialized_root = runtime_root / "serialized"
    analysis_root = runtime_root / "analysis_outputs"
    plot_root = analysis_root / "plots"

    fanmod_path = project_root / "FANMOD_binaries"
    fanmod_exe = "LocalFANMOD"
    tissue_state_csv_path = dataset_dir / "patient_class.csv"
    node_color_map_path = prep_summary_dir / "node_color_map.csv"

    files = validate_network_dataset_directory(dataset_dir, require_patient_class=True)
    common_cells_type = load_common_cells(node_color_map_path)
    labels = ["Responder", "NonResponder"]
    tissue_state_to_string = {"Responder": "Responder", "NonResponder": "NonResponder"}

    preflight = {
        "dataset_dir": str(dataset_dir),
        "dataset_folder": args.dataset_folder,
        "dataset_name": args.dataset_name,
        "graph_files": len(files),
        "patient_class_csv": str(tissue_state_csv_path),
        "node_colors": len(common_cells_type),
        "fanmod_binary": str(fanmod_path / fanmod_exe),
        "runtime_root": str(runtime_root),
        "approved": args.approved,
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))

    if not args.approved:
        print("Approval required: rerun with --approved to execute FANMOD/CISM tutorial parts 2-3.")
        return

    for directory in [fanmod_output_root, fanmod_cache_root, serialized_root, analysis_root, plot_root]:
        directory.mkdir(parents=True, exist_ok=True)

    cism = CISM(
        fanmod_exe=fanmod_exe,
        fanmod_path=str(fanmod_path),
        network_dataset_root_path=str(data_root),
        fanmod_output_root_path=str(fanmod_output_root),
        fanmod_cache_root_path=str(fanmod_cache_root),
        motif_size=args.motif_size,
        iterations=args.iterations,
    )

    cism.add_dataset(
        dataset_folder=args.dataset_folder,
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        force_run_fanmod=args.force_run_fanmod,
        force_parse=args.force_parse,
        n_jobs=args.n_jobs,
        require_patient_class=True,
    )
    if cism.motifs_dataset is None or cism.motifs_dataset.empty:
        raise RuntimeError("CISM completed but produced no parsed motifs.")

    discriminator = TissueStateDiscriminativeMotifs(
        cism=cism,
        tissue_state_csv_path=str(tissue_state_csv_path),
        tissue_state_to_string=tissue_state_to_string,
        common_cells_type=common_cells_type,
    )

    discover_result = discriminator.discover(
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        classes=labels,
    )
    discriminative_motifs_df = discover_result.get_discriminative_motifs().copy()
    discriminative_motifs_df.to_csv(analysis_root / "discriminative_motifs.csv", index=False)

    motif_count_vs_stringency = (
        discriminative_motifs_df.groupby("ID", observed=True)
        .agg(
            stringency=("patient_percentage", "max"),
            total_count=("Count", "sum"),
            patient_class=("patient_class", lambda values: values.mode().iloc[0]),
        )
        .reset_index()
    )
    motif_count_vs_stringency.to_csv(analysis_root / "motif_count_vs_stringency.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=motif_count_vs_stringency,
        x="stringency",
        y="total_count",
        hue="patient_class",
        s=80,
    )
    plt.xlabel("Stringency (patient_percentage)")
    plt.ylabel("Motif count (sum of Count)")
    plt.title("cSCC CosMx Motif Count Vs Stringency")
    plt.tight_layout()
    plt.savefig(plot_root / "motif_count_vs_stringency.png", dpi=200)
    plt.close()

    feature_conf_template = HardDiscriminativeFC(
        labels=labels,
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        use_cells_type_composition=False,
        use_motifs=True,
        shared_percentage=0.5,
        max_class_features=10,
    )
    tuning_result = discriminator.tune_stringency(
        feature_conf=feature_conf_template,
        n_trials=args.tuning_trials,
        metric="roc_auc",
        random_state=0,
        n_jobs=args.n_jobs,
        prefer="processes",
        shared_percentage_range=(0.35, 0.5),
        max_class_features_range=(30, 30),
    )
    tuning_trials_df = tuning_result.trials_dataframe()
    tuning_trials_df.to_csv(analysis_root / "stringency_tuning_trials.csv", index=False)

    best_feature_conf = HardDiscriminativeFC(
        labels=labels,
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        use_cells_type_composition=False,
        use_motifs=True,
        shared_percentage=tuning_result.best_params["shared_percentage"],
        max_class_features=tuning_result.best_params["max_class_features"],
    )
    best_analysis_result = discriminator.analyze_motifs(
        feature_conf=best_feature_conf,
        exclude_patients=[],
        n_jobs=args.n_jobs,
        prefer="processes",
    )
    best_analysis_result.results.to_csv(analysis_root / "selected_stringency_loocv_results.csv")

    top5_motifs_df = rank_motifs_by_stringency_count(discover_result=discover_result, top_k=5)
    top5_motifs_df.to_csv(analysis_root / "top5_motifs_by_stringency_count.csv", index=False)

    motif_space_features = discriminator.get_features(
        feature_conf=best_feature_conf,
        exclude_patients=[],
        n_jobs=args.n_jobs,
        prefer="processes",
    )
    motif_space_features.to_csv(analysis_root / "selected_motif_space_features.csv", index=False)
    motif_result = GetMotifsResult(
        motif_space_features=motif_space_features,
        discriminator=discriminator,
        classes=labels,
    )
    motif_mean_weight = motif_result.get_motifs_mean_weight()
    motif_mean_weight.to_csv(analysis_root / "selected_motif_mean_weight.csv")

    full_graph_df = GraphReader.get_graphs(
        full_graph_df=pd.DataFrame(),
        raw_data_folder=str(data_root),
        raw_data_folder_type=f"/{args.dataset_folder}",
        disease=args.dataset_name,
        cells_type=common_cells_type,
    )
    pairwise_analysis = PairwiseAnalysis(
        tissue_state_csv_path=str(tissue_state_csv_path),
        tissue_state_to_string=tissue_state_to_string,
    )

    motifs_by_class = motif_result.get_motifs_by_class()
    for label in labels:
        write_matrix(
            pairwise_analysis.get_pairwise_counter(
                full_graph_df=full_graph_df,
                patient_class=label,
                cells_type=common_cells_type,
                normalize=True,
            ),
            analysis_root / f"general_pairwise_{label}.csv",
        )
        write_matrix(
            create_weight_matrix_from_motifs(
                discriminator=discriminator,
                cells_type=common_cells_type,
                cell_identity_to_motif_hash=motif_result.cell_identity_to_motif_hash,
                motifs_ids=motifs_by_class[label],
                motifs_weight=motif_mean_weight,
            ),
            analysis_root / f"motif_induced_pairwise_{label}.csv",
        )

    artifact_path = serialized_root / f"{args.dataset_folder}_cism_ready.pkl"
    artifact = {
        "cism": cism,
        "discriminator": discriminator,
        "best_feature_conf": best_feature_conf,
        "tuning_best_params": tuning_result.best_params,
        "tuning_best_score": tuning_result.best_score,
        "tuning_trials_df": tuning_trials_df,
        "network_dataset_root_path": str(data_root),
        "dataset_folder": args.dataset_folder,
        "dataset_name": args.dataset_name,
        "fanmod_path": str(fanmod_path),
        "fanmod_exe": fanmod_exe,
        "fanmod_output_root_path": str(fanmod_output_root),
        "fanmod_cache_root_path": str(fanmod_cache_root),
        "tissue_state_csv_path": str(tissue_state_csv_path),
        "tissue_state_to_string": tissue_state_to_string,
        "tissue_state_mode": "pathologic_response_baseline_reconstructed",
        "common_cells_type": common_cells_type,
        "labels": labels,
        "preparation_summary_path": str(prep_summary_dir / "preparation_summary.json"),
    }
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    run_summary = {
        "artifact_path": str(artifact_path),
        "analysis_output_root": str(analysis_root),
        "motifs_dataset_rows": int(cism.motifs_dataset.shape[0]),
        "discriminative_motif_rows": int(discriminative_motifs_df.shape[0]),
        "top5_motif_ids": [int(motif_id) for motif_id in top5_motifs_df["ID"].tolist()],
        "best_params": tuning_result.best_params,
        "best_tuning_roc_auc": float(tuning_result.best_score),
        "selected_setting_roc_auc": float(best_analysis_result.get_roc_auc_score()),
    }
    with (analysis_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)
    print(json.dumps(run_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
