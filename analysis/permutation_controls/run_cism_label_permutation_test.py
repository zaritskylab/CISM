from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, power_divergence, ttest_1samp
from joblib import Parallel, delayed
from tqdm.auto import tqdm

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.cism import AnalyzeMotifsResult, DiscriminativeFeatureKey, DiscriminativeMotifs, GetMotifsResult
from cism.helpers import string_base64_pickle
from cism.optimization import OptunaTuningResult, _clone_feature_conf
from pairwise import GraphReader


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "cism_label_permutation_test"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and aggregate CISM label-shuffle permutation experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one shuffled-label CISM permutation trial.")
    run_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--model-seed", type=int, default=None,
                            help="Classifier seed; defaults to the permutation seed for compatibility.")
    run_parser.add_argument(
        "--no-shuffle-labels",
        action="store_false",
        dest="shuffle_labels",
        help="Debug only: run without permuting patient labels.",
    )
    run_parser.set_defaults(shuffle_labels=True)
    run_parser.add_argument("--force", action="store_true", help="Overwrite an existing trial directory.")
    run_parser.add_argument("--fixed-stringency", type=float, default=0.46)
    run_parser.add_argument("--n-trials", type=int, default=20, help="Optuna stringency trials.")
    run_parser.add_argument("--shared-percentage-min", type=float, default=0.0)
    run_parser.add_argument("--shared-percentage-max", type=float, default=0.95)
    run_parser.add_argument("--max-class-features", type=int, default=None)
    run_parser.add_argument("--n-jobs", type=int, default=4)
    run_parser.add_argument("--prefer", choices=["processes", "threads"], default="threads")
    run_parser.add_argument("--metric", choices=["roc_auc", "feature_count"], default="roc_auc")
    run_parser.add_argument("--auc-only", action="store_true", help="Skip cell-type and pairwise distributions.")
    run_parser.set_defaults(func=run_one_trial)

    original_parser = subparsers.add_parser(
        "export-original",
        help="Export original/unshuffled artifact results in aggregation-compatible format.",
    )
    original_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    original_parser.add_argument("--output-dir", type=Path, required=True)
    original_parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    original_parser.add_argument("--seed", type=int, default=0)
    original_parser.add_argument("--fixed-stringency", type=float, default=0.46)
    original_parser.add_argument("--max-class-features", type=int, default=None)
    original_parser.add_argument("--n-jobs", type=int, default=4)
    original_parser.add_argument("--prefer", choices=["processes", "threads"], default="threads")
    original_parser.add_argument("--force", action="store_true")
    original_parser.add_argument("--auc-only", action="store_true", help="Skip cell-type and pairwise distributions.")
    original_parser.set_defaults(func=export_original_from_artifact)

    agg_parser = subparsers.add_parser("aggregate", help="Aggregate original and shuffled trials.")
    agg_parser.add_argument("--original-dir", type=Path, required=True)
    agg_parser.add_argument("--shuffle-root", type=Path, required=True)
    agg_parser.add_argument("--output-dir", type=Path, required=True)
    agg_parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    agg_parser.add_argument("--auc-only", action="store_true", help="Aggregate AUC and no-motif-fold summaries only.")
    agg_parser.set_defaults(func=aggregate_trials)

    status_parser = subparsers.add_parser("status", help="Summarize permutation trial completion status.")
    status_parser.add_argument("--shuffle-root", type=Path, required=True)
    status_parser.set_defaults(func=summarize_status)

    args = parser.parse_args()
    args.func(args)


def run_one_trial(args: argparse.Namespace) -> None:
    summary_path = args.output_dir / "trial_summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(
            f"{summary_path} already exists. Refusing to reuse seed/output directory without --force."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)

    discriminator = copy.copy(artifact["discriminator"])
    discriminator.cism = copy.copy(artifact["cism"])
    discriminator.patient_class_df = discriminator.patient_class_df.copy()
    labels = list(args.labels)
    rng = np.random.RandomState(args.seed)

    if args.shuffle_labels:
        discriminator.patient_class_df = shuffled_patient_classes(
            discriminator.patient_class_df,
            labels=labels,
            random_state=rng,
        )
    patient_class_csv = args.output_dir / "patient_class_permuted.csv"
    write_patient_class_csv(discriminator.patient_class_df, patient_class_csv)

    template_feature_conf = artifact.get("feature_conf", artifact.get("best_feature_conf"))
    if template_feature_conf is None:
        raise ValueError("Artifact must include either 'best_feature_conf' or 'feature_conf'.")

    feature_overrides = {"labels": labels}
    if args.max_class_features is not None:
        feature_overrides["max_class_features"] = args.max_class_features
    base_feature_conf = _clone_feature_conf(template_feature_conf, **feature_overrides)

    discovered = discriminator.discover(
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        classes=labels,
    )
    tuning_result, tuning_flags = tune_permutation_stringency(
        discriminator=discriminator,
        feature_conf=base_feature_conf,
        n_trials=args.n_trials,
        metric=args.metric,
        random_state=args.seed,
        model_random_state=args.seed if args.model_seed is None else args.model_seed,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
        shared_percentage_range=(args.shared_percentage_min, args.shared_percentage_max),
    )
    optimal_feature_conf = _clone_feature_conf(base_feature_conf, **tuning_result.best_params)
    fixed_feature_conf = _clone_feature_conf(base_feature_conf, shared_percentage=args.fixed_stringency)

    optimal = evaluate_feature_conf(
        discriminator=discriminator,
        feature_conf=optimal_feature_conf,
        random_state=args.seed if args.model_seed is None else args.model_seed,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
        labels=labels,
    )
    fixed = evaluate_feature_conf(
        discriminator=discriminator,
        feature_conf=fixed_feature_conf,
        random_state=args.seed if args.model_seed is None else args.model_seed,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
        labels=labels,
    )

    selected_motifs = selected_motif_table(optimal["motif_result"].get_motifs_by_class())
    selected_motifs.to_csv(args.output_dir / "selected_motifs_by_class.csv", index=False)
    optimal["feature_rows"].to_csv(args.output_dir / "optimal_motif_space_features.csv", index=False)
    fixed["feature_rows"].to_csv(args.output_dir / "fixed_motif_space_features.csv", index=False)
    optimal["results"].to_csv(args.output_dir / "optimal_loocv_results.csv")
    fixed["results"].to_csv(args.output_dir / "fixed_loocv_results.csv")
    tuning_result.trials_dataframe().to_csv(args.output_dir / "stringency_optuna_trials.csv", index=False)

    if not args.auc_only:
        write_distribution_outputs(
            output_dir=args.output_dir,
            artifact=artifact,
            discriminator=discriminator,
            labels=labels,
            motif_result=optimal["motif_result"],
        )

    summary = build_trial_summary(
        artifact_path=args.artifact,
        seed=args.seed,
        shuffle_labels=bool(args.shuffle_labels),
        labels=labels,
        fixed_stringency=args.fixed_stringency,
        optimal_feature_conf=optimal_feature_conf,
        optimal=optimal,
        fixed=fixed,
        discovered=discovered,
        best_params=tuning_result.best_params,
        best_score=tuning_result.best_score,
        patient_class_csv=patient_class_csv,
        optuna_zero_stringency_failed_seed=bool(tuning_flags["zero_stringency_failed_seed"]),
    )
    summary["model_seed"] = args.seed if args.model_seed is None else args.model_seed
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(args.output_dir / "trial_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


def write_distribution_outputs(
    output_dir: Path,
    artifact: dict[str, Any],
    discriminator,
    labels: list[str],
    motif_result,
) -> None:
    cell_distribution = motif_induced_cell_type_distribution(
        discriminator=discriminator,
        motifs_by_class=motif_result.get_motifs_by_class(),
        motifs_weight=motif_result.get_motifs_mean_weight(),
    )
    pairwise_distribution = motif_induced_pairwise_distribution(
        discriminator=discriminator,
        cells_type=discriminator.common_cells,
        cell_identity_to_motif_hash=motif_result.cell_identity_to_motif_hash,
        motifs_by_class=motif_result.get_motifs_by_class(),
        motifs_weight=motif_result.get_motifs_mean_weight(),
    )
    full_graph_df = load_full_graph_df(artifact, cells_type=discriminator.common_cells)
    general_cell_distribution = general_cell_type_distribution(
        full_graph_df=full_graph_df,
        patient_class_df=discriminator.patient_class_df,
        labels=labels,
        cells_type=discriminator.common_cells,
    )
    general_pairwise_dist = general_pairwise_distribution(
        full_graph_df=full_graph_df,
        patient_class_df=discriminator.patient_class_df,
        labels=labels,
        cells_type=discriminator.common_cells,
    )
    cell_distribution.to_csv(output_dir / "motif_induced_cell_type_distribution.csv", index=False)
    pairwise_distribution.to_csv(output_dir / "motif_induced_pairwise_distribution.csv", index=False)
    general_cell_distribution.to_csv(output_dir / "general_cell_type_distribution.csv", index=False)
    general_pairwise_dist.to_csv(output_dir / "general_pairwise_distribution.csv", index=False)


def build_trial_summary(
    artifact_path: Path,
    seed: int,
    shuffle_labels: bool,
    labels: list[str],
    fixed_stringency: float,
    optimal_feature_conf,
    optimal: dict[str, Any],
    fixed: dict[str, Any],
    discovered,
    best_params: dict[str, Any],
    best_score: float,
    patient_class_csv: Path,
    optuna_zero_stringency_failed_seed: bool,
) -> dict[str, Any]:
    return {
        "artifact": str(artifact_path),
        "seed": seed,
        "shuffle_labels": bool(shuffle_labels),
        "labels": labels,
        "fixed_stringency": fixed_stringency,
        "optimal_stringency": float(getattr(optimal_feature_conf, "shared_percentage")),
        "optimal_auc": float(optimal["auc"]),
        "fixed_auc": float(fixed["auc"]),
        "optimal_skipped_folds_no_motifs": int(optimal["skipped_folds_no_motifs"]),
        "optimal_no_feature_fold_patient_ids": ";".join(map(str, optimal["no_feature_fold_patient_ids"])),
        "fixed_skipped_folds_no_motifs": int(fixed["skipped_folds_no_motifs"]),
        "fixed_no_feature_fold_patient_ids": ";".join(map(str, fixed["no_feature_fold_patient_ids"])),
        "n_loocv_folds": int(optimal["total_folds"]),
        "optuna_zero_stringency_failed_seed": bool(optuna_zero_stringency_failed_seed),
        "n_discriminative_motif_rows": int(len(discovered.discriminative_motifs)),
        "n_selected_unique_motifs": int(len(optimal["motif_result"].all_motif_features)),
        "mean_selected_motifs_per_fold": float(optimal["feature_rows"]["features"].map(len).mean()),
        "fixed_mean_selected_motifs_per_fold": float(fixed["feature_rows"]["features"].map(len).mean()),
        "best_params": best_params,
        "best_score": float(best_score),
        "patient_class_csv": str(patient_class_csv),
    }


def feature_conf_params(feature_conf) -> dict[str, Any]:
    return {
        "shared_percentage": float(getattr(feature_conf, "shared_percentage")),
        "max_class_features": int(getattr(feature_conf, "max_class_features")),
    }


def artifact_tuning_trials(artifact: dict[str, Any], feature_conf) -> pd.DataFrame:
    trials_df = artifact.get("tuning_trials_df")
    if isinstance(trials_df, pd.DataFrame):
        return trials_df.copy()
    return pd.DataFrame(
        [
            {
                "params_shared_percentage": float(getattr(feature_conf, "shared_percentage")),
                "params_max_class_features": int(getattr(feature_conf, "max_class_features")),
                "value": artifact.get("tuning_best_score", np.nan),
                "state": "ARTIFACT_BEST",
            }
        ]
    )


def export_original_from_artifact(args: argparse.Namespace) -> None:
    summary_path = args.output_dir / "trial_summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"{summary_path} already exists. Use --force to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)

    discriminator = copy.copy(artifact["discriminator"])
    discriminator.cism = copy.copy(artifact["cism"])
    discriminator.patient_class_df = discriminator.patient_class_df.copy()
    labels = list(args.labels)

    template_feature_conf = artifact.get("feature_conf", artifact.get("best_feature_conf"))
    if template_feature_conf is None:
        raise ValueError("Artifact must include either 'best_feature_conf' or 'feature_conf'.")

    feature_overrides = {"labels": labels}
    if args.max_class_features is not None:
        feature_overrides["max_class_features"] = args.max_class_features
    optimal_feature_conf = _clone_feature_conf(template_feature_conf, **feature_overrides)
    fixed_feature_conf = _clone_feature_conf(optimal_feature_conf, shared_percentage=args.fixed_stringency)

    patient_class_csv = args.output_dir / "patient_class_original.csv"
    write_patient_class_csv(discriminator.patient_class_df, patient_class_csv)

    discovered = discriminator.discover(
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        classes=labels,
    )
    optimal = evaluate_feature_conf(
        discriminator=discriminator,
        feature_conf=optimal_feature_conf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
        labels=labels,
    )
    fixed = evaluate_feature_conf(
        discriminator=discriminator,
        feature_conf=fixed_feature_conf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
        labels=labels,
    )

    selected_motifs = selected_motif_table(optimal["motif_result"].get_motifs_by_class())
    selected_motifs.to_csv(args.output_dir / "selected_motifs_by_class.csv", index=False)
    optimal["feature_rows"].to_csv(args.output_dir / "optimal_motif_space_features.csv", index=False)
    fixed["feature_rows"].to_csv(args.output_dir / "fixed_motif_space_features.csv", index=False)
    optimal["results"].to_csv(args.output_dir / "optimal_loocv_results.csv")
    fixed["results"].to_csv(args.output_dir / "fixed_loocv_results.csv")
    artifact_tuning_trials(artifact, optimal_feature_conf).to_csv(
        args.output_dir / "stringency_optuna_trials.csv",
        index=False,
    )

    if not args.auc_only:
        write_distribution_outputs(
            output_dir=args.output_dir,
            artifact=artifact,
            discriminator=discriminator,
            labels=labels,
            motif_result=optimal["motif_result"],
        )

    summary = build_trial_summary(
        artifact_path=args.artifact,
        seed=args.seed,
        shuffle_labels=False,
        labels=labels,
        fixed_stringency=args.fixed_stringency,
        optimal_feature_conf=optimal_feature_conf,
        optimal=optimal,
        fixed=fixed,
        discovered=discovered,
        best_params=feature_conf_params(optimal_feature_conf),
        best_score=artifact.get("tuning_best_score", optimal["auc"]),
        patient_class_csv=patient_class_csv,
        optuna_zero_stringency_failed_seed=False,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(args.output_dir / "trial_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


def aggregate_trials(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_summary = read_summary(args.original_dir)
    shuffle_dirs = sorted(path for path in args.shuffle_root.iterdir() if path.is_dir())
    shuffle_summaries = [read_summary(path) for path in shuffle_dirs if (path / "trial_summary.json").exists()]
    if not shuffle_summaries:
        raise ValueError(f"No shuffle trial summaries found under {args.shuffle_root}.")

    summary_df = pd.concat(
        [pd.DataFrame([original_summary]), pd.DataFrame(shuffle_summaries)],
        ignore_index=True,
    )
    summary_df.to_csv(args.output_dir / "all_trial_summaries.csv", index=False)

    metric_tests = auc_tests(original_summary, shuffle_summaries)
    failure_summary = permutation_failure_summary(shuffle_summaries)
    if args.auc_only:
        plot_auc_null_histograms(original_summary, shuffle_summaries, args.output_dir)
    distribution_tests = []
    for family, name, file_name, category_cols in ([] if args.auc_only else [
        (
            "motif_induced",
            "cell_type",
            "motif_induced_cell_type_distribution.csv",
            ["patient_class", "cell_type"],
        ),
        (
            "motif_induced",
            "pairwise",
            "motif_induced_pairwise_distribution.csv",
            ["patient_class", "source_cell_type", "target_cell_type"],
        ),
        (
            "general",
            "cell_type",
            "general_cell_type_distribution.csv",
            ["patient_class", "cell_type"],
        ),
        (
            "general",
            "pairwise",
            "general_pairwise_distribution.csv",
            ["patient_class", "source_cell_type", "target_cell_type"],
        ),
    ]):
        original_dist = pd.read_csv(args.original_dir / file_name)
        shuffled = [
            pd.read_csv(path / file_name)
            for path in shuffle_dirs
            if (path / file_name).exists()
        ]
        distribution_tests.extend(
            distribution_stat_tests(
                family=family,
                name=name,
                original=original_dist,
                shuffled=shuffled,
                category_cols=category_cols,
            )
        )

    pd.DataFrame(metric_tests).to_csv(args.output_dir / "auc_statistical_tests.csv", index=False)
    pd.DataFrame([failure_summary]).to_csv(args.output_dir / "permutation_failure_summary.csv", index=False)
    if not args.auc_only:
        pd.DataFrame(distribution_tests).to_csv(args.output_dir / "distribution_statistical_tests.csv", index=False)
    (args.output_dir / "README.md").write_text(
        build_aggregate_readme(metric_tests, distribution_tests, auc_only=args.auc_only) + "\n"
    )
    print(
        json.dumps(
            {
                "auc_tests": metric_tests,
                "failure_summary": failure_summary,
                "distribution_tests": distribution_tests,
            },
            indent=2,
        )
    )


def summarize_status(args: argparse.Namespace) -> None:
    seed_dirs = sorted(
        [path for path in args.shuffle_root.glob("seed_*") if path.is_dir()],
        key=lambda path: int(path.name.replace("seed_", "")) if path.name.replace("seed_", "").isdigit() else path.name,
    )
    rows = []
    for seed_dir in seed_dirs:
        files = {path.name for path in seed_dir.iterdir() if path.is_file()}
        if "trial_summary.json" in files:
            status = "complete"
        elif required_intermediate_files().issubset(files):
            status = "partial_after_cism"
        elif "patient_class_permuted.csv" in files:
            status = "partial_early"
        else:
            status = "empty"
        rows.append({"seed": seed_dir.name.replace("seed_", ""), "status": status, "n_files": len(files)})

    status_df = pd.DataFrame(rows)
    if status_df.empty:
        print(json.dumps({"total_seed_dirs": 0, "counts": {}}, indent=2))
        return

    counts = status_df["status"].value_counts().sort_index().to_dict()
    status_df.to_csv(args.shuffle_root / "permutation_trial_status.csv", index=False)
    print(json.dumps({"total_seed_dirs": int(len(status_df)), "counts": counts}, indent=2))


def required_intermediate_files() -> set[str]:
    return {
        "fixed_loocv_results.csv",
        "fixed_motif_space_features.csv",
        "optimal_loocv_results.csv",
        "optimal_motif_space_features.csv",
        "patient_class_permuted.csv",
        "selected_motifs_by_class.csv",
        "stringency_optuna_trials.csv",
    }


def evaluate_feature_conf(
    discriminator,
    feature_conf,
    random_state: int,
    n_jobs: int,
    prefer: str,
    labels: list[str],
) -> dict[str, Any]:
    fold_results = evaluate_folds_with_no_feature_skips(
        discriminator=discriminator,
        feature_conf=feature_conf,
        random_state=random_state,
        n_jobs=n_jobs,
        prefer=prefer,
        labels=labels,
    )
    analyze_result = fold_results["analyze_result"]
    feature_rows = fold_results["feature_rows"]

    if fold_results["all_folds_no_motifs"]:
        motif_result = EmptyMotifResult(labels=labels)
    else:
        motif_result = GetMotifsResult(
            motif_space_features=feature_rows,
            discriminator=discriminator,
            classes=labels,
        )
    return {
        "analyze_result": analyze_result,
        "results": analyze_result.results,
        "auc": safe_analyze_auc(analyze_result),
        "feature_rows": feature_rows,
        "motif_result": motif_result,
        "valid_model": not fold_results["all_folds_no_motifs"],
        "failure_reason": (
            "All folds had no discriminative motifs at this stringency."
            if fold_results["all_folds_no_motifs"]
            else None
        ),
        "skipped_folds_no_motifs": fold_results["skipped_folds_no_motifs"],
        "no_feature_fold_patient_ids": fold_results["no_feature_fold_patient_ids"],
        "total_folds": fold_results["total_folds"],
        "all_folds_no_motifs": fold_results["all_folds_no_motifs"],
    }


def tune_permutation_stringency(
    discriminator,
    feature_conf,
    n_trials: int,
    metric: str,
    random_state: int,
    model_random_state: int,
    n_jobs: int,
    prefer: str,
    shared_percentage_range: tuple[float, float],
) -> tuple[OptunaTuningResult, dict[str, bool]]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for permutation stringency tuning. Install it with 'pip install optuna'."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    max_class_features = int(getattr(feature_conf, "max_class_features"))

    def objective(trial):
        tuned_feature_conf = _clone_feature_conf(
            feature_conf,
            shared_percentage=trial.suggest_float(
                "shared_percentage",
                float(shared_percentage_range[0]),
                float(shared_percentage_range[1]),
            ),
            max_class_features=trial.suggest_int(
                "max_class_features",
                max_class_features,
                max_class_features,
            ),
        )
        result = evaluate_feature_conf(
            discriminator=discriminator,
            feature_conf=tuned_feature_conf,
            random_state=model_random_state,
            n_jobs=n_jobs,
            prefer=prefer,
            labels=feature_conf.labels,
        )
        trial.set_user_attr("skipped_folds_no_motifs", result["skipped_folds_no_motifs"])
        trial.set_user_attr(
            "no_feature_fold_patient_ids",
            ";".join(map(str, result["no_feature_fold_patient_ids"])),
        )
        trial.set_user_attr("all_folds_no_motifs", result["all_folds_no_motifs"])
        if result["all_folds_no_motifs"]:
            return 0.0
        if metric == "roc_auc":
            return result["auc"]
        if metric == "feature_count":
            return float(result["feature_rows"]["features"].map(len).mean())
        raise ValueError("metric must be one of: roc_auc, feature_count.")

    study.optimize(objective, n_trials=n_trials)
    tuning_result = OptunaTuningResult(study=study, metric_name=metric)

    zero_result = evaluate_feature_conf(
        discriminator=discriminator,
        feature_conf=_clone_feature_conf(feature_conf, shared_percentage=0.0),
        random_state=model_random_state,
        n_jobs=n_jobs,
        prefer=prefer,
        labels=feature_conf.labels,
    )
    flags = {
        "zero_stringency_all_folds_no_motifs": bool(zero_result["all_folds_no_motifs"]),
        "zero_stringency_failed_seed": bool(zero_result["all_folds_no_motifs"]),
    }
    if tuning_result.best_score <= 0.0 and not zero_result["all_folds_no_motifs"]:
        study.enqueue_trial({"shared_percentage": 0.0, "max_class_features": max_class_features})
        study.optimize(objective, n_trials=1)
        tuning_result = OptunaTuningResult(study=study, metric_name=metric)
    return tuning_result, flags


def evaluate_folds_with_no_feature_skips(
    discriminator,
    feature_conf,
    random_state: int,
    n_jobs: int,
    prefer: str,
    labels: list[str],
) -> dict[str, Any]:
    discriminator._add_color_vec_hash()
    unique_patients_ids = discriminator.get_patients_class(feature_conf.labels).index.unique()
    local_patient_class = discriminator.patient_class_df[
        discriminator.patient_class_df.patient_class.isin(feature_conf.labels)
    ].copy()
    local_motifs_dataset = discriminator.cism.motifs_dataset

    raw_results = Parallel(n_jobs=n_jobs, verbose=0, prefer=prefer, pre_dispatch=1)(
        delayed(validate_fold_with_no_feature_skip)(
            discriminator=discriminator,
            motifs_dataset=local_motifs_dataset,
            unique_patients_ids=unique_patients_ids,
            trial_i=trial_i,
            feature_conf=feature_conf,
            random_state=random_state,
            patient_class_df=local_patient_class,
            labels=labels,
        )
        for trial_i in tqdm(range(len(unique_patients_ids)), leave=False)
    )

    analyze_rows = [row["analyze_row"] for row in raw_results]
    feature_rows = pd.DataFrame(
        {
            "test_patient_id": [row["test_patient_id"] for row in raw_results],
            "features": [row["features"] for row in raw_results],
        }
    )
    skipped_folds = sum(1 for row in raw_results if row["skipped_no_motifs"])
    no_feature_fold_patient_ids = [
        row["test_patient_id"] for row in raw_results if row["skipped_no_motifs"]
    ]
    analyze_result = AnalyzeMotifsResult(
        analyze_results=analyze_rows,
        patients_ids=unique_patients_ids,
        labels=feature_conf.labels,
    )
    return {
        "analyze_result": analyze_result,
        "feature_rows": feature_rows,
        "skipped_folds_no_motifs": skipped_folds,
        "no_feature_fold_patient_ids": no_feature_fold_patient_ids,
        "total_folds": len(unique_patients_ids),
        "all_folds_no_motifs": skipped_folds == len(unique_patients_ids),
    }


def validate_fold_with_no_feature_skip(
    discriminator,
    motifs_dataset: pd.DataFrame,
    unique_patients_ids,
    trial_i: int,
    feature_conf,
    random_state: int,
    patient_class_df: pd.DataFrame,
    labels: list[str],
) -> dict[str, Any]:
    test_patient_uid = unique_patients_ids[trial_i]
    train_motifs = get_motif_dataset_of_patient(motifs_dataset, unique_patients_ids, trial_i, train=True)
    test_motifs = get_motif_dataset_of_patient(motifs_dataset, unique_patients_ids, trial_i, train=False)
    try:
        analyze_row = discriminator._validate(
            x_data=train_motifs,
            x_test=test_motifs,
            test_patient_uid=test_patient_uid,
            feature_conf=feature_conf,
            random_state=np.random.RandomState(random_state),
            patient_class_df=patient_class_df,
        )
    except ValueError as exc:
        if "No discriminative features were selected" not in str(exc):
            raise
        return {
            "test_patient_id": test_patient_uid,
            "features": [],
            "skipped_no_motifs": True,
            "analyze_row": skipped_fold_analyze_row(
                test_patient_uid=test_patient_uid,
                patient_class_df=patient_class_df,
                labels=labels,
            ),
        }

    features = features_from_analyze_row(analyze_row)
    analyze_row = tuple(list(analyze_row[:9]) + [features] + [analyze_row[10]])
    return {
        "test_patient_id": test_patient_uid,
        "features": features,
        "skipped_no_motifs": False,
        "analyze_row": analyze_row,
    }


def get_motif_dataset_of_patient(
    motifs_dataset: pd.DataFrame,
    unique_patients_ids,
    trial_i: int,
    train: bool,
) -> pd.DataFrame:
    patient_mask = motifs_dataset.Patient_uId.isin(unique_patients_ids)
    if train:
        fold_mask = motifs_dataset.Patient_uId != unique_patients_ids[trial_i]
    else:
        fold_mask = motifs_dataset.Patient_uId == unique_patients_ids[trial_i]
    return motifs_dataset[patient_mask & fold_mask & (motifs_dataset.p_value < 0.05)].copy()


def skipped_fold_analyze_row(
    test_patient_uid: str,
    patient_class_df: pd.DataFrame,
    labels: list[str],
) -> tuple:
    true_class = patient_class_df.loc[test_patient_uid, DiscriminativeMotifs.PATIENT_CLASS]
    return (
        0,
        0,
        0,
        0,
        0,
        np.asarray([[0.5, 0.5]], dtype=float),
        true_class,
        "SKIPPED_NO_MOTIFS",
        np.asarray(labels),
        [],
        (None, None),
    )


def features_from_analyze_row(analyze_row: tuple) -> list[Any]:
    features = analyze_row[9]
    if isinstance(features, list):
        return [item[0] if isinstance(item, tuple) and len(item) == 1 else item for item in features]
    return [item[0] if isinstance(item, tuple) and len(item) == 1 else item for item in list(features)]


def safe_analyze_auc(analyze_result: AnalyzeMotifsResult) -> float:
    try:
        return float(analyze_result.get_roc_auc_score())
    except ValueError:
        return 0.5


class EmptyMotifResult:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.all_motif_features = []
        self.cell_identity_to_motif_hash = {}

    def get_motifs_by_class(self) -> dict[str, list[Any]]:
        return {label: [] for label in self.labels}

    def get_motifs_mean_weight(self) -> pd.Series:
        return pd.Series(dtype=float)


def shuffled_patient_classes(
    patient_class_df: pd.DataFrame,
    labels: list[str],
    random_state: np.random.RandomState,
) -> pd.DataFrame:
    shuffled = patient_class_df.copy()
    mask = shuffled[DiscriminativeMotifs.PATIENT_CLASS].isin(labels)
    values = shuffled.loc[mask, DiscriminativeMotifs.PATIENT_CLASS].to_numpy(copy=True)
    shuffled.loc[mask, DiscriminativeMotifs.PATIENT_CLASS] = random_state.permutation(values)
    return shuffled


def write_patient_class_csv(patient_class_df: pd.DataFrame, path: Path) -> None:
    export = patient_class_df[[DiscriminativeMotifs.PATIENT_CLASS]].copy()
    export.index.name = "patient_id"
    export.to_csv(path, header=False)


def selected_motif_table(motifs_by_class: dict[str, list[Any]]) -> pd.DataFrame:
    rows = []
    for label, motif_ids in motifs_by_class.items():
        rows.extend({"patient_class": label, "ID": motif_id} for motif_id in motif_ids)
    return pd.DataFrame(rows)


def load_full_graph_df(artifact: dict[str, Any], cells_type: dict[int, str]) -> pd.DataFrame:
    if "full_graph_df" in artifact:
        return artifact["full_graph_df"].copy()

    required = ["network_dataset_root_path", "dataset_folder"]
    missing = [key for key in required if key not in artifact or artifact[key] is None]
    if missing:
        raise ValueError(
            "Cannot compute general distributions because the artifact is missing "
            f"{missing}. Add a full_graph_df to the artifact or include the dataset path keys."
        )

    return GraphReader.get_graphs(
        full_graph_df=pd.DataFrame(),
        raw_data_folder=str(artifact["network_dataset_root_path"]),
        raw_data_folder_type=f"/{artifact['dataset_folder']}",
        disease=str(artifact.get("dataset_name", artifact["dataset_folder"])),
        cells_type=cells_type,
    )


def general_cell_type_distribution(
    full_graph_df: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    labels: list[str],
    cells_type: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for label in labels:
        patients = patient_class_df[
            patient_class_df[DiscriminativeMotifs.PATIENT_CLASS] == label
        ].index.astype(str)
        filtered = full_graph_df[full_graph_df["Patient_uId"].astype(str).isin(patients)]
        counts = {name: 0.0 for name in cells_type.values()}
        for graph in filtered["graph"]:
            for _, data in graph.nodes(data=True):
                counts[cells_type[int(data["type"])]] += 1.0
        total = sum(counts.values())
        for cell_type, count in counts.items():
            rows.append(
                {
                    "patient_class": label,
                    "cell_type": cell_type,
                    "count": count,
                    "proportion": count / total if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def general_pairwise_distribution(
    full_graph_df: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    labels: list[str],
    cells_type: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for label in labels:
        patients = patient_class_df[
            patient_class_df[DiscriminativeMotifs.PATIENT_CLASS] == label
        ].index.astype(str)
        filtered = full_graph_df[full_graph_df["Patient_uId"].astype(str).isin(patients)]
        matrix = np.zeros((len(cells_type), len(cells_type)), dtype=float)
        if not filtered.empty:
            matrix = filtered["pairwise_count"].sum()
        total = float(matrix.sum())
        for source_id, source_cell_type in cells_type.items():
            for target_id, target_cell_type in cells_type.items():
                count = float(matrix[int(source_id), int(target_id)])
                rows.append(
                    {
                        "patient_class": label,
                        "source_cell_type": source_cell_type,
                        "target_cell_type": target_cell_type,
                        "count": count,
                        "proportion": count / total if total > 0 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def motif_induced_cell_type_distribution(
    discriminator,
    motifs_by_class: dict[str, list[Any]],
    motifs_weight: pd.Series,
) -> pd.DataFrame:
    rows = []
    motif_table = discriminator.cism.motifs_dataset
    cell_names = discriminator.common_cells
    for label, motif_ids in motifs_by_class.items():
        counts = {name: 0.0 for name in cell_names.values()}
        for motif_id in motif_ids:
            graph = deserialize_graph(motif_table[motif_table.ID == motif_id].iloc[0].motif)
            weight = float(motifs_weight.get(motif_id, 1.0))
            for cell_type in nx.get_node_attributes(graph, name="type").values():
                counts[cell_names[int(cell_type)]] += weight
        total = sum(counts.values())
        for cell_type, count in counts.items():
            rows.append(
                {
                    "patient_class": label,
                    "cell_type": cell_type,
                    "count": count,
                    "proportion": count / total if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def motif_induced_pairwise_distribution(
    discriminator,
    cells_type: dict[int, str],
    cell_identity_to_motif_hash: dict[Any, list[int]],
    motifs_by_class: dict[str, list[Any]],
    motifs_weight: pd.Series,
) -> pd.DataFrame:
    rows = []
    motif_table = discriminator.cism.motifs_dataset
    for label, motif_ids in motifs_by_class.items():
        motif_set = set(motif_ids)
        matrix = pd.DataFrame(index=cells_type.values(), columns=cells_type.values(), dtype=float).fillna(0.0)
        for motif_ids_for_hash in cell_identity_to_motif_hash.values():
            for motif_id in motif_ids_for_hash:
                if motif_id not in motif_set:
                    continue
                graph = deserialize_graph(motif_table[motif_table.ID == motif_id].iloc[0].motif)
                weight = float(motifs_weight.get(motif_id, 1.0))
                for left, right in nx.Graph(graph).edges():
                    left_type = cells_type[int(graph.nodes[left]["type"])]
                    right_type = cells_type[int(graph.nodes[right]["type"])]
                    matrix.loc[left_type, right_type] += weight
        total = float(matrix.to_numpy().sum())
        if total > 0:
            matrix = matrix / total
        for source_cell_type in matrix.index:
            for target_cell_type in matrix.columns:
                rows.append(
                    {
                        "patient_class": label,
                        "source_cell_type": source_cell_type,
                        "target_cell_type": target_cell_type,
                        "proportion": float(matrix.loc[source_cell_type, target_cell_type]),
                        "count": float(matrix.loc[source_cell_type, target_cell_type] * total),
                    }
                )
    return pd.DataFrame(rows)


def deserialize_graph(motif_obj):
    if isinstance(motif_obj, (nx.Graph, nx.DiGraph)):
        return motif_obj
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    raise TypeError(f"Unsupported motif graph representation: {type(motif_obj)!r}")


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "trial_summary.json").read_text())


def auc_tests(original_summary: dict[str, Any], shuffle_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in ["optimal_auc", "fixed_auc"]:
        original = float(original_summary[metric])
        null_values = np.asarray([float(row[metric]) for row in shuffle_summaries], dtype=float)
        rows.append(
            {
                "metric": metric,
                "test": "empirical_permutation_greater_equal",
                "original": original,
                "null_mean": float(np.mean(null_values)),
                "null_std": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else np.nan,
                "n_null": int(len(null_values)),
                "p_value": empirical_p_value(original, null_values, alternative="greater"),
            }
        )
        if len(null_values) > 1:
            test = ttest_1samp(null_values, popmean=original)
            rows.append(
                {
                    "metric": metric,
                    "test": "one_sample_t_test_null_mean_equals_original",
                    "original": original,
                    "null_mean": float(np.mean(null_values)),
                    "null_std": float(np.std(null_values, ddof=1)),
                    "n_null": int(len(null_values)),
                    "p_value": float(test.pvalue),
                }
            )
    return rows


def plot_auc_null_histograms(
    original_summary: dict[str, Any], shuffle_summaries: list[dict[str, Any]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), dpi=220, sharey=True)
    for ax, metric, title in zip(
        axes,
        ("optimal_auc", "fixed_auc"),
        ("Retuned stringency", "Fixed original stringency"),
    ):
        values = np.asarray([float(row[metric]) for row in shuffle_summaries])
        observed = float(original_summary[metric])
        p_value = empirical_p_value(observed, values, alternative="greater")
        ax.hist(values, bins=np.linspace(0, 1, 21), color="#8aa3b3", edgecolor="white")
        ax.axvline(observed, color="#c85058", linewidth=2, label=f"Observed AUC = {observed:.3f}")
        ax.set(xlim=(0, 1), xlabel="LOOCV AUC", title=title)
        ax.text(0.98, 0.95, f"Permutation p = {p_value:.3f}", ha="right", va="top",
                transform=ax.transAxes, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Permutations")
    fig.tight_layout()
    fig.savefig(output_dir / "auc_null_histograms.png", facecolor="white")
    fig.savefig(output_dir / "auc_null_histograms.pdf", facecolor="white")
    plt.close(fig)


def permutation_failure_summary(shuffle_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(shuffle_summaries)
    optuna_zero_failed = [
        row["seed"] for row in shuffle_summaries if row.get("optuna_zero_stringency_failed_seed", False)
    ]
    fixed_all_empty = [
        row["seed"]
        for row in shuffle_summaries
        if skipped_count(row, "fixed") >= n_loocv_folds(row)
    ]
    optimal_all_empty = [
        row["seed"]
        for row in shuffle_summaries
        if skipped_count(row, "optimal") >= n_loocv_folds(row)
    ]
    optimal_partial_empty = [
        row["seed"]
        for row in shuffle_summaries
        if 0 < skipped_count(row, "optimal") < n_loocv_folds(row)
    ]
    fixed_partial_empty = [
        row["seed"]
        for row in shuffle_summaries
        if 0 < skipped_count(row, "fixed") < n_loocv_folds(row)
    ]
    return {
        "n_shuffle_trials": total,
        "n_optuna_zero_stringency_failed_seeds": len(optuna_zero_failed),
        "optuna_zero_stringency_failed_seeds": ";".join(map(str, optuna_zero_failed)),
        "n_fixed_all_folds_no_motifs": len(fixed_all_empty),
        "fixed_all_folds_no_motifs_seeds": ";".join(map(str, fixed_all_empty)),
        "n_fixed_partial_no_feature_fold_seeds": len(fixed_partial_empty),
        "fixed_partial_no_feature_fold_seeds": ";".join(map(str, fixed_partial_empty)),
        "n_optimal_partial_no_feature_fold_seeds": len(optimal_partial_empty),
        "optimal_partial_no_feature_fold_seeds": ";".join(map(str, optimal_partial_empty)),
        "n_optimal_all_folds_no_motifs": len(optimal_all_empty),
        "optimal_all_folds_no_motifs_seeds": ";".join(map(str, optimal_all_empty)),
    }


def skipped_count(summary: dict[str, Any], setting: str) -> int:
    return int(summary.get(f"{setting}_skipped_folds_no_motifs", 0))


def n_loocv_folds(summary: dict[str, Any]) -> int:
    return int(
        summary.get(
            "n_loocv_folds",
            summary.get("optimal_total_folds", summary.get("fixed_total_folds", 0)),
        )
    )


def distribution_stat_tests(
    family: str,
    name: str,
    original: pd.DataFrame,
    shuffled: list[pd.DataFrame],
    category_cols: list[str],
) -> list[dict[str, Any]]:
    rows = []
    if not shuffled:
        return rows
    all_dist = align_distributions([original] + shuffled, category_cols)
    original_vec = all_dist[0]
    null_matrix = np.vstack(all_dist[1:])
    expected = null_matrix.mean(axis=0)
    if expected.sum() > 0 and original_vec.sum() > 0:
        scaled_expected = expected + 1e-12
        scaled_expected = scaled_expected * (original_vec.sum() / scaled_expected.sum())
        chi = chisquare(original_vec, f_exp=scaled_expected)
        g = power_divergence(original_vec, f_exp=scaled_expected, lambda_="log-likelihood")
        rows.extend(
            [
                {
                    "distribution_family": family,
                    "distribution": name,
                    "test": "chi_square_goodness_of_fit_vs_mean_shuffle",
                    "statistic": float(chi.statistic),
                    "p_value": float(chi.pvalue),
                    "n_shuffle": len(shuffled),
                },
                {
                    "distribution_family": family,
                    "distribution": name,
                    "test": "g_test_goodness_of_fit_vs_mean_shuffle",
                    "statistic": float(g.statistic),
                    "p_value": float(g.pvalue),
                    "n_shuffle": len(shuffled),
                },
            ]
        )

    original_prop = normalize_vector(original_vec)
    null_props = np.vstack([normalize_vector(row) for row in null_matrix])
    null_centroid = normalize_vector(null_props.mean(axis=0))
    observed_js = float(jensenshannon(original_prop, null_centroid, base=2.0))
    null_js = []
    for idx in range(len(null_props)):
        others = np.delete(null_props, idx, axis=0)
        centroid = normalize_vector(others.mean(axis=0)) if len(others) else null_centroid
        null_js.append(float(jensenshannon(null_props[idx], centroid, base=2.0)))
    rows.append(
        {
            "distribution_family": family,
            "distribution": name,
            "test": "jensen_shannon_empirical_distance",
            "statistic": observed_js,
            "p_value": empirical_p_value(observed_js, np.asarray(null_js), alternative="greater"),
            "n_shuffle": len(shuffled),
        }
    )
    return rows


def align_distributions(distributions: list[pd.DataFrame], category_cols: list[str]) -> list[np.ndarray]:
    keys = sorted(
        set().union(
            *[
                set(map(tuple, dist[category_cols].astype(str).to_numpy()))
                for dist in distributions
            ]
        )
    )
    vectors = []
    for dist in distributions:
        values = (
            dist.assign(_key=list(map(tuple, dist[category_cols].astype(str).to_numpy())))
            .groupby("_key", observed=True)["count"]
            .sum()
        )
        vectors.append(np.asarray([float(values.get(key, 0.0)) for key in keys], dtype=float))
    return vectors


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    total = float(vector.sum())
    if total <= 0:
        return np.full(len(vector), 1.0 / len(vector)) if len(vector) else vector
    return vector / total


def empirical_p_value(observed: float, null_values: np.ndarray, alternative: str) -> float:
    if len(null_values) == 0:
        return np.nan
    if alternative == "greater":
        extreme = np.sum(null_values >= observed)
    elif alternative == "less":
        extreme = np.sum(null_values <= observed)
    else:
        raise ValueError("alternative must be 'greater' or 'less'.")
    return float((extreme + 1) / (len(null_values) + 1))


def build_aggregate_readme(
    metric_tests: list[dict[str, Any]], distribution_tests: list[dict[str, Any]], auc_only: bool = False
) -> str:
    lines = [
        "# CISM label permutation aggregation",
        "",
        "Primary AUC significance is the empirical permutation p-value. A one-sample t-test",
        "is included only as a descriptive normal-approximation check because the original",
        "label run is a single observed value.",
        "",
    ]
    if not auc_only:
        lines.extend([
            "Distribution tests include chi-square and G-test goodness-of-fit against the mean",
            "shuffle distribution, plus an empirical Jensen-Shannon distance test. They are run",
            "separately for motif-induced and general cell-type/pairwise distributions.",
            "",
        ])
    else:
        lines.extend([
            "No-motif LOOCV folds receive neutral 0.5/0.5 class probabilities and remain in AUC.",
            "The trial summaries report skipped folds and all-empty seeds. Only NN/NP labels",
            "were shuffled, preserving their class counts; PN/PP labels were unchanged.",
            "The fixed comparator uses the original best stringency; the optimal comparator",
            "retunes stringency within each permutation.",
            "",
            "See `auc_null_histograms.png` for the observed and permutation AUCs.",
            "",
        ])
    lines.extend(["## Key p-values", ""])
    for row in metric_tests + distribution_tests:
        label = row.get("metric", f"{row.get('distribution_family')} {row.get('distribution')}")
        lines.append(f"- {row['test']} ({label}): p={row['p_value']:.4g}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
