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
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.cism import DiscriminativeFeatureKey
from cism.optimization import _clone_feature_conf


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np_onetailed"
)
DEFAULT_FANMOD_OUTPUT_ROOT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "fanmod_output"
    / "Melanoma"
)


def normal_two_tailed_p_value(z_scores: pd.Series) -> pd.Series:
    """Return two-tailed normal p-values from FANMOD Z-scores."""
    try:
        from scipy.special import erfc

        return pd.Series(erfc(np.abs(z_scores.astype(float)) / math.sqrt(2.0)), index=z_scores.index)
    except ImportError:
        return z_scores.astype(float).abs().transform(lambda z: math.erfc(z / math.sqrt(2.0)))


def normal_absence_one_tailed_p_value(z_scores: pd.Series) -> pd.Series:
    """Return lower-tail normal p-values from FANMOD Z-scores."""
    try:
        from scipy.special import erfc

        return pd.Series(0.5 * erfc(-z_scores.astype(float) / math.sqrt(2.0)), index=z_scores.index)
    except ImportError:
        return z_scores.astype(float).transform(lambda z: 0.5 * math.erfc(-z / math.sqrt(2.0)))


def prepare_absence_motif_dataset(
    motifs_dataset: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    labels: list[str],
    alpha: float,
    significance_mode: str = "absence_one_tailed",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform FANMOD-enriched motif rows into an absence/depletion signal table.

    CISM expects positive feature values in ``Freq`` and internally reconstructs
    total subgraphs as ``Count / Freq``. For underrepresented motifs we therefore
    preserve the original FANMOD frequency columns and replace ``Freq`` with the
    positive depletion magnitude ``Mean_Freq - observed Freq``.
    """
    required_columns = {"Patient_uId", "FOV", "ID", "Freq", "Count", "Mean_Freq", "STD", "z_score"}
    missing_columns = required_columns.difference(motifs_dataset.columns)
    if missing_columns:
        raise ValueError(f"motifs_dataset is missing required columns: {sorted(missing_columns)}")

    relevant_patients = patient_class_df[patient_class_df["patient_class"].isin(labels)].index.astype(str)
    data = motifs_dataset[motifs_dataset["Patient_uId"].astype(str).isin(relevant_patients)].copy()
    data["Patient_uId"] = data["Patient_uId"].astype(str)
    data["ID"] = pd.to_numeric(data["ID"], errors="coerce")
    data = data.dropna(subset=["ID"])
    data["ID"] = data["ID"].astype("int64")
    for column in ("Freq", "Count", "Mean_Freq", "STD", "z_score"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    original_rows = len(data)
    data["fanmod_empirical_p_value"] = data["p_value"]
    data["observed_Freq"] = data["Freq"].astype(float)
    data["observed_Count"] = data["Count"].astype(float)
    data["expected_Freq"] = data["Mean_Freq"].astype(float)
    data["p_value_normal_two_tailed"] = normal_two_tailed_p_value(data["z_score"])
    data["p_value_absence_one_tailed"] = normal_absence_one_tailed_p_value(data["z_score"])
    data["p_value_direction"] = np.where(data["z_score"].astype(float) < 0, "absence", "proliferation")
    data["absence_depletion"] = data["expected_Freq"] - data["observed_Freq"]

    if significance_mode == "absence_one_tailed":
        active_p_value_col = "p_value_absence_one_tailed"
        underrepresentation_mode = "normal_one_tailed_absence"
    elif significance_mode == "normal_two_tailed":
        active_p_value_col = "p_value_normal_two_tailed"
        underrepresentation_mode = "normal_two_tailed_absence"
    else:
        raise ValueError(
            "significance_mode must be either 'absence_one_tailed' or 'normal_two_tailed'."
        )

    finite_mask = np.isfinite(data[active_p_value_col])
    absence_mask = (
        finite_mask
        & (data[active_p_value_col] < alpha)
        & (data["p_value_direction"] == "absence")
        & (data["absence_depletion"] > 0)
        & (data["observed_Freq"] > 0)
    )
    absence = data.loc[absence_mask].copy()

    absence["_total_subgraphs"] = absence["observed_Count"] / absence["observed_Freq"]
    valid_total = np.isfinite(absence["_total_subgraphs"]) & (absence["_total_subgraphs"] > 0)
    absence = absence.loc[valid_total].copy()
    absence["Freq"] = absence["absence_depletion"].astype(float)
    absence["Count"] = absence["Freq"] * absence["_total_subgraphs"]
    absence["p_value"] = absence[active_p_value_col].astype(float)
    absence["underrepresentation_mode"] = underrepresentation_mode

    summary = {
        "labels": labels,
        "alpha": alpha,
        "significance_mode": significance_mode,
        "active_p_value": active_p_value_col,
        "nn_np_patients": int(len(relevant_patients)),
        "rows_after_label_filter": int(original_rows),
        "absence_rows": int(len(absence)),
        "unique_absence_motifs": int(absence["ID"].nunique()),
        "unique_absence_patients": int(absence["Patient_uId"].nunique()),
    }
    return absence, summary


def load_absence_dataset_from_fanmod_outputs(
    fanmod_output_root: Path,
    patient_class_df: pd.DataFrame,
    labels: list[str],
    alpha: float,
    motif_size: int,
    dataset_name: str,
    significance_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import networkx as nx

    from cism.helpers import parse_csv

    label_patients = patient_class_df[patient_class_df["patient_class"].isin(labels)].index.astype(str)
    patient_numbers = {patient.replace(dataset_name, "", 1) for patient in label_patients}
    csv_paths = sorted((fanmod_output_root / str(motif_size)).glob("*.csv"))
    selected_paths = [
        path
        for path in csv_paths
        if "_" in path.stem and path.stem.split("_", 1)[0] in patient_numbers
    ]
    if not selected_paths:
        raise ValueError(
            f"No FANMOD CSV files for labels {labels} under {fanmod_output_root / str(motif_size)}"
        )

    frames: list[pd.DataFrame] = []
    raw_rows = 0
    for index, path in enumerate(selected_paths, start=1):
        patient, fov = path.stem.split("_", 1)
        print(f"[{index}/{len(selected_paths)}] parsing {path.name}", flush=True)
        parsed = parse_csv(str(path), patient_num=int(patient), fov=fov)
        if parsed.empty:
            continue
        if not parsed["motif"].map(lambda graph: graph.number_of_nodes() == motif_size).all():
            raise ValueError(f"FANMOD file contains motifs other than {motif_size} nodes: {path}")
        raw_rows += int(len(parsed))
        parsed["Patient_uId"] = f"{dataset_name}{patient}"
        parsed["nunique_colors"] = parsed["motif"].apply(
            lambda graph: len(set(nx.get_node_attributes(graph, name="type").values()))
        )
        absence, _ = prepare_absence_motif_dataset(
            motifs_dataset=parsed,
            patient_class_df=patient_class_df,
            labels=labels,
            alpha=alpha,
            significance_mode=significance_mode,
        )
        if not absence.empty:
            frames.append(absence)

    if frames:
        absence_dataset = pd.concat(frames, ignore_index=True)
    else:
        absence_dataset = pd.DataFrame()

    summary = {
        "labels": labels,
        "alpha": alpha,
        "significance_mode": significance_mode,
        "active_p_value": (
            "p_value_absence_one_tailed"
            if significance_mode == "absence_one_tailed"
            else "p_value_normal_two_tailed"
        ),
        "nn_np_patients": int(len(label_patients)),
        "fanmod_files_scanned": int(len(selected_paths)),
        "rows_after_label_filter": int(raw_rows),
        "absence_rows": int(len(absence_dataset)),
        "unique_absence_motifs": int(absence_dataset["ID"].nunique()) if not absence_dataset.empty else 0,
        "unique_absence_patients": int(absence_dataset["Patient_uId"].nunique()) if not absence_dataset.empty else 0,
        "source": "fanmod_output_csv",
    }
    return absence_dataset, summary


def write_summary(
    output_dir: Path,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    selected: pd.DataFrame,
) -> None:
    lines = [
        "# Absent Discriminative CISM: Melanoma NN/NP",
        "",
        "This run keeps significantly underrepresented motif rows and uses depletion",
        "(`Mean_Freq - observed Freq`) as the CISM feature magnitude.",
        f"Active significance mode: `{summary.get('significance_mode', 'unknown')}`.",
        "",
        "## Cohort",
        "",
        f"- Labels: {', '.join(summary['labels'])}",
        f"- Patients: {summary['nn_np_patients']}",
        f"- FANMOD rows after label filter: {summary['rows_after_label_filter']:,}",
        f"- Significant absence rows: {summary['absence_rows']:,}",
        f"- Unique absent motifs: {summary['unique_absence_motifs']:,}",
        f"- Patients with at least one absent motif row: {summary['unique_absence_patients']:,}",
        "",
        "## LOOCV Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Reviewer Response Anchor",
            "",
            "Underrepresented motifs were explicitly explored by re-running CISM on",
            "normal two-tailed significant motif rows whose observed frequency was",
            "below FANMOD's random-network expectation. The resulting CISM object",
            "therefore represents absence/depletion patterns rather than enriched",
            "motif proliferation.",
            "",
            f"Selected discriminative motif rows: {len(selected):,}",
        ]
    )
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run melanoma NN/NP discriminative CISM using underrepresented FANMOD motifs."
    )
    parser.add_argument("--serialized-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--fanmod-output-root", type=Path, default=DEFAULT_FANMOD_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--significance-mode",
        choices=["absence_one_tailed", "normal_two_tailed"],
        default="absence_one_tailed",
        help="Significance test for underrepresented motifs. "
        "absence_one_tailed uses lower-tail normal p-values and matches enriched one-tailed alpha.",
    )
    parser.add_argument("--motif-size", type=int, default=4)
    parser.add_argument(
        "--fanmod-source-only",
        action="store_true",
        help="Read raw FANMOD CSVs rather than reusing motif rows from the serialized artifact. "
        "Required when changing motif size.",
    )
    parser.add_argument("--dataset-name", default="Melanoma")
    parser.add_argument(
        "--shared-percentage",
        type=float,
        default=0.0,
        help="Minimum within-class patient fraction for absent motif discovery. "
        "Default keeps motifs seen in at least one training patient.",
    )
    parser.add_argument(
        "--tune-stringency",
        action="store_true",
        help="Tune shared_percentage with Optuna before the final absent CISM run.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for absence-specific shared_percentage tuning.",
    )
    parser.add_argument(
        "--shared-percentage-min",
        type=float,
        default=0.0,
        help="Lower bound for Optuna shared_percentage tuning.",
    )
    parser.add_argument(
        "--shared-percentage-max",
        type=float,
        default=0.95,
        help="Upper bound for Optuna shared_percentage tuning.",
    )
    parser.add_argument(
        "--max-class-features",
        type=int,
        default=None,
        help="Override the saved discriminative feature cap per class.",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--prefer", choices=["processes", "threads"], default="threads")
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.serialized_artifact.open("rb") as handle:
        artifact = pickle.load(handle)

    labels = list(args.labels)
    if args.motif_size != artifact["cism"].motif_size and not args.fanmod_source_only:
        raise ValueError("Motif size differs from the artifact; use --fanmod-source-only.")
    if args.fanmod_source_only:
        absence_dataset = pd.DataFrame()
        summary = {}
    else:
        absence_dataset, summary = prepare_absence_motif_dataset(
            motifs_dataset=artifact["cism"].motifs_dataset,
            patient_class_df=artifact["discriminator"].patient_class_df,
            labels=labels,
            alpha=args.alpha,
            significance_mode=args.significance_mode,
        )
    if absence_dataset.empty:
        absence_dataset, summary = load_absence_dataset_from_fanmod_outputs(
            fanmod_output_root=args.fanmod_output_root,
            patient_class_df=artifact["discriminator"].patient_class_df,
            labels=labels,
            alpha=args.alpha,
            motif_size=args.motif_size,
            dataset_name=args.dataset_name,
            significance_mode=args.significance_mode,
        )
    if absence_dataset.empty:
        raise ValueError("No significant underrepresented motif rows were found for the requested labels.")
    summary["motif_size"] = args.motif_size

    absent_cism = copy.copy(artifact["cism"])
    absent_cism.motif_size = args.motif_size
    absent_cism.motifs_dataset = absence_dataset
    absent_discriminator = copy.copy(artifact["discriminator"])
    absent_discriminator.cism = absent_cism

    feature_overrides = {"labels": labels, "shared_percentage": args.shared_percentage}
    if args.max_class_features is not None:
        feature_overrides["max_class_features"] = args.max_class_features
    feature_conf = _clone_feature_conf(artifact["best_feature_conf"], **feature_overrides)
    summary["feature_conf"] = {
        "type": type(feature_conf).__name__,
        "shared_percentage": getattr(feature_conf, "shared_percentage", None),
        "max_class_features": getattr(feature_conf, "max_class_features", None),
        "extract_by": getattr(getattr(feature_conf, "extract_by", None), "value", None),
        "use_motifs": getattr(feature_conf, "use_motifs", None),
        "use_cells_type_composition": getattr(feature_conf, "use_cells_type_composition", None),
    }
    tuning_result = None
    if args.tune_stringency:
        tuning_result = absent_discriminator.tune_stringency(
            feature_conf=feature_conf,
            n_trials=args.n_trials,
            metric="roc_auc",
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            prefer=args.prefer,
            exclude_patients=[],
            shared_percentage_range=(args.shared_percentage_min, args.shared_percentage_max),
            max_class_features_range=(30, 30),
        )
        feature_conf = _clone_feature_conf(
            feature_conf,
            shared_percentage=tuning_result.best_params["shared_percentage"],
            max_class_features=30,
        )
        summary["tuning"] = {
            "n_trials": args.n_trials,
            "metric": tuning_result.metric_name,
            "best_params": tuning_result.best_params,
            "best_score": tuning_result.best_score,
            "shared_percentage_range": [args.shared_percentage_min, args.shared_percentage_max],
            "max_class_features_fixed": 30,
        }
        summary["feature_conf"].update(
            {
                "shared_percentage": getattr(feature_conf, "shared_percentage", None),
                "max_class_features": getattr(feature_conf, "max_class_features", None),
            }
        )

    discovered = absent_discriminator.discover(
        extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
        classes=labels,
    )
    analyze_result = absent_discriminator.analyze_motifs(
        feature_conf=feature_conf,
        exclude_patients=[],
        n_jobs=args.n_jobs,
        prefer=args.prefer,
        random_state=np.random.RandomState(args.random_seed),
    )
    metrics = analyze_result.get_metrics()
    metrics["roc_auc"] = analyze_result.get_roc_auc_score()

    summary["post_discover_rows"] = int(len(absent_cism.motifs_dataset))
    summary["post_discover_unique_motifs"] = int(absent_cism.motifs_dataset["ID"].nunique())

    selected = discovered.discriminative_motifs.copy()
    selected_export = selected.drop(columns=["motif", "colors_vec"], errors="ignore")
    selected_export.to_csv(args.output_dir / "absent_discriminative_motifs.csv", index=False)
    analyze_result.results.to_csv(args.output_dir / "absent_discriminative_loocv_results.csv", index=False)
    if tuning_result is not None:
        tuning_result.trials_dataframe().to_csv(
            args.output_dir / "absent_stringency_optuna_trials.csv",
            index=False,
        )

    absence_summary_columns = [
        "Patient",
        "FOV",
        "Patient_uId",
        "patient_class",
        "ID",
        "observed_Freq",
        "expected_Freq",
        "absence_depletion",
        "Freq",
        "observed_Count",
        "Count",
        "Mean_Freq",
        "STD",
        "z_score",
        "fanmod_empirical_p_value",
        "p_value_normal_two_tailed",
        "p_value_absence_one_tailed",
        "p_value",
        "p_value_direction",
        "nunique_colors",
    ]
    existing_summary_columns = [col for col in absence_summary_columns if col in absent_cism.motifs_dataset.columns]
    absent_cism.motifs_dataset[existing_summary_columns].to_csv(
        args.output_dir / "absent_motif_rows_summary.csv",
        index=False,
    )

    output_artifact = {
        **artifact,
        "cism": absent_cism,
        "discriminator": absent_discriminator,
        "feature_conf": feature_conf,
        "absent_analyze_result": analyze_result,
        "absent_discover_result": discovered,
        "absent_stringency_tuning_result": tuning_result,
        "absence_summary": summary,
        "absence_metrics": metrics,
    }
    with (args.output_dir / "Melanoma_NN_NP_absent_cism_ready.pkl").open("wb") as handle:
        pickle.dump(output_artifact, handle)

    (args.output_dir / "absence_run_summary.json").write_text(
        json.dumps({"summary": summary, "metrics": metrics}, indent=2, default=str) + "\n"
    )
    write_summary(args.output_dir, summary, metrics, selected_export)

    print(json.dumps({"summary": summary, "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
