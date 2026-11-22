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
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "Melanoma_NN_NP_absent_cism_ready.pkl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
    / "class_absent_motifs_90pct"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report motifs depleted in at least 90% of patients in each class."
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs=2, default=["NN", "NP"])
    parser.add_argument("--prevalence-threshold", type=float, default=0.9)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)

    motifs = artifact["cism"].motifs_dataset.copy()
    patient_class_df = artifact["discriminator"].patient_class_df.copy()
    labels = list(args.labels)
    patient_class_df = patient_class_df[patient_class_df["patient_class"].isin(labels)]

    motifs = motifs[motifs["Patient_uId"].isin(patient_class_df.index)].copy()
    if "p_value_direction" in motifs.columns:
        motifs = motifs[motifs["p_value_direction"] == "absence"].copy()

    patient_motif = build_patient_motif_table(motifs, patient_class_df)
    report = build_class_report(patient_motif, patient_class_df, labels, args.prevalence_threshold)

    report.to_csv(args.output_dir / "class_absent_motifs_90pct.csv", index=False)
    for label in labels:
        report[report["patient_class"] == label].to_csv(
            args.output_dir / f"class_absent_motifs_90pct_{label}.csv",
            index=False,
        )

    summary = {
        "artifact": str(args.artifact),
        "labels": labels,
        "prevalence_threshold": args.prevalence_threshold,
        "significance_mode": artifact.get("absence_summary", {}).get("significance_mode", "normal_two_tailed"),
        "active_p_value": artifact.get("absence_summary", {}).get("active_p_value", "p_value_normal_two_tailed"),
        "motif_rows": int(len(motifs)),
        "unique_absence_motifs": int(motifs["ID"].nunique()),
        "n_patients_by_class": patient_class_df["patient_class"].value_counts().to_dict(),
        "n_report_rows": int(len(report)),
        "n_motifs_by_class": report.groupby("patient_class", observed=True)["ID"].nunique().to_dict(),
    }
    (args.output_dir / "class_absent_motifs_90pct_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_readme(args.output_dir, summary)
    print(json.dumps(summary, indent=2))


def build_patient_motif_table(motifs: pd.DataFrame, patient_class_df: pd.DataFrame) -> pd.DataFrame:
    required = {"Patient_uId", "ID", "observed_Count", "observed_Freq", "absence_depletion"}
    missing = required.difference(motifs.columns)
    if missing:
        raise ValueError(f"motifs is missing required columns: {sorted(missing)}")

    grouped = (
        motifs.groupby(["Patient_uId", "ID"], observed=True)
        .agg(
            observed_appearances=("observed_Count", "sum"),
            mean_observed_freq=("observed_Freq", "mean"),
            mean_absence_depletion=("absence_depletion", "mean"),
            min_p_value=("p_value", "min"),
            n_fovs=("FOV", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped.merge(
        patient_class_df[["patient_class"]],
        left_on="Patient_uId",
        right_index=True,
        how="left",
    )
    return grouped


def build_class_report(
    patient_motif: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    labels: list[str],
    prevalence_threshold: float,
) -> pd.DataFrame:
    class_sizes = patient_class_df["patient_class"].value_counts().to_dict()
    prevalence = (
        patient_motif.groupby(["patient_class", "ID"], observed=True)
        .agg(
            depleted_patient_count=("Patient_uId", "nunique"),
            observed_appearances_mean=("observed_appearances", "mean"),
            observed_appearances_std=("observed_appearances", "std"),
            observed_frequency_mean=("mean_observed_freq", "mean"),
            observed_frequency_std=("mean_observed_freq", "std"),
            absence_depletion_mean=("mean_absence_depletion", "mean"),
            absence_depletion_std=("mean_absence_depletion", "std"),
            min_p_value=("min_p_value", "min"),
            mean_n_fovs=("n_fovs", "mean"),
        )
        .reset_index()
    )
    prevalence["class_patient_count"] = prevalence["patient_class"].map(class_sizes).astype(int)
    prevalence["absence_rate_within_class"] = (
        prevalence["depleted_patient_count"] / prevalence["class_patient_count"]
    )

    rows = []
    for label in labels:
        other_label = next(other for other in labels if other != label)
        class_rows = prevalence[
            (prevalence["patient_class"] == label)
            & (prevalence["absence_rate_within_class"] >= prevalence_threshold)
        ].copy()
        other_rates = prevalence[prevalence["patient_class"] == other_label].set_index("ID")
        class_rows["other_class"] = other_label
        class_rows["other_class_absence_rate"] = class_rows["ID"].map(
            other_rates["absence_rate_within_class"]
        ).fillna(0.0)
        class_rows["other_class_depleted_patient_count"] = class_rows["ID"].map(
            other_rates["depleted_patient_count"]
        ).fillna(0).astype(int)
        class_rows["absent_in_other_class_90pct"] = (
            class_rows["other_class_absence_rate"] >= prevalence_threshold
        )
        rows.append(class_rows)

    if not rows:
        return pd.DataFrame()

    report = pd.concat(rows, ignore_index=True)
    report["observed_appearances_std"] = report["observed_appearances_std"].fillna(0.0)
    report = report.sort_values(
        ["patient_class", "absence_rate_within_class", "observed_appearances_mean", "ID"],
        ascending=[True, False, False, True],
    )
    return report


def write_readme(output_dir: Path, summary: dict) -> None:
    lines = [
        "# Class-Level 90% Underrepresented Motifs",
        "",
        "This report uses the two-tailed absent motif CISM object and asks, for each class independently,",
        "which motifs are significantly underrepresented in at least 90% of that class's patients.",
        "",
        f"- Significance mode: {summary['significance_mode']}",
        f"- Active p-value: {summary['active_p_value']}",
        f"- Motif rows: {summary['motif_rows']:,}",
        f"- Unique absence motifs: {summary['unique_absence_motifs']:,}",
        f"- Report rows: {summary['n_report_rows']:,}",
        f"- Motifs by class: {summary['n_motifs_by_class']}",
        "",
        "`observed_appearances_mean` and `observed_appearances_std` summarize observed motif counts",
        "among patients in the class where the motif was significantly depleted.",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
