#!/usr/bin/env python3
"""Create a compact cropped-FOV scale summary figure."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import gc
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = ANALYSIS_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.spatial_scale.crop_fov_scale_experiment import artifact_path, motif_set  # noqa: E402
from analysis.cohort_imbalance.imbalanced_bootstrap_stringency import load_patient_classes  # noqa: E402


CROP_ROOT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "crop_fov_scale_experiment"
)
REPORT_ROOT = CROP_ROOT / "melanoma_cropped_fov_report"
ORIGINAL_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
NATIVE_REFERENCE_CSV = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "split_fov_scale_experiment"
    / "fold_union_stringency_046_stability"
    / "native_fold_union_selected_motifs.csv"
)
CACHE_CSV = REPORT_ROOT / "melanoma_cropped_fov_compact_summary_data.csv"
OUTPUT_PNG = REPORT_ROOT / "melanoma_cropped_fov_compact_summary.png"
OUTPUT_PDF = REPORT_ROOT / "melanoma_cropped_fov_compact_summary.pdf"
CROP_FRACTIONS = [0.9, 0.8, 0.7, 0.6, 0.5]
LABELS = ["NN", "NP"]
STRINGENCY = 0.46
MAX_FEATURES = 30
P_VALUE_THRESHOLD = 0.05


def load_artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def optimized_fold_union_selected_motifs(
    motifs: pd.DataFrame,
    patient_classes: pd.DataFrame,
    cohort_patients: list[str],
) -> set[int]:
    local = motifs[motifs["Patient_uId"].isin(cohort_patients) & (motifs["p_value"] < P_VALUE_THRESHOLD)].copy()
    local["patient_class"] = local["Patient_uId"].map(patient_classes["patient_class"])

    patient_presence = local[["ID", "patient_class", "Patient_uId"]].drop_duplicates()
    class_patient_counts = (
        patient_presence.groupby(["ID", "patient_class"], observed=True)["Patient_uId"]
        .nunique()
        .unstack(fill_value=0)
    )
    class_patient_counts = class_patient_counts.reindex(columns=LABELS, fill_value=0)

    class_stats = (
        local.groupby(["ID", "patient_class"], observed=True)
        .agg(freq_sum=("Freq", "sum"), row_count=("Freq", "size"), total_count=("Count", "sum"))
        .reset_index()
    )
    stats_by_label = {}
    for label in LABELS:
        label_stats = class_stats[class_stats["patient_class"].eq(label)].set_index("ID")
        stats_by_label[label] = label_stats[["freq_sum", "row_count", "total_count"]]

    heldout_presence = patient_presence.set_index(["Patient_uId", "ID", "patient_class"]).sort_index()
    heldout_stats = (
        local.groupby(["Patient_uId", "ID", "patient_class"], observed=True)
        .agg(freq_sum=("Freq", "sum"), row_count=("Freq", "size"), total_count=("Count", "sum"))
        .sort_index()
    )
    class_sizes = patient_classes.loc[cohort_patients, "patient_class"].value_counts().to_dict()
    selected_union: set[int] = set()

    for held_out in cohort_patients:
        held_label = patient_classes.loc[held_out, "patient_class"]
        train_class_sizes = class_sizes.copy()
        train_class_sizes[held_label] -= 1

        counts = class_patient_counts.copy()
        try:
            present_for_heldout = heldout_presence.loc[held_out].reset_index()
        except KeyError:
            present_for_heldout = pd.DataFrame(columns=["ID", "patient_class"])
        if not present_for_heldout.empty:
            subtract = (
                present_for_heldout.groupby(["ID", "patient_class"], observed=True)
                .size()
                .reset_index(name="present")
            )
            for row in subtract.itertuples(index=False):
                if row.patient_class in counts.columns and row.ID in counts.index:
                    counts.at[row.ID, row.patient_class] -= 1

        try:
            held_stats = heldout_stats.loc[held_out].reset_index()
        except KeyError:
            held_stats = pd.DataFrame(columns=["ID", "patient_class", "freq_sum", "row_count", "total_count"])

        for label in LABELS:
            other_labels = [other for other in LABELS if other != label]
            label_counts = counts[label]
            other_count_sum = counts[other_labels].sum(axis=1)
            eligible_ids = label_counts.index[(label_counts > 0) & (other_count_sum == 0)]
            if len(eligible_ids) == 0:
                continue

            summary = pd.DataFrame({"ID": eligible_ids})
            summary["patient_count"] = summary["ID"].map(label_counts).astype(float)
            summary["patient_percentage"] = summary["patient_count"] / train_class_sizes[label]
            summary = summary[summary["patient_percentage"] >= STRINGENCY]
            if summary.empty:
                continue

            label_stats = stats_by_label[label].reindex(summary["ID"]).fillna(0).copy()
            if not held_stats.empty:
                subtract = held_stats[held_stats["patient_class"].eq(label)].set_index("ID")
                common = label_stats.index.intersection(subtract.index)
                if len(common):
                    label_stats.loc[common, ["freq_sum", "row_count", "total_count"]] -= subtract.loc[
                        common, ["freq_sum", "row_count", "total_count"]
                    ]
            summary = summary.join(label_stats, on="ID")
            summary["mean_freq"] = summary["freq_sum"] / summary["row_count"].replace(0, pd.NA)
            summary = summary.sort_values(
                ["patient_percentage", "patient_count", "mean_freq", "total_count"],
                ascending=False,
            )
            selected_union.update(map(int, summary["ID"].head(MAX_FEATURES)))

    return selected_union


def compute_discriminative_rows() -> pd.DataFrame:
    native_reference = pd.read_csv(NATIVE_REFERENCE_CSV)
    native_ids = set(map(int, native_reference["ID"]))
    original = load_artifact(ORIGINAL_ARTIFACT)
    patient_classes = load_patient_classes(original["discriminator"], LABELS)
    cohort_patients = patient_classes.index.tolist()

    rows = [
        {
            "tissue_retention": 1.0,
            "discriminative_motifs": len(native_ids),
            "preserved_original_discriminative_motifs": len(native_ids),
            "reference_motifs": len(native_ids),
        }
    ]
    del original
    gc.collect()

    for fraction in CROP_FRACTIONS:
        print(f"Computing fold-union discriminative motifs for {fraction:.0%}", flush=True)
        artifact = load_artifact(artifact_path(CROP_ROOT, fraction))
        motifs = artifact["cism"].motifs_dataset.copy()
        selected = optimized_fold_union_selected_motifs(motifs, patient_classes, cohort_patients)
        rows.append(
            {
                "tissue_retention": fraction,
                "discriminative_motifs": len(selected),
                "preserved_original_discriminative_motifs": len(selected & native_ids),
                "reference_motifs": len(native_ids),
            }
        )
        del artifact, motifs
        gc.collect()
    return pd.DataFrame(rows)


def build_summary_data(force: bool = False) -> pd.DataFrame:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(REPORT_ROOT / "melanoma_cropped_fov_enriched_motif_counts.csv")
    counts = counts[counts["status"].isin(["native_uncut", "complete"])].copy()
    counts = counts.rename(
        columns={
            "crop_area_fraction": "tissue_retention",
            "mean_enriched_motifs_per_crop": "mean_enriched_subgraphs",
            "sd_enriched_motifs_per_crop": "sd_enriched_subgraphs",
        }
    )
    counts = counts[["tissue_retention", "mean_enriched_subgraphs", "sd_enriched_subgraphs"]]

    if CACHE_CSV.exists() and not force:
        disc = pd.read_csv(CACHE_CSV)
    else:
        disc = compute_discriminative_rows()
        disc.to_csv(CACHE_CSV, index=False)
    return counts.merge(disc, on="tissue_retention", how="inner").sort_values("tissue_retention")


def plot_summary(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    x = data["tissue_retention"] * 100

    axes[0].errorbar(
        x,
        data["mean_enriched_subgraphs"],
        yerr=data["sd_enriched_subgraphs"],
        marker="o",
        linewidth=2,
        capsize=3,
        color="#2f6f8f",
    )
    axes[0].set_ylabel("Enriched subgraphs")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(x, data["discriminative_motifs"], marker="o", linewidth=2, color="#8f4f2f")
    axes[1].set_ylabel("Discriminative motifs")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(
        x,
        data["preserved_original_discriminative_motifs"],
        marker="o",
        linewidth=2,
        color="#4f7f45",
    )
    axes[2].axhline(data["reference_motifs"].iloc[0], color="#4f7f45", linestyle=":", linewidth=1)
    axes[2].set_ylabel("Original motifs preserved")
    axes[2].set_ylim(bottom=0)

    for ax in axes:
        ax.invert_xaxis()
        ax.set_xlabel("Tissue retained (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Cropped FOV Scale Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=240)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)


def main() -> None:
    force = "--force" in sys.argv
    data = build_summary_data(force=force)
    plot_summary(data)
    print(data.to_string(index=False))
    print(OUTPUT_PNG.resolve())
    print(OUTPUT_PDF.resolve())


if __name__ == "__main__":
    main()
