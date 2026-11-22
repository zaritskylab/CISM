#!/usr/bin/env python3
"""Compile the report for bootstrapped cropped-FOV CISM artifacts."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import gc
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy import stats
except ImportError:  # pragma: no cover - cluster env should have scipy, but keep report usable.
    stats = None


REPO_ROOT = _REPO_ROOT
CROP_ROOT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "crop_fov_scale_experiment"
)
REPORT_ROOT = CROP_ROOT / "melanoma_cropped_fov_report"
REFERENCE_CSV = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "split_fov_scale_experiment"
    / "fold_union_stringency_046_stability"
    / "native_fold_union_selected_motifs.csv"
)
ORIGINAL_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
CROP_FRACTIONS = [0.9, 0.8, 0.7, 0.6, 0.5]


def token(fraction: float) -> str:
    return f"{int(round(fraction * 100)):03d}"


def artifact_path(fraction: float) -> Path:
    return CROP_ROOT / f"crop_{token(fraction)}_cism_ready.pkl"


def prep_summary_path(fraction: float) -> Path:
    return CROP_ROOT / f"crop_{token(fraction)}_preparation_summary.json"


def metadata_path(fraction: float) -> Path:
    return CROP_ROOT / f"crop_{token(fraction)}_metadata.csv"


def load_selected_sources() -> pd.DataFrame:
    for fraction in CROP_FRACTIONS:
        path = metadata_path(fraction)
        if path.exists():
            metadata = pd.read_csv(path, usecols=["patient_uid", "source_fov"])
            return (
                metadata.rename(columns={"patient_uid": "Patient_uId"})
                .drop_duplicates(["Patient_uId", "source_fov"])
                .reset_index(drop=True)
            )
    raise FileNotFoundError("No crop metadata CSV was found; cannot define the source-FOV cohort.")


def crop_status_row(fraction: float) -> dict:
    summary_path = prep_summary_path(fraction)
    prep = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    artifact = artifact_path(fraction)
    return {
        "crop_area_fraction": fraction,
        "side_fraction": prep.get("side_fraction", np.nan),
        "fraction_mode": prep.get("fraction_mode", ""),
        "prepared_crops": prep.get("n_crop_graphs_written", np.nan),
        "skipped_crops": prep.get("n_crops_skipped", np.nan),
        "cism_artifact": "complete" if artifact.exists() else "pending",
    }


def source_fov_from_crop(fov: str) -> str:
    return str(fov).split("_crop", 1)[0]


def bootstrap_index_from_crop(fov: str) -> int:
    return int(str(fov).rsplit("_b", 1)[1])


def significant_enriched_rows(motifs: pd.DataFrame) -> pd.DataFrame:
    if "p_value_direction" in motifs.columns:
        return motifs[motifs["p_value_direction"].eq("proliferation")]
    return motifs


def summarize_original(
    reference_ids: set[int],
    selected_sources: pd.DataFrame,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    with ORIGINAL_ARTIFACT.open("rb") as handle:
        artifact = pickle.load(handle)
    motifs = artifact["cism"].motifs_dataset
    enriched = significant_enriched_rows(motifs)[["Patient_uId", "FOV", "ID"]].drop_duplicates()
    enriched = enriched.rename(columns={"FOV": "source_fov"}).merge(
        selected_sources,
        on=["Patient_uId", "source_fov"],
        how="inner",
    )

    per_fov_counts = (
        enriched.groupby(["Patient_uId", "source_fov"], observed=True)["ID"]
        .nunique()
        .reset_index(name="enriched_motif_count")
    )
    per_source = per_fov_counts.rename(columns={"enriched_motif_count": "original_enriched_motif_count"})
    count_row = {
        "crop_area_fraction": 1.0,
        "status": "native_uncut",
        "n_crop_graphs": int(per_fov_counts.shape[0]),
        "mean_enriched_motifs_per_crop": float(per_fov_counts["enriched_motif_count"].mean()),
        "sd_enriched_motifs_per_crop": float(per_fov_counts["enriched_motif_count"].std()),
        "median_enriched_motifs_per_crop": float(per_fov_counts["enriched_motif_count"].median()),
        "min_enriched_motifs_per_crop": int(per_fov_counts["enriched_motif_count"].min()),
        "max_enriched_motifs_per_crop": int(per_fov_counts["enriched_motif_count"].max()),
        "unique_enriched_motifs_all_crops": int(enriched["ID"].nunique()),
    }

    all_ids = set(map(int, enriched["ID"].unique()))
    shared = all_ids & reference_ids
    retention_rows = [
        {
            "crop_area_fraction": 1.0,
            "scenario": "native_uncut_fovs",
            "bootstrap_index": "",
            "unique_enriched_motifs": len(all_ids),
            "native_reference_motifs": len(reference_ids),
            "shared_native_reference_motifs": len(shared),
            "reference_recovered_fraction": len(shared) / len(reference_ids),
            "jaccard_vs_reference": len(shared) / len(all_ids | reference_ids),
        }
    ]
    del artifact, motifs, enriched, per_fov_counts
    gc.collect()
    return [count_row], retention_rows, per_source


def summarize_artifact(
    fraction: float,
    reference_ids: set[int],
) -> tuple[list[dict], list[dict], pd.DataFrame, pd.DataFrame]:
    with artifact_path(fraction).open("rb") as handle:
        artifact = pickle.load(handle)
    motifs = artifact["cism"].motifs_dataset
    enriched = significant_enriched_rows(motifs)[["Patient_uId", "FOV", "ID"]].drop_duplicates()
    enriched["bootstrap_index"] = enriched["FOV"].map(bootstrap_index_from_crop)
    enriched["source_fov"] = enriched["FOV"].map(source_fov_from_crop)

    per_crop_counts = (
        enriched.groupby(["Patient_uId", "FOV", "source_fov", "bootstrap_index"], observed=True)["ID"]
        .nunique()
        .reset_index(name="enriched_motif_count")
    )
    per_source = (
        per_crop_counts.groupby(["Patient_uId", "source_fov"], observed=True)["enriched_motif_count"]
        .agg(
            crop_mean_enriched_motif_count="mean",
            crop_sd_enriched_motif_count="std",
            crop_min_enriched_motif_count="min",
            crop_max_enriched_motif_count="max",
            n_bootstraps="count",
        )
        .reset_index()
    )
    per_source["crop_area_fraction"] = fraction

    distribution_rows = []
    for value in per_crop_counts["enriched_motif_count"]:
        distribution_rows.append(
            {
                "crop_area_fraction": fraction,
                "distribution_unit": "individual_crop",
                "value": float(value),
            }
        )
    bootstrap_means = (
        per_crop_counts.groupby("bootstrap_index", observed=True)["enriched_motif_count"]
        .mean()
        .reset_index(name="mean_enriched_motif_count")
    )
    for value in bootstrap_means["mean_enriched_motif_count"]:
        distribution_rows.append(
            {
                "crop_area_fraction": fraction,
                "distribution_unit": "bootstrap_mean",
                "value": float(value),
            }
        )
    count_row = {
        "crop_area_fraction": fraction,
        "n_crop_graphs": int(per_crop_counts.shape[0]),
        "mean_enriched_motifs_per_crop": float(per_crop_counts["enriched_motif_count"].mean()),
        "sd_enriched_motifs_per_crop": float(per_crop_counts["enriched_motif_count"].std()),
        "median_enriched_motifs_per_crop": float(per_crop_counts["enriched_motif_count"].median()),
        "min_enriched_motifs_per_crop": int(per_crop_counts["enriched_motif_count"].min()),
        "max_enriched_motifs_per_crop": int(per_crop_counts["enriched_motif_count"].max()),
        "unique_enriched_motifs_all_crops": int(enriched["ID"].nunique()),
    }

    retention_rows = []
    all_ids = set(map(int, enriched["ID"].unique()))
    shared = all_ids & reference_ids
    retention_rows.append(
        {
            "crop_area_fraction": fraction,
            "scenario": "all_bootstrap_crops",
            "bootstrap_index": "",
            "unique_enriched_motifs": len(all_ids),
            "native_reference_motifs": len(reference_ids),
            "shared_native_reference_motifs": len(shared),
            "reference_recovered_fraction": len(shared) / len(reference_ids),
            "jaccard_vs_reference": len(shared) / len(all_ids | reference_ids),
        }
    )
    for bootstrap_index, group in enriched.groupby("bootstrap_index", observed=True):
        ids = set(map(int, group["ID"].unique()))
        shared = ids & reference_ids
        retention_rows.append(
            {
                "crop_area_fraction": fraction,
                "scenario": "one_crop_per_source_fov",
                "bootstrap_index": int(bootstrap_index),
                "unique_enriched_motifs": len(ids),
                "native_reference_motifs": len(reference_ids),
                "shared_native_reference_motifs": len(shared),
                "reference_recovered_fraction": len(shared) / len(reference_ids),
                "jaccard_vs_reference": len(shared) / len(ids | reference_ids),
            }
        )

    del artifact, motifs, enriched, per_crop_counts
    gc.collect()
    return [count_row], retention_rows, per_source, pd.DataFrame(distribution_rows)


def paired_tests(original_per_source: pd.DataFrame, crop_per_source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if original_per_source.empty or crop_per_source.empty:
        return pd.DataFrame()
    for fraction in [0.9, 0.5]:
        group = crop_per_source[crop_per_source["crop_area_fraction"].eq(fraction)]
        if group.empty:
            continue
        if fraction not in {0.9, 0.5}:
            continue
        paired = original_per_source.merge(group, on=["Patient_uId", "source_fov"], how="inner")
        original = paired["original_enriched_motif_count"].astype(float).to_numpy()
        cropped = paired["crop_mean_enriched_motif_count"].astype(float).to_numpy()
        delta = cropped - original
        fraction_delta = delta / original
        row = {
            "comparison": f"100% vs {int(round(fraction * 100))}%",
            "n_source_fovs": int(len(paired)),
            "mean_100pct": float(np.mean(original)),
            "mean_crop": float(np.mean(cropped)),
            "mean_delta_crop_minus_100pct": float(np.mean(delta)),
            "mean_fraction_delta": float(np.mean(fraction_delta)),
            "median_fraction_delta": float(np.median(fraction_delta)),
            "cohens_dz": float(np.mean(delta) / np.std(delta, ddof=1)) if len(delta) > 1 and np.std(delta, ddof=1) else np.nan,
        }
        if stats is not None and len(delta) > 1:
            row["paired_t_p_value"] = float(stats.ttest_rel(cropped, original).pvalue)
            try:
                row["wilcoxon_p_value"] = float(stats.wilcoxon(cropped, original, zero_method="wilcox").pvalue)
            except ValueError:
                row["wilcoxon_p_value"] = np.nan
        else:
            row["paired_t_p_value"] = np.nan
            row["wilcoxon_p_value"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def native_vs_crop_distribution_tests(native_mean: float, crop_distribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if crop_distribution.empty:
        return pd.DataFrame()
    for (fraction, unit), group in crop_distribution.groupby(["crop_area_fraction", "distribution_unit"], observed=True):
        values = group["value"].astype(float).to_numpy()
        sample_mean = float(np.mean(values))
        sample_sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        row = {
            "comparison": f"100% vs {int(round(fraction * 100))}%",
            "distribution_unit": unit,
            "n": int(len(values)),
            "native_100pct_mean": float(native_mean),
            "crop_mean": sample_mean,
            "crop_sd": sample_sd,
            "native_minus_crop_mean": float(native_mean - sample_mean),
            "native_z_vs_crop_distribution": float((native_mean - sample_mean) / sample_sd)
            if sample_sd and not np.isnan(sample_sd)
            else np.nan,
            "native_within_mean_plusminus_1sd": bool(sample_mean - sample_sd <= native_mean <= sample_mean + sample_sd)
            if sample_sd and not np.isnan(sample_sd)
            else False,
            "native_within_mean_plusminus_2sd": bool(sample_mean - 2 * sample_sd <= native_mean <= sample_mean + 2 * sample_sd)
            if sample_sd and not np.isnan(sample_sd)
            else False,
            "native_percentile_in_crop_distribution": float(np.mean(values <= native_mean)),
        }
        if stats is not None and len(values) > 1:
            row["one_sample_t_p_value"] = float(stats.ttest_1samp(values, popmean=native_mean).pvalue)
        else:
            row["one_sample_t_p_value"] = np.nan
        rows.append(row)
    order = pd.CategoricalDtype(
        [f"100% vs {int(round(fraction * 100))}%" for fraction in CROP_FRACTIONS],
        ordered=True,
    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["comparison"] = out["comparison"].astype(order)
        out = out.sort_values(["comparison", "distribution_unit"]).reset_index(drop=True)
        out["comparison"] = out["comparison"].astype(str)
    return out


def retention_summary(retention: pd.DataFrame) -> pd.DataFrame:
    complete = retention[retention["scenario"].ne("pending")].copy()
    if complete.empty:
        return pd.DataFrame()
    return (
        complete.groupby(["crop_area_fraction", "scenario"], observed=True)
        .agg(
            repeats=("bootstrap_index", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            mean_unique_enriched_motifs=("unique_enriched_motifs", "mean"),
            mean_shared_native_reference_motifs=("shared_native_reference_motifs", "mean"),
            mean_reference_recovered_fraction=("reference_recovered_fraction", "mean"),
            sd_reference_recovered_fraction=("reference_recovered_fraction", "std"),
            mean_jaccard_vs_reference=("jaccard_vs_reference", "mean"),
            sd_jaccard_vs_reference=("jaccard_vs_reference", "std"),
        )
        .reset_index()
    )


def fmt_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            if "p_value" in col:
                out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3e}")
            else:
                out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    return out


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = fmt_df(df).fillna("")
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def save_plots(counts: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    paths = []
    complete_counts = counts[counts["status"].isin(["complete", "native_uncut"])].sort_values("crop_area_fraction")
    if not complete_counts.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(
            complete_counts["crop_area_fraction"],
            complete_counts["mean_enriched_motifs_per_crop"],
            yerr=complete_counts["sd_enriched_motifs_per_crop"],
            marker="o",
            capsize=4,
        )
        ax.set_xlabel("Crop area fraction")
        ax.set_ylabel("Enriched motifs per crop")
        ax.set_title("Motifs Arising From Cropped Tissue")
        ax.invert_xaxis()
        fig.tight_layout()
        path = REPORT_ROOT / "melanoma_cropped_fov_motif_counts.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    complete_summary = summary[summary["scenario"].eq("one_crop_per_source_fov")]
    if not complete_summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(
            complete_summary["crop_area_fraction"],
            complete_summary["mean_reference_recovered_fraction"],
            yerr=complete_summary["sd_reference_recovered_fraction"].fillna(0),
            marker="o",
            capsize=4,
        )
        ax.set_xlabel("Crop area fraction")
        ax.set_ylabel("Fraction of native 19 motifs recovered")
        ax.set_ylim(0, 1.02)
        ax.set_title("Native Motif Retention Across Bootstrapped Crops")
        ax.invert_xaxis()
        fig.tight_layout()
        path = REPORT_ROOT / "melanoma_cropped_fov_retention.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def add_pdf_table(pdf: PdfPages, title: str, df: pd.DataFrame, fontsize: int = 7) -> None:
    view = fmt_df(df).fillna("")
    fig_height = max(2.2, min(10.5, 1.0 + 0.34 * (len(view) + 1)))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    table = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.25)
    for (row, _col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    status: pd.DataFrame,
    counts: pd.DataFrame,
    tests: pd.DataFrame,
    distribution_tests: pd.DataFrame,
    retention: pd.DataFrame,
    summary: pd.DataFrame,
    plots: list[Path],
) -> Path:
    report = REPORT_ROOT / "melanoma_cropped_fov_report.md"
    plot_map = {path.name: path for path in plots}
    lines = [
        "# Melanoma Cropped FOV Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report covers the completed area-crop CISM artifacts and includes the native 100% uncut-FOV baseline.",
        "",
        "Reference set: the native 19 LOOCV fold-union discriminative motifs selected at stringency 0.46.",
        "",
        "Retention definition in this report: native-reference motifs recovered among enriched cropped-FOV motifs. This is intentionally lighter than rerunning the full fold-union discriminative selection for every bootstrap.",
        "",
        "## Artifact Status",
        "",
        md_table(status),
        "",
        "## Enriched Motifs Per Crop",
        "",
        md_table(counts),
        "",
        "## Paired Statistical Tests",
        "",
        "Tests compare each source FOV's native 100% enriched-motif count against the mean count across its 10 bootstrapped crops. The Wilcoxon signed-rank test is the primary paired non-parametric test; the paired t-test is included as a parametric companion.",
        "",
        md_table(tests),
        "",
        "## Native 100% Within Crop Distributions",
        "",
        "For each crop fraction, this checks where the native 100% mean motif count falls relative to the crop-count distribution. The `bootstrap_mean` rows use 10 bootstrap-level means and are less inflated by repeated crops from the same source FOV than the `individual_crop` rows.",
        "",
        md_table(distribution_tests),
        "",
    ]
    if "melanoma_cropped_fov_motif_counts.png" in plot_map:
        lines += ["![Motif count summary](melanoma_cropped_fov_motif_counts.png)", ""]
    lines += [
        "## Native Motif Retention Summary",
        "",
        md_table(summary),
        "",
    ]
    if "melanoma_cropped_fov_retention.png" in plot_map:
        lines += ["![Retention summary](melanoma_cropped_fov_retention.png)", ""]
    lines += [
        "## Native Motif Retention Per Bootstrap",
        "",
        md_table(retention),
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def write_pdf(
    status: pd.DataFrame,
    counts: pd.DataFrame,
    tests: pd.DataFrame,
    distribution_tests: pd.DataFrame,
    retention: pd.DataFrame,
    summary: pd.DataFrame,
    plots: list[Path],
) -> Path:
    pdf_path = REPORT_ROOT / "melanoma_cropped_fov_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.02, 0.92, "Melanoma Cropped FOV Report", fontsize=20, fontweight="bold")
        ax.text(0.02, 0.84, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
        ax.text(
            0.02,
            0.74,
            "Completed area crops are summarized alongside the native 100% uncut-FOV baseline.",
            fontsize=11,
            wrap=True,
        )
        ax.text(
            0.02,
            0.66,
            "Reference: native 19 LOOCV fold-union discriminative motifs at stringency 0.46.",
            fontsize=11,
            wrap=True,
        )
        ax.text(
            0.02,
            0.58,
            "Retention here means recovery of those native motifs among enriched cropped-FOV motifs.",
            fontsize=11,
            wrap=True,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_pdf_table(pdf, "Artifact Status", status)
        add_pdf_table(pdf, "Enriched Motifs Per Crop", counts, fontsize=6)
        add_pdf_table(pdf, "Paired Statistical Tests", tests, fontsize=6)
        add_pdf_table(pdf, "Native 100% Within Crop Distributions", distribution_tests, fontsize=5)
        add_pdf_table(pdf, "Native Motif Retention Summary", summary, fontsize=6)
        for path in plots:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            img = plt.imread(path)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        add_pdf_table(pdf, "Native Motif Retention Per Bootstrap", retention, fontsize=5)
    return pdf_path


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE_CSV)
    reference_ids = set(map(int, reference["ID"]))
    selected_sources = load_selected_sources()

    status_rows = [crop_status_row(fraction) for fraction in CROP_FRACTIONS]
    count_rows = []
    retention_rows = []
    crop_per_source_rows = []
    crop_distribution_rows = []
    print("Summarizing native 100% uncut FOVs", flush=True)
    original_counts, original_retention, original_per_source = summarize_original(reference_ids, selected_sources)
    native_mean = float(original_counts[0]["mean_enriched_motifs_per_crop"])
    count_rows.extend(original_counts)
    retention_rows.extend(original_retention)
    for row in status_rows:
        fraction = float(row["crop_area_fraction"])
        if row["cism_artifact"] != "complete":
            count_rows.append(
                {
                    "crop_area_fraction": fraction,
                    "status": "pending",
                    "n_crop_graphs": row["prepared_crops"],
                    "mean_enriched_motifs_per_crop": np.nan,
                    "sd_enriched_motifs_per_crop": np.nan,
                    "median_enriched_motifs_per_crop": np.nan,
                    "min_enriched_motifs_per_crop": np.nan,
                    "max_enriched_motifs_per_crop": np.nan,
                    "unique_enriched_motifs_all_crops": np.nan,
                }
            )
            retention_rows.append(
                {
                    "crop_area_fraction": fraction,
                    "scenario": "pending",
                    "bootstrap_index": "",
                    "unique_enriched_motifs": np.nan,
                    "native_reference_motifs": len(reference_ids),
                    "shared_native_reference_motifs": np.nan,
                    "reference_recovered_fraction": np.nan,
                    "jaccard_vs_reference": np.nan,
                }
            )
            continue
        print(f"Summarizing crop area fraction {fraction:.2f}", flush=True)
        counts, retention, per_source, distribution = summarize_artifact(fraction, reference_ids)
        counts[0]["status"] = "complete"
        count_rows.extend(counts)
        retention_rows.extend(retention)
        crop_per_source_rows.append(per_source)
        crop_distribution_rows.append(distribution)

    status = pd.DataFrame(status_rows)
    counts = pd.DataFrame(count_rows)
    retention = pd.DataFrame(retention_rows)
    summary = retention_summary(retention)
    crop_per_source = pd.concat(crop_per_source_rows, ignore_index=True) if crop_per_source_rows else pd.DataFrame()
    crop_distribution = (
        pd.concat(crop_distribution_rows, ignore_index=True) if crop_distribution_rows else pd.DataFrame()
    )
    tests = paired_tests(original_per_source, crop_per_source)
    distribution_tests = native_vs_crop_distribution_tests(native_mean, crop_distribution)

    status.to_csv(REPORT_ROOT / "melanoma_cropped_fov_status.csv", index=False)
    counts.to_csv(REPORT_ROOT / "melanoma_cropped_fov_enriched_motif_counts.csv", index=False)
    original_per_source.to_csv(REPORT_ROOT / "melanoma_cropped_fov_native_100pct_per_source_fov.csv", index=False)
    crop_per_source.to_csv(REPORT_ROOT / "melanoma_cropped_fov_crop_mean_per_source_fov.csv", index=False)
    tests.to_csv(REPORT_ROOT / "melanoma_cropped_fov_paired_tests.csv", index=False)
    crop_distribution.to_csv(REPORT_ROOT / "melanoma_cropped_fov_crop_count_distributions.csv", index=False)
    distribution_tests.to_csv(REPORT_ROOT / "melanoma_cropped_fov_native_vs_crop_distribution_tests.csv", index=False)
    retention.to_csv(REPORT_ROOT / "melanoma_cropped_fov_native_motif_retention.csv", index=False)
    summary.to_csv(REPORT_ROOT / "melanoma_cropped_fov_native_motif_retention_summary.csv", index=False)

    plots = save_plots(counts, summary)
    md_path = write_markdown(status, counts, tests, distribution_tests, retention, summary, plots)
    pdf_path = write_pdf(status, counts, tests, distribution_tests, retention, summary, plots)
    print(json.dumps({"markdown": str(md_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
