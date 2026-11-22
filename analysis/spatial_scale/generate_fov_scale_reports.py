#!/usr/bin/env python3
"""Generate Markdown and PDF reports for melanoma FOV-scale motif analyses."""

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
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUNTIME = REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
SPATIAL_DIR = RUNTIME / "fov_spatial_scale_nn_np"
NATIVE_DIR = RUNTIME / "fov_scale_motif_counts_nn_np"
SPLIT_DIR = RUNTIME / "split_fov_scale_experiment"
REPORT_DIR = SPLIT_DIR / "reports"

ORIGINAL_ARTIFACT = RUNTIME / "serialized" / "Melanoma_cism_ready.pkl"
SPLIT_ARTIFACT_TEMPLATE = SPLIT_DIR / "split_{parts:02d}_cism_ready.pkl"
PARTS = [2, 4, 8, 16]


def fmt_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = df if columns is None else df[columns]
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def load_enriched_motifs(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    motifs = artifact["cism"].motifs_dataset.copy()
    if "p_value_direction" in motifs.columns:
        motifs = motifs[motifs["p_value_direction"].eq("proliferation")].copy()
    return motifs


def source_fov_from_split(fov: str) -> str:
    return str(fov).split("_p", 1)[0]


def build_original_sets() -> pd.Series:
    original = load_enriched_motifs(ORIGINAL_ARTIFACT)
    selected = pd.read_csv(SPATIAL_DIR / "fov_spatial_scale_per_fov.csv")
    selected["FOV"] = "FOV" + selected["fov_number"].astype(int).astype(str)
    keys = set(zip(selected["patient_uid"], selected["FOV"]))
    original = original[
        pd.MultiIndex.from_frame(original[["Patient_uId", "FOV"]]).isin(keys)
    ].copy()
    sets = (
        original.drop_duplicates(["Patient_uId", "FOV", "ID"])
        .groupby(["Patient_uId", "FOV"], observed=True)["ID"]
        .agg(lambda values: set(map(int, values)))
    )
    del original
    gc.collect()
    return sets


def summarize_split_results(original_sets: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregation_rows = []
    patient_rows = []
    tile_rows = []

    original_patient_sets = {}
    for (patient_uid, _fov), motif_set in original_sets.items():
        original_patient_sets.setdefault(patient_uid, set()).update(motif_set)

    for parts in PARTS:
        split = load_enriched_motifs(SPLIT_ARTIFACT_TEMPLATE.with_name(f"split_{parts:02d}_cism_ready.pkl"))
        split["source_fov"] = split["FOV"].map(source_fov_from_split)

        split_sets = (
            split.drop_duplicates(["Patient_uId", "source_fov", "ID"])
            .groupby(["Patient_uId", "source_fov"], observed=True)["ID"]
            .agg(lambda values: set(map(int, values)))
        )
        split_tile_counts = (
            split.drop_duplicates(["Patient_uId", "FOV", "ID"])
            .groupby(["Patient_uId", "FOV"], observed=True)["ID"]
            .nunique()
            .rename("tile_enriched_motif_count")
            .reset_index()
        )
        split_tile_counts["grid_parts"] = parts
        tile_rows.append(split_tile_counts)

        patient_split_sets = {}
        for (patient_uid, _source_fov), motif_set in split_sets.items():
            patient_split_sets.setdefault(patient_uid, set()).update(motif_set)

        keys = sorted(set(original_sets.index).union(split_sets.index))
        for patient_uid, fov in keys:
            original_set = original_sets.get((patient_uid, fov), set())
            split_set = split_sets.get((patient_uid, fov), set())
            union = original_set | split_set
            shared = original_set & split_set
            aggregation_rows.append(
                {
                    "grid_parts": parts,
                    "patient_uid": patient_uid,
                    "source_fov": fov,
                    "native_motif_count": len(original_set),
                    "split_aggregated_motif_count": len(split_set),
                    "shared_motif_count": len(shared),
                    "split_only_motif_count": len(split_set - original_set),
                    "native_only_motif_count": len(original_set - split_set),
                    "jaccard": len(shared) / len(union) if union else np.nan,
                }
            )

        for patient_uid in sorted(set(original_patient_sets).union(patient_split_sets)):
            original_set = original_patient_sets.get(patient_uid, set())
            split_set = patient_split_sets.get(patient_uid, set())
            union = original_set | split_set
            shared = original_set & split_set
            patient_rows.append(
                {
                    "grid_parts": parts,
                    "patient_uid": patient_uid,
                    "native_patient_motif_count": len(original_set),
                    "split_patient_motif_count": len(split_set),
                    "shared_patient_motif_count": len(shared),
                    "split_patient_only_count": len(split_set - original_set),
                    "native_patient_only_count": len(original_set - split_set),
                    "patient_jaccard": len(shared) / len(union) if union else np.nan,
                }
            )

        del split, split_sets, split_tile_counts
        gc.collect()

    aggregation = pd.DataFrame(aggregation_rows)
    patient = pd.DataFrame(patient_rows)
    tile = pd.concat(tile_rows, ignore_index=True)
    return aggregation, patient, tile


def write_plots(
    spatial_summary: pd.DataFrame,
    native_scale: pd.DataFrame,
    aggregation: pd.DataFrame,
    patient: pd.DataFrame,
    tile: pd.DataFrame,
) -> list[Path]:
    plot_dir = REPORT_DIR / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [f"{r.patient_class} {r.scale_label}" for r in native_scale.itertuples()]
    ax.bar(labels, native_scale["mean_significant_motif_count"], color=["#2f6f9f", "#d9843b", "#4f8f5f"])
    ax.set_ylabel("Mean enriched motifs per FOV")
    ax.set_title("Native FOV enriched motif counts by class and scale")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = plot_dir / "native_motif_counts_by_class_scale.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    split_summary = (
        aggregation.groupby("grid_parts", observed=True)
        .agg(
            native_mean=("native_motif_count", "mean"),
            split_mean=("split_aggregated_motif_count", "mean"),
            jaccard_mean=("jaccard", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(split_summary["grid_parts"], split_summary["native_mean"], marker="o", label="Native FOV")
    ax.plot(split_summary["grid_parts"], split_summary["split_mean"], marker="o", label="Split aggregate")
    ax.set_xlabel("Grid parts per FOV")
    ax.set_ylabel("Mean enriched motifs per source FOV")
    ax.set_title("Split aggregation compared with native FOV extraction")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "split_vs_native_mean_motifs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(split_summary["grid_parts"], split_summary["jaccard_mean"], marker="o", color="#7d4e9e")
    ax.set_xlabel("Grid parts per FOV")
    ax.set_ylabel("Mean Jaccard")
    ax.set_ylim(0, 1)
    ax.set_title("Motif identity overlap: split aggregate vs native FOV")
    fig.tight_layout()
    path = plot_dir / "split_vs_native_jaccard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    patient_summary = (
        patient.groupby("grid_parts", observed=True)
        .agg(
            native_patient_mean=("native_patient_motif_count", "mean"),
            split_patient_mean=("split_patient_motif_count", "mean"),
            patient_jaccard_mean=("patient_jaccard", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(patient_summary["grid_parts"], patient_summary["native_patient_mean"], marker="o", label="Native patient union")
    ax.plot(patient_summary["grid_parts"], patient_summary["split_patient_mean"], marker="o", label="Split patient union")
    ax.set_xlabel("Grid parts per FOV")
    ax.set_ylabel("Mean enriched motifs per patient")
    ax.set_title("Patient-level motif union after split aggregation")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "patient_union_split_vs_native.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    tile_counts = [tile.loc[tile["grid_parts"].eq(parts), "tile_enriched_motif_count"] for parts in PARTS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(tile_counts, labels=[str(parts) for parts in PARTS], showfliers=False)
    ax.set_xlabel("Grid parts per FOV")
    ax.set_ylabel("Enriched motifs per split tile")
    ax.set_title("Motifs recovered in progressively smaller tiles")
    fig.tight_layout()
    path = plot_dir / "split_tile_motif_count_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def pdf_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=18, weight="bold", va="top")
    y = 0.9
    for line in lines:
        fig.text(0.08, y, line, fontsize=10, va="top", family="monospace" if line.startswith("|") else None)
        y -= 0.025 if line else 0.018
        if y < 0.06:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.95
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pdf_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=16, weight="bold", y=0.96)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report() -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    spatial_summary = pd.read_csv(SPATIAL_DIR / "fov_spatial_scale_summary_by_class.csv")
    native_scale = pd.read_csv(NATIVE_DIR / "fov_motif_counts_by_class_and_scale.csv")
    same_patient_pairs = pd.read_csv(NATIVE_DIR / "same_patient_different_scale_pairs.csv")
    melanoma50 = pd.read_csv(NATIVE_DIR / "fov_motif_counts_per_fov.csv")
    melanoma50 = melanoma50[melanoma50["patient_uid"].eq("Melanoma50")].copy()

    original_sets = build_original_sets()
    aggregation, patient, tile = summarize_split_results(original_sets)

    aggregation.to_csv(REPORT_DIR / "split_vs_native_source_fov_aggregation.csv", index=False)
    patient.to_csv(REPORT_DIR / "split_vs_native_patient_aggregation.csv", index=False)
    tile.to_csv(REPORT_DIR / "split_tile_enriched_motif_counts.csv", index=False)

    split_summary = (
        aggregation.groupby("grid_parts", observed=True)
        .agg(
            mean_native_motifs=("native_motif_count", "mean"),
            mean_split_aggregated_motifs=("split_aggregated_motif_count", "mean"),
            mean_split_over_native=("split_aggregated_motif_count", lambda s: np.nan),
            mean_shared=("shared_motif_count", "mean"),
            mean_split_only=("split_only_motif_count", "mean"),
            mean_native_only=("native_only_motif_count", "mean"),
            mean_jaccard=("jaccard", "mean"),
        )
        .reset_index()
    )
    split_summary["mean_split_over_native"] = (
        aggregation.assign(ratio=aggregation["split_aggregated_motif_count"] / aggregation["native_motif_count"].replace(0, np.nan))
        .groupby("grid_parts", observed=True)["ratio"]
        .mean()
        .values
    )
    split_summary.to_csv(REPORT_DIR / "split_summary_by_grid_parts.csv", index=False)

    patient_summary = (
        patient.groupby("grid_parts", observed=True)
        .agg(
            mean_native_patient_motifs=("native_patient_motif_count", "mean"),
            mean_split_patient_motifs=("split_patient_motif_count", "mean"),
            mean_split_over_native_patient=("split_patient_motif_count", lambda s: np.nan),
            mean_patient_jaccard=("patient_jaccard", "mean"),
        )
        .reset_index()
    )
    patient_summary["mean_split_over_native_patient"] = (
        patient.assign(ratio=patient["split_patient_motif_count"] / patient["native_patient_motif_count"].replace(0, np.nan))
        .groupby("grid_parts", observed=True)["ratio"]
        .mean()
        .values
    )
    patient_summary.to_csv(REPORT_DIR / "patient_summary_by_grid_parts.csv", index=False)

    tile_summary = (
        tile.groupby("grid_parts", observed=True)["tile_enriched_motif_count"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    tile_summary.to_csv(REPORT_DIR / "tile_summary_by_grid_parts.csv", index=False)

    plots = write_plots(spatial_summary, native_scale, aggregation, patient, tile)

    spatial_md = spatial_summary.assign(
        mean_width=spatial_summary["mean_width"].map(lambda x: fmt_float(x, 1)),
        std_width=spatial_summary["std_width"].map(lambda x: fmt_float(x, 1)),
        mean_height=spatial_summary["mean_height"].map(lambda x: fmt_float(x, 1)),
        std_height=spatial_summary["std_height"].map(lambda x: fmt_float(x, 1)),
        mean_size=spatial_summary["mean_size"].map(fmt_int),
        std_size=spatial_summary["std_size"].map(fmt_int),
        mean_cell_count=spatial_summary["mean_cell_count"].map(lambda x: fmt_float(x, 1)),
    )
    native_md = native_scale.assign(
        mean_width=native_scale["mean_width"].map(lambda x: fmt_float(x, 1)),
        mean_height=native_scale["mean_height"].map(lambda x: fmt_float(x, 1)),
        mean_size=native_scale["mean_size"].map(fmt_int),
        mean_significant_motif_count=native_scale["mean_significant_motif_count"].map(lambda x: fmt_float(x, 1)),
        std_significant_motif_count=native_scale["std_significant_motif_count"].map(lambda x: fmt_float(x, 1)),
    )
    split_md = split_summary.assign(
        mean_native_motifs=split_summary["mean_native_motifs"].map(lambda x: fmt_float(x, 1)),
        mean_split_aggregated_motifs=split_summary["mean_split_aggregated_motifs"].map(lambda x: fmt_float(x, 1)),
        mean_split_over_native=split_summary["mean_split_over_native"].map(lambda x: fmt_float(x, 2)),
        mean_shared=split_summary["mean_shared"].map(lambda x: fmt_float(x, 1)),
        mean_split_only=split_summary["mean_split_only"].map(lambda x: fmt_float(x, 1)),
        mean_native_only=split_summary["mean_native_only"].map(lambda x: fmt_float(x, 1)),
        mean_jaccard=split_summary["mean_jaccard"].map(lambda x: fmt_float(x, 3)),
    )
    patient_md = patient_summary.assign(
        mean_native_patient_motifs=patient_summary["mean_native_patient_motifs"].map(lambda x: fmt_float(x, 1)),
        mean_split_patient_motifs=patient_summary["mean_split_patient_motifs"].map(lambda x: fmt_float(x, 1)),
        mean_split_over_native_patient=patient_summary["mean_split_over_native_patient"].map(lambda x: fmt_float(x, 2)),
        mean_patient_jaccard=patient_summary["mean_patient_jaccard"].map(lambda x: fmt_float(x, 3)),
    )
    tile_md = tile_summary.assign(
        mean=tile_summary["mean"].map(lambda x: fmt_float(x, 1)),
        std=tile_summary["std"].map(lambda x: fmt_float(x, 1)),
        median=tile_summary["median"].map(lambda x: fmt_float(x, 1)),
        min=tile_summary["min"].map(fmt_int),
        max=tile_summary["max"].map(fmt_int),
    )

    mel50_display = melanoma50[["fov", "scale_label", "size", "significant_motif_count"]].copy()
    mel50_display["size"] = mel50_display["size"].map(fmt_int)
    mel50_display["significant_motif_count"] = mel50_display["significant_motif_count"].map(fmt_int)

    md_lines = [
        "# Melanoma NN/NP FOV Spatial-Scale Motif Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Scope: enriched FANMOD/CISM motifs only (`p_value_direction == proliferation`).",
        "",
        "## Main Takeaways",
        "",
        "- NP FOVs are effectively one full-field spatial scale, whereas NN contains full FOVs and a smaller approximately quarter-area FOV scale.",
        "- Native small NN FOVs are roughly one quarter of full-FOV area but retain more than half of the enriched motif count on average.",
        "- In Melanoma50, the two small FOVs together recover essentially the same number of enriched motif IDs as the large FOV, while contributing a large small-only motif component.",
        "- Split-FOV aggregation increasingly changes motif identity relative to native FOV extraction; the detailed split/native overlap is summarized below by grid size.",
        "",
        "## FOV Spatial Scale By Class",
        "",
        markdown_table(spatial_md[["patient_class", "n_fovs", "mean_width", "std_width", "mean_height", "std_height", "mean_size", "std_size", "mean_cell_count"]]),
        "",
        "## Native Enriched Motif Counts By Class And Scale",
        "",
        markdown_table(native_md[["patient_class", "scale_label", "n_fovs", "mean_width", "mean_height", "mean_size", "mean_significant_motif_count", "std_significant_motif_count"]]),
        "",
        "## Melanoma50 Native FOV Comparison",
        "",
        markdown_table(mel50_display),
        "",
        "Melanoma50 enriched motif overlaps: FOV418 = 7,618, FOV420 = 7,282, FOV418 intersection FOV420 = 3,212, small-FOV union = 11,688, all-FOV patient union = 17,814.",
        "",
        "## Same-Patient Small Versus Full FOVs",
        "",
        markdown_table(
            same_patient_pairs[
                [
                    "patient_uid",
                    "small_fov",
                    "full_fov",
                    "small_significant_motif_count",
                    "full_significant_motif_count",
                    "small_over_full_significant_motif_count",
                ]
            ].assign(
                small_over_full_significant_motif_count=lambda d: d["small_over_full_significant_motif_count"].map(lambda x: fmt_float(x, 3))
            )
        ),
        "",
        "## Split-FOV Aggregation Versus Native FOV Extraction",
        "",
        markdown_table(split_md),
        "",
        "## Patient-Level Split Aggregation",
        "",
        markdown_table(patient_md),
        "",
        "## Motifs Per Split Tile",
        "",
        markdown_table(tile_md),
        "",
        "## Figures",
        "",
    ]
    for path in plots:
        md_lines.extend([f"![{path.stem}]({path.resolve()})", ""])
    md_lines.extend(
        [
            "## Output Tables",
            "",
            f"- `{(REPORT_DIR / 'split_vs_native_source_fov_aggregation.csv').resolve()}`",
            f"- `{(REPORT_DIR / 'split_vs_native_patient_aggregation.csv').resolve()}`",
            f"- `{(REPORT_DIR / 'split_summary_by_grid_parts.csv').resolve()}`",
            f"- `{(REPORT_DIR / 'patient_summary_by_grid_parts.csv').resolve()}`",
            f"- `{(REPORT_DIR / 'tile_summary_by_grid_parts.csv').resolve()}`",
            "",
            "Note: this report summarizes enriched motif extraction and split/native aggregation. Classification-performance and optimal-stringency re-evaluation are not included in this PDF because the full tuning summarizer is substantially heavier than the reporting pass.",
            "",
        ]
    )

    md_path = REPORT_DIR / "melanoma_fov_scale_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    pdf_path = REPORT_DIR / "melanoma_fov_scale_report.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf_text_page(pdf, "Melanoma NN/NP FOV Spatial-Scale Motif Report", md_lines[:80])
        pdf_text_page(pdf, "Split-FOV Summary Tables", md_lines[80:160])
        for plot in plots:
            pdf_image_page(pdf, plot.stem.replace("_", " ").title(), plot)

    return md_path, pdf_path


def main() -> None:
    md_path, pdf_path = build_report()
    print(f"Markdown report: {md_path}")
    print(f"PDF report: {pdf_path}")


if __name__ == "__main__":
    main()
