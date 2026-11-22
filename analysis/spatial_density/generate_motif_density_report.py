from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
import shutil
import textwrap
from pathlib import Path
from typing import Any

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
REPORT_DIR = ANALYSIS_DIR / "reports"
REPORT_FIGURE_DIR = REPORT_DIR / "figures" / "motif_density"
REPORT_MD = REPORT_DIR / "motif_density_report.md"
REPORT_PDF = REPORT_DIR / "motif_density_report.pdf"
SOURCE_REPORT_DIR = (
    PROJECT_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "motif_density_analysis_report_notebook03_discriminative"
)
SOURCE_PLOT_DIR = SOURCE_REPORT_DIR / "plots"

INSTANCE_CSV = SOURCE_REPORT_DIR / "motif_density_instance_metrics.csv"
NULL_CSV = SOURCE_REPORT_DIR / "motif_density_null_samples.csv"
SUMMARY_CSV = SOURCE_REPORT_DIR / "motif_density_summary.csv"
MOTIF_AVERAGE_TESTS_CSV = SOURCE_REPORT_DIR / "motif_density_motif_average_tests.csv"

EXCLUDED_MOTIFS: list[str] = []
FOCUS_RADIUS_METRIC = "radius_200_density"
FOCUS_RADIUS_PIXELS = 200
PIXEL_SIZE_MICRONS = 0.5


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if df.empty:
        return "_No rows._"
    table_df = df.copy()

    for column in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[column]):
            table_df[column] = table_df[column].map(
                lambda value: format(value, floatfmt) if pd.notna(value) else ""
            )
        else:
            table_df[column] = table_df[column].map(lambda value: "" if pd.isna(value) else str(value))

    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [clean(column) for column in table_df.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table_df.iterrows():
        rows.append("| " + " | ".join(clean(row[column]) for column in table_df.columns) + " |")
    return "\n".join(rows)


def copy_figures() -> list[Path]:
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for stale_figure in REPORT_FIGURE_DIR.glob("*.png"):
        stale_figure.unlink()
    copied: list[Path] = []
    for source in sorted(SOURCE_PLOT_DIR.glob("*.png")):
        destination = REPORT_FIGURE_DIR / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def grouped_figures(figures: list[Path]) -> dict[str, list[Path]]:
    groups = {
        "Boxplots": [],
        "Ranked Enrichment Plots": [],
        "Observed Versus Null Distributions": [],
        "Patch Density Overlays": [],
        "Other": [],
    }
    for figure in figures:
        name = figure.name
        if name.startswith("boxplot_"):
            groups["Boxplots"].append(figure)
        elif name.startswith("enrichment_"):
            groups["Ranked Enrichment Plots"].append(figure)
        elif name.startswith("observed_vs_null_"):
            groups["Observed Versus Null Distributions"].append(figure)
        elif name.startswith("patch_density_overlay_"):
            groups["Patch Density Overlays"].append(figure)
        else:
            groups["Other"].append(figure)
    return {group: paths for group, paths in groups.items() if paths}


def title_from_filename(path: Path) -> str:
    title = path.stem.replace("_", " ")
    title = title.replace("knn", "kNN")
    title = title.replace("fov", "FOV")
    title = title.replace("nn", "NN")
    title = title.replace("np", "NP")
    return title.title().replace("Knn", "kNN")


def summarize_instances(instances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if instances.empty:
        return pd.DataFrame(), pd.DataFrame()

    association = (
        instances.groupby("motif_association", as_index=False, observed=True)
        .agg(
            n_instances=("instance_id", "nunique"),
            n_motifs=("motif_id", "nunique"),
            n_patients=("patient", "nunique"),
            n_fovs=("FOV", "nunique"),
            median_radius_200_density=("radius_200_density", "median"),
            mean_radius_200_density=("radius_200_density", "mean"),
            median_graph_hop_1_count=("graph_hop_1_count", "median"),
        )
        .sort_values("motif_association")
    )

    motif = (
        instances.groupby(["motif_association", "motif_id"], as_index=False, observed=True)
        .agg(
            n_instances=("instance_id", "nunique"),
            n_patients=("patient", "nunique"),
            n_fovs=("FOV", "nunique"),
            motif_size_median=("motif_size", "median"),
            median_radius_200_density=("radius_200_density", "median"),
            mean_radius_200_density=("radius_200_density", "mean"),
            median_knn_10_density=("knn_10_density", "median"),
            median_graph_hop_1_count=("graph_hop_1_count", "median"),
        )
        .sort_values(["motif_association", "motif_id"])
    )
    return association, motif


def summarize_nulls(nulls: pd.DataFrame) -> pd.DataFrame:
    if nulls.empty:
        return pd.DataFrame()
    return (
        nulls.groupby("null_type", as_index=False, observed=True)
        .agg(
            n_samples=("sample_index", "count"),
            n_motifs=("motif_id", "nunique"),
            n_instances=("instance_id", "nunique"),
            median_radius_200_density=("radius_200_density", "median"),
            mean_radius_200_density=("radius_200_density", "mean"),
        )
        .sort_values("null_type")
    )


def summarize_signals(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    top = (
        summary.sort_values(["q_value", "p_value", "enrichment_delta"], ascending=[True, True, False])
        .head(15)
        .copy()
    )
    focus_radius = summary[
        (summary["metric"] == FOCUS_RADIUS_METRIC)
        & (summary["null_type"] == "random_connected_cellset")
    ].copy()
    if not focus_radius.empty:
        focus_radius = focus_radius.sort_values("enrichment_delta").copy()
    return top, focus_radius


def compact_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = [
        "motif_id",
        "metric",
        "null_type",
        "n_observed",
        "n_null",
        "observed_mean",
        "null_mean",
        "enrichment_delta",
        "enrichment_ratio",
        "p_value",
        "q_value",
    ]
    return df[columns].copy()


def compact_motif_average_tests(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = [
        "metric",
        "distribution",
        "n_nn_motifs",
        "n_np_motifs",
        "nn_mean",
        "np_mean",
        "mean_difference_nn_minus_np",
        "welch_t_p_value",
        "mannwhitney_p_value",
        "kl_nn_to_np",
        "kl_np_to_nn",
    ]
    return df[[column for column in columns if column in df.columns]].copy()


def generate_markdown(
    instances: pd.DataFrame,
    nulls: pd.DataFrame,
    summary: pd.DataFrame,
    motif_average_tests: pd.DataFrame,
    figures: list[Path],
) -> str:
    association, motif = summarize_instances(instances)
    null_summary = summarize_nulls(nulls)
    top_signals, focus_radius = summarize_signals(summary)
    groups = grouped_figures(figures)
    focus_radius_microns = FOCUS_RADIUS_PIXELS * PIXEL_SIZE_MICRONS

    n_significant_10 = int((summary["q_value"] <= 0.10).sum()) if "q_value" in summary else 0
    n_significant_20 = int((summary["q_value"] <= 0.20).sum()) if "q_value" in summary else 0
    tissue_states = sorted(instances["tissue_state_filter"].dropna().unique()) if "tissue_state_filter" in instances else []

    lines = [
        "# Motif Appearance Versus Tissue Density Report",
        "",
        f"Generated from saved outputs under `{SOURCE_REPORT_DIR}`.",
        "",
        "## Analysis Status",
        "",
        f"- Motif-instance rows: **{len(instances)}**.",
        f"- Null-sample rows: **{len(nulls)}**.",
        f"- Summary rows: **{len(summary)}**.",
        f"- Motifs with recovered instances: **{instances['motif_id'].nunique() if not instances.empty else 0}**.",
        f"- Motif associations represented: **{', '.join(association['motif_association'].astype(str)) if not association.empty else 'none'}**.",
        f"- Tissue-state filters present in instance table: **{', '.join(tissue_states) if tissue_states else 'none'}**.",
        f"- Copied plot files included in this report: **{len(figures)}**.",
        f"- Excluded motif IDs for this bounded run: **{', '.join(EXCLUDED_MOTIFS) if EXCLUDED_MOTIFS else 'none'}**.",
        "",
        "## Run Settings",
        "",
        "- Motif panel: 19 hard discriminative motifs reconstructed from notebooks 2/3 (`discriminator.get_features(best_feature_conf)` and `GetMotifsResult.get_motifs_by_class()`).",
        "- Motif association labels: `direction` column from the source panel.",
        "- Notebook 03b/04b soft-selection, LASSO, and refined-panel outputs are excluded.",
        "- Tissue scope: NN and NP FOVs only.",
        "- Euclidean radius thresholds: 25, 50, 100, 150, 200 pixels.",
        f"- Pixel-to-micron conversion: 1 pixel = {PIXEL_SIZE_MICRONS:g} micron; the {FOCUS_RADIUS_PIXELS} pixel focus radius is {focus_radius_microns:g} microns.",
        f"- Euclidean density units: cells per micron^2, using area pi * (radius_pixels * {PIXEL_SIZE_MICRONS:g})^2.",
        "- Euclidean kNN mean distances are reported in microns; graph-distance metrics remain in graph hops.",
        "- Euclidean kNN values: 5, 10, 20.",
        "- Graph hop radii: 1, 2, 3.",
        "- Patch size for visual overlays: 100 x 100 pixels, equivalent to 50 x 50 microns.",
        "- Null samples in this bounded report run: 20 per observed motif instance.",
        "- Bounded execution flags: ascending-count FOV selection, max one FOV per motif, max 100 hits per FOV.",
        "- Motif-hit coordinates use FOV-local graph node IDs mapped back to raw cell rows; raw `cell_id` values are retained separately.",
        "",
        "## What The Null Local Density Means",
        "",
        "For each observed motif instance, the analysis creates local-density measurements from randomized locations or randomized connected cell sets within the same FOV. The null value is therefore not a single global background number. It is a motif-specific background distribution built from the same FOVs, same number of observed instances, and, for connected-cell-set nulls, the same motif size.",
        "",
        "This is why each motif has a different null local density. Motifs occur in different patients and FOVs, have different instance counts, and can have different motif sizes. A motif found mostly in dense NP FOVs gets a denser null background than a motif found mostly in sparse NN FOVs. That FOV matching is intentional: it asks whether a motif is unusually dense or sparse relative to the tissue regions where that motif could have appeared, instead of comparing every motif to one pooled tissue-wide background.",
        "",
        "## Motif-Average Association Tests",
        "",
        "These tests use one value per motif: the mean local density across that motif's recovered instances. This avoids treating multiple instances of the same motif as independent motifs. Welch's t-test compares the NN and NP motif-average expectations, Mann-Whitney compares the rank distributions, and KL divergence summarizes the smoothed histogram difference between NN and NP motif-average distributions.",
        "",
        dataframe_to_markdown(compact_motif_average_tests(motif_average_tests), ".4g"),
        "",
        "## Instance Coverage",
        "",
        dataframe_to_markdown(association, ".4g"),
        "",
        "## Motif-Level Coverage And Local Density",
        "",
        dataframe_to_markdown(motif, ".4g"),
        "",
        "## Null Sample Coverage",
        "",
        dataframe_to_markdown(null_summary, ".4g"),
        "",
        "## Top Density Signals",
        "",
        f"- BH-significant rows at q <= 0.10: **{n_significant_10}**.",
        f"- BH-significant rows at q <= 0.20: **{n_significant_20}**.",
        "- The table below is ranked by q-value and raw permutation p-value. Treat these as screening signals because this was a bounded, sampled run.",
        "",
        dataframe_to_markdown(compact_signal_table(top_signals), ".4g"),
        "",
        f"## Radius-{FOCUS_RADIUS_PIXELS} Connected-Cellset Comparison",
        "",
        f"This table focuses on the {FOCUS_RADIUS_PIXELS}-pixel ({focus_radius_microns:g}-micron) Euclidean density metric using the random connected-cell-set null. Density is reported as cells per micron^2. Negative enrichment means observed motif neighborhoods were sparser than matched connected random cell sets in the same FOV; positive enrichment means denser.",
        "",
        dataframe_to_markdown(compact_signal_table(focus_radius), ".4g"),
        "",
        "## Figure Index",
        "",
    ]

    for group, paths in groups.items():
        lines.extend([f"### {group}", ""])
        for figure in paths:
            rel_path = figure.relative_to(REPORT_DIR)
            lines.extend([f"#### {title_from_filename(figure)}", "", f"![{title_from_filename(figure)}]({rel_path})", ""])

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            f"- The boxplots summarize the local radius-{FOCUS_RADIUS_PIXELS} pixel cell density ({focus_radius_microns:g} microns), reported as cells per micron^2. NN-associated motifs use the blue/light-blue style, NP-associated motifs use the red/orange style, and PN is reserved as greenish when present in comparison plots.",
            "- The observed-by-motif boxplot shows the spread of measured local densities around real motif instances.",
            "- The null-by-motif boxplot shows the local densities expected for the same motif's FOVs and instance structure under randomization.",
            "- The association-level boxplots aggregate motif instances by final motif association, so they answer a broader question than motif-specific plots.",
            "- No rows pass BH correction at q <= 0.10 in this bounded report. The strongest raw signals are therefore best used to prioritize figure-making and follow-up runs, not as standalone claims.",
            "",
            "## Output Tables Used",
            "",
            f"- `{INSTANCE_CSV}`",
            f"- `{NULL_CSV}`",
            f"- `{SUMMARY_CSV}`",
            f"- `{MOTIF_AVERAGE_TESTS_CSV}`",
            "",
        ]
    )
    return "\n".join(lines)


def add_text_page(pdf: PdfPages, title: str, body: str, *, fontsize: int = 9) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=15, weight="bold", va="top")
    wrapped_lines: list[str] = []
    for paragraph in body.split("\n"):
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=95))
    ax.text(0, 1, "\n".join(wrapped_lines), fontsize=fontsize, va="top", family="DejaVu Sans")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, *, max_rows: int = 24) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor="white")
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    table_df = df.head(max_rows).copy()
    if table_df.empty:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
    else:
        for column in table_df.columns:
            if pd.api.types.is_float_dtype(table_df[column]):
                table_df[column] = table_df[column].map(lambda value: f"{value:.4g}")
            else:
                table_df[column] = table_df[column].astype(str)
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 1.2)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#EFEFEF")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor="white")
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf(
    markdown_text: str,
    instances: pd.DataFrame,
    nulls: pd.DataFrame,
    summary: pd.DataFrame,
    motif_average_tests: pd.DataFrame,
    figures: list[Path],
) -> None:
    association, motif = summarize_instances(instances)
    null_summary = summarize_nulls(nulls)
    top_signals, focus_radius = summarize_signals(summary)

    with PdfPages(REPORT_PDF) as pdf:
        overview = "\n".join(markdown_text.split("## Motif-Level Coverage", 1)[0].splitlines()[2:])
        add_text_page(pdf, "Motif Appearance Versus Tissue Density Report", overview, fontsize=9)
        add_table_page(pdf, "Motif-Average Association Tests", compact_motif_average_tests(motif_average_tests), max_rows=12)
        add_table_page(pdf, "Instance Coverage", association, max_rows=12)
        add_table_page(pdf, "Motif-Level Coverage And Local Density", motif, max_rows=30)
        add_table_page(pdf, "Null Sample Coverage", null_summary, max_rows=12)
        add_table_page(pdf, "Top Density Signals", compact_signal_table(top_signals), max_rows=18)
        add_table_page(pdf, f"Radius-{FOCUS_RADIUS_PIXELS} Connected-Cellset Comparison", compact_signal_table(focus_radius), max_rows=22)
        for group, paths in grouped_figures(figures).items():
            add_text_page(pdf, group, f"{len(paths)} plots included in this section.", fontsize=11)
            for figure in paths:
                add_image_page(pdf, title_from_filename(figure), figure)
        notes = markdown_text.split("## Interpretation Notes", 1)[1] if "## Interpretation Notes" in markdown_text else ""
        add_text_page(pdf, "Interpretation Notes", "## Interpretation Notes" + notes, fontsize=9)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    instances = read_csv(INSTANCE_CSV)
    nulls = read_csv(NULL_CSV)
    summary = read_csv(SUMMARY_CSV)
    motif_average_tests = read_csv(MOTIF_AVERAGE_TESTS_CSV)
    figures = copy_figures()

    markdown_text = generate_markdown(instances, nulls, summary, motif_average_tests, figures)
    REPORT_MD.write_text(markdown_text, encoding="utf-8")
    generate_pdf(markdown_text, instances, nulls, summary, motif_average_tests, figures)

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_PDF}")
    print(f"Copied {len(figures)} figures to {REPORT_FIGURE_DIR}")


if __name__ == "__main__":
    main()
