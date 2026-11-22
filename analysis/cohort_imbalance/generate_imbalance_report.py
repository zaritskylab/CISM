from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import json
import os
import shutil
import textwrap
from datetime import datetime
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
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
OUTPUT_FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ANALYSIS_DIR / "reports"
REPORT_FIGURE_DIR = REPORT_DIR / "figures"
REPORT_MD = REPORT_DIR / "imbalance_sensitivity_report.md"
REPORT_PDF = REPORT_DIR / "imbalance_sensitivity_report.pdf"
PANEL_CD_CSV = OUTPUT_DIR / "panels_C_D_auc_stringency_summary.csv"
PANEL_CD_MD = REPORT_DIR / "panels_C_D_auc_stringency_summary.md"


FIGURES = [
    "imbalance_summary_four_panel.png",
    "imbalance_auc_and_stringency.png",
    "imbalance_motif_pool_by_class.png",
    "imbalance_original_motif_reuse.png",
    "imbalance_motifs_vs_stringency_aggregated.png",
    "imbalance_NP_minus_NN_heatmap_sampled_NN_signed_log.png",
    "imbalance_NP_minus_NN_heatmap_sampled_NP_signed_log.png",
    "imbalance_NP_over_NN_heatmap_sampled_NN_log10.png",
    "imbalance_NP_over_NN_heatmap_sampled_NP_log10.png",
    "imbalance_NP_over_NN_heatmap_sampled_NN_normal.png",
    "imbalance_NP_over_NN_heatmap_sampled_NP_normal.png",
    "imbalance_motifs_vs_stringency_sampled_NN.png",
    "imbalance_motifs_vs_stringency_sampled_NP.png",
]


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(name: str) -> dict[str, Any]:
    path = OUTPUT_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    table_df = df.copy()

    def stringify(value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return format(value, floatfmt)
        return str(value)

    for column in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[column]):
            table_df[column] = table_df[column].map(
                lambda value: format(value, floatfmt) if pd.notna(value) else ""
            )
        else:
            table_df[column] = table_df[column].map(stringify)

    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [clean(column) for column in table_df.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table_df.iterrows():
        rows.append("| " + " | ".join(clean(row[column]) for column in table_df.columns) + " |")
    return "\n".join(rows)


def collect_tables() -> dict[str, pd.DataFrame]:
    return {
        "results": read_csv("imbalanced_bootstrap_results.csv"),
        "ci": read_csv("imbalanced_bootstrap_auc_ci.csv"),
        "trials": read_csv("stringency_optuna_trials.csv"),
        "motifs_vs_stringency": read_csv("discriminative_motifs_vs_stringency_by_class.csv"),
        "original_usage": read_csv("bootstrap_original_fold_union_motif_usage.csv"),
    }


def summarize_results(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = tables["results"]
    usage = tables["original_usage"]
    motifs_vs_stringency = tables["motifs_vs_stringency"]

    core = pd.DataFrame()
    if not results.empty:
        core = (
            results.groupby(["sampled_label", "bootstrap_fraction"], as_index=False, observed=True)
            .agg(
                n_bootstraps=("bootstrap_index", "count"),
                sampled_n=("sampled_n", "mean"),
                raw_discriminative_total=("n_discriminative_motifs", "mean"),
                raw_NN=("n_discriminative_motifs_NN", "mean"),
                raw_NP=("n_discriminative_motifs_NP", "mean"),
                pass_stringency_total=("optimal_full_bootstrap_discriminative_motifs", "mean"),
                pass_NN=("optimal_full_bootstrap_discriminative_motifs_NN", "mean"),
                pass_NP=("optimal_full_bootstrap_discriminative_motifs_NP", "mean"),
                fold_union_NN=("optimal_fold_union_discriminative_motifs_NN", "mean"),
                fold_union_NP=("optimal_fold_union_discriminative_motifs_NP", "mean"),
                optimized_stringency=("optimal_stringency", "mean"),
                stringency_std=("optimal_stringency", "std"),
                auc=("optimal_auc", "mean"),
                auc_std=("optimal_auc", "std"),
            )
            .sort_values(["sampled_label", "bootstrap_fraction"])
        )

    reuse = pd.DataFrame()
    if not usage.empty:
        reuse = (
            usage.groupby(["sampled_label", "bootstrap_fraction"], as_index=False, observed=True)
            .agg(
                fold_union_total=("fold_union_motifs_cleared_count", "mean"),
                fraction_original_NN=("fraction_original_NN_motifs_used", "mean"),
                fraction_original_NP=("fraction_original_NP_motifs_used", "mean"),
                fraction_original_all=("fraction_all_original_motifs_used", "mean"),
            )
            .sort_values(["sampled_label", "bootstrap_fraction"])
        )

    stringency_counts = pd.DataFrame()
    if not motifs_vs_stringency.empty:
        stringency_counts = (
            motifs_vs_stringency.groupby(
                ["sampled_label", "bootstrap_fraction", "stringency", "motif_label"],
                as_index=False,
                observed=True,
            )
            .agg(mean_motifs_passing=("discriminative_motifs_passing", "mean"))
            .sort_values(["sampled_label", "motif_label", "bootstrap_fraction", "stringency"])
        )

    return core, reuse, stringency_counts


def compact_core_table(core: pd.DataFrame) -> pd.DataFrame:
    if core.empty:
        return core
    cols = [
        "sampled_label",
        "bootstrap_fraction",
        "sampled_n",
        "raw_discriminative_total",
        "pass_stringency_total",
        "optimized_stringency",
        "auc",
    ]
    return core[cols].copy()


def compact_reuse_table(reuse: pd.DataFrame) -> pd.DataFrame:
    if reuse.empty:
        return reuse
    cols = [
        "sampled_label",
        "bootstrap_fraction",
        "fold_union_total",
        "fraction_original_NN",
        "fraction_original_NP",
        "fraction_original_all",
    ]
    return reuse[cols].copy()


def compact_panel_cd_table(core: pd.DataFrame) -> pd.DataFrame:
    if core.empty:
        return core
    return core[
        [
            "sampled_label",
            "bootstrap_fraction",
            "n_bootstraps",
            "optimized_stringency",
            "stringency_std",
            "auc",
            "auc_std",
        ]
    ].rename(
        columns={
            "sampled_label": "sampled_class",
            "bootstrap_fraction": "sampling_rate",
            "optimized_stringency": "mean_stringency",
            "auc": "mean_auc",
        }
    )


def copy_figures() -> dict[str, Path]:
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for figure_name in FIGURES:
        source = OUTPUT_FIGURE_DIR / figure_name
        if not source.exists():
            continue
        destination = REPORT_FIGURE_DIR / figure_name
        shutil.copy2(source, destination)
        copied[figure_name] = destination
    return copied


def generate_markdown(
    config: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    core: pd.DataFrame,
    reuse: pd.DataFrame,
    stringency_counts: pd.DataFrame,
    figures: dict[str, Path],
) -> str:
    results = tables["results"]
    ci = tables["ci"]
    trials = tables["trials"]
    original_usage = tables["original_usage"]

    n_bootstraps = int(config.get("n_bootstraps", results["bootstrap_index"].nunique() if not results.empty else 0))
    fractions = sorted(results["bootstrap_fraction"].unique()) if not results.empty else config.get("bootstrap_fractions", [])
    labels = config.get("labels", ["NN", "NP"])
    original_motif_count = len([c for c in original_usage.columns if c.startswith("original_motif_")])

    lines = [
        "# Imbalance Sensitivity Analysis Report",
        "",
        f"Generated from saved outputs under `{OUTPUT_DIR}`.",
        "",
        "## Analysis Status",
        "",
        f"- Bootstrap result rows: **{len(results)}**.",
        f"- Optuna trial rows: **{len(trials)}**.",
        f"- Bootstrap fractions: **{', '.join(str(x) for x in fractions)}**.",
        f"- Bootstraps per sampled class/fraction: **{n_bootstraps}**.",
        f"- Original no-bootstrap fold-union motif count: **{original_motif_count}**.",
        "",
        "## Experiment Settings",
        "",
        f"- Labels: {', '.join(labels)}.",
        "- In each experiment, one class was downsampled and the full opposite class was retained.",
        "- Sampling fractions were evaluated as fractions, not percentages.",
        "- Hard-discriminative motifs were re-extracted in a leakage-safe LOOCV workflow.",
        f"- Stringency was optimized independently for each bootstrap using {config.get('n_optuna_trials', 10)} Optuna trials.",
        f"- Optuna stringency range: {config.get('stringency_range', [0.10, 0.95])}.",
        f"- Maximum features per class/stringency: {config.get('max_features', 30)}.",
        f"- FANMOD p-value threshold: {config.get('fanmod_p_value_threshold', 0.05)}.",
        "- Classifier: RandomForestClassifier with the same settings used in the notebooks (`random_state`, `n_jobs=-1`).",
        "",
        "## Summary Figures",
        "",
    ]

    figure_titles = {
        "imbalance_summary_four_panel.png": "Overall imbalance summary",
        "imbalance_auc_and_stringency.png": "AUC and optimized stringency",
        "imbalance_motif_pool_by_class.png": "Discriminative motif pool by class",
        "imbalance_original_motif_reuse.png": "Original motif reuse",
        "imbalance_motifs_vs_stringency_aggregated.png": "Aggregated motifs vs stringency",
        "imbalance_NP_minus_NN_heatmap_sampled_NN_signed_log.png": "NP minus NN discriminative motifs, sampled NN, signed log scale",
        "imbalance_NP_minus_NN_heatmap_sampled_NP_signed_log.png": "NP minus NN discriminative motifs, sampled NP, signed log scale",
        "imbalance_NP_over_NN_heatmap_sampled_NN_log10.png": "NP-to-NN discriminative motif ratio, sampled NN, log10 scale",
        "imbalance_NP_over_NN_heatmap_sampled_NP_log10.png": "NP-to-NN discriminative motif ratio, sampled NP, log10 scale",
        "imbalance_NP_over_NN_heatmap_sampled_NN_normal.png": "NP-to-NN discriminative motif ratio, sampled NN, normal scale",
        "imbalance_NP_over_NN_heatmap_sampled_NP_normal.png": "NP-to-NN discriminative motif ratio, sampled NP, normal scale",
        "imbalance_motifs_vs_stringency_sampled_NN.png": "Motifs vs stringency when sampling NN",
        "imbalance_motifs_vs_stringency_sampled_NP.png": "Motifs vs stringency when sampling NP",
    }
    for figure_name in FIGURES:
        if figure_name in figures:
            lines.extend(
                [
                    f"### {figure_titles[figure_name]}",
                    "",
                    f"![{figure_titles[figure_name]}](figures/{figure_name})",
                    "",
                ]
            )

    lines.extend(
        [
            "## Panels C and D Summary",
            "",
            "Means and sample standard deviations (`ddof=1`) were calculated across the 10 bootstrap replicates for each sampled class and sampling rate.",
            "",
            dataframe_to_markdown(compact_panel_cd_table(core), ".3f"),
            "",
            "## Core Results",
            "",
            dataframe_to_markdown(compact_core_table(core), ".3f"),
            "",
            "## Original Motif Reuse",
            "",
            dataframe_to_markdown(compact_reuse_table(reuse), ".3f"),
            "",
            "## Motifs vs Stringency",
            "",
            "The fixed-grid motif-count analysis used stringency values from `0.0` to `1.0` in increments of `0.1`. Counts are class-specific and are not restricted to the Optuna-selected best stringency.",
            "",
        ]
    )
    selected_stringencies = stringency_counts[
        stringency_counts["stringency"].isin([0.0, 0.5, 0.9, 1.0])
    ].copy()
    lines.extend([dataframe_to_markdown(selected_stringencies.head(60), ".3f"), ""])

    lines.extend(
        [
            "## Findings",
            "",
            "- Severe imbalance inflated the apparent hard-discriminative motif pool. At a sampled-class fraction of `0.1`, approximately 29,000 hard-discriminative motifs were found on average; by `0.9`, this decreased to approximately 15,000.",
            "- The inflation was asymmetric and favored the fully represented class. When `NN` was downsampled to `0.1`, the raw motif pool was dominated by `NP`-associated motifs; when `NP` was downsampled, the reciprocal pattern was observed.",
            "- Optimized stringency partially compensated for this inflation. The mean selected stringency was highest at the most imbalanced settings, approximately `0.67-0.69`, and stabilized near `0.49-0.50` at high sampling fractions.",
            "- The number of motifs passing both hard discriminativity and optimized stringency dropped sharply as balance improved. At `0.1`, thousands of motifs could pass both filters; at `0.9`, only a few motifs passed on average.",
            "- Original motif reuse was low under severe imbalance. The fraction of original fold-union motifs recovered was below approximately `0.1` at `0.1` sampling and improved with more balanced sampling, peaking around intermediate-to-high fractions.",
            "- The fixed-grid motif-vs-stringency plots show that low stringency thresholds retain large motif sets under imbalance, whereas moderate-to-high stringency values rapidly reduce the selected pool.",
            "",
            "## Interpretation",
            "",
            "These results indicate that class imbalance can substantially distort hard-discriminative motif discovery by creating many apparent class-specific motifs, particularly for the overrepresented class. The stringency optimization acts as an important regularizing step, but motif identity is still less stable under strong imbalance. Balanced group representation improves motif stability, reduces the inflation of apparent discriminative motifs, and makes the selected motif panel easier to interpret.",
            "",
            "## Output Tables Used",
            "",
            "- `imbalanced_bootstrap_results.csv`",
            "- `imbalanced_bootstrap_auc_ci.csv`",
            "- `stringency_optuna_trials.csv`",
            "- `discriminative_motifs_vs_stringency_by_class.csv`",
            "- `bootstrap_original_fold_union_motif_usage.csv`",
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


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, *, max_rows: int = 28) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor="white")
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    table_df = df.head(max_rows).copy()
    if table_df.empty:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
    else:
        for column in table_df.columns:
            if pd.api.types.is_float_dtype(table_df[column]):
                table_df[column] = table_df[column].map(lambda value: f"{value:.3f}")
            else:
                table_df[column] = table_df[column].astype(str)
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
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
    core: pd.DataFrame,
    reuse: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(REPORT_PDF) as pdf:
        add_text_page(
            pdf,
            "Imbalance Sensitivity Analysis Report",
            "Generated from saved imbalance-analysis outputs.\n\n"
            + "\n".join(markdown_text.split("## Findings", 1)[0].splitlines()[2:40]),
            fontsize=9,
        )
        add_table_page(
            pdf,
            "Panels C and D: Stringency and AUC Summary",
            compact_panel_cd_table(core),
            max_rows=24,
        )
        add_table_page(pdf, "Core Results", compact_core_table(core), max_rows=40)
        add_table_page(pdf, "Original Motif Reuse", compact_reuse_table(reuse), max_rows=40)
        for figure_name in FIGURES:
            if figure_name in figures:
                title = figure_name.replace(".png", "").replace("_", " ").title()
                add_image_page(pdf, title, figures[figure_name])
        findings = markdown_text.split("## Findings", 1)[1] if "## Findings" in markdown_text else ""
        add_text_page(pdf, "Findings and Interpretation", "## Findings" + findings, fontsize=9)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = read_json("imbalanced_bootstrap_config.json")
    tables = collect_tables()
    core, reuse, stringency_counts = summarize_results(tables)
    panel_cd = compact_panel_cd_table(core)
    panel_cd.to_csv(PANEL_CD_CSV, index=False)
    PANEL_CD_MD.write_text(
        "# Panels C and D: Stringency and AUC Summary\n\n"
        "Means and sample standard deviations (`ddof=1`) across 10 bootstrap replicates.\n\n"
        + dataframe_to_markdown(panel_cd, ".3f")
        + "\n",
        encoding="utf-8",
    )
    figures = copy_figures()
    markdown_text = generate_markdown(config, tables, core, reuse, stringency_counts, figures)
    REPORT_MD.write_text(markdown_text, encoding="utf-8")
    generate_pdf(markdown_text, core, reuse, figures)
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_PDF}")
    print(f"Wrote {PANEL_CD_CSV}")
    print(f"Wrote {PANEL_CD_MD}")


if __name__ == "__main__":
    main()
