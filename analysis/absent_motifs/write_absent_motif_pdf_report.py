from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = _REPO_ROOT
DEFAULT_BASE_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
)
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "absent_motif_analysis_report.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the absent motif analysis report as a PDF.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pairwise_dir = args.base_dir / "motif_induced_pairwise_stringency0"
    best_pairwise_dir = args.base_dir / "motif_induced_pairwise_best_stringency"
    class_report_dir = args.base_dir / "class_absent_motifs_90pct"
    class_pairwise_dir = args.base_dir / "class_depleted_pairwise_heatmaps"

    run_summary = read_json(args.base_dir / "absence_run_summary.json")
    stringency_summary = read_json(args.base_dir / "absent_stringency_0364_subgraph_count.json")
    pairwise_summary = read_json(pairwise_dir / "motif_induced_pairwise_summary.json")
    best_pairwise_summary = read_json(best_pairwise_dir / "motif_induced_pairwise_summary.json")
    class_summary = read_json(class_report_dir / "class_absent_motifs_90pct_summary.json")
    class_pairwise_summary = read_json(class_pairwise_dir / "class_depleted_pairwise_summary.json")

    class_report = pd.read_csv(class_report_dir / "class_absent_motifs_90pct.csv")
    nn_top = pd.read_csv(class_pairwise_dir / "NN_depleted_top_interactions.csv")
    np_top = pd.read_csv(class_pairwise_dir / "NP_depleted_top_interactions.csv")
    intersection_top = pd.read_csv(class_pairwise_dir / "NN_NP_intersection_depleted_top_interactions.csv")
    best_nn_top = top_interactions_from_matrix(best_pairwise_dir / "motif_induced_pairwise_NN.csv")
    best_np_top = top_interactions_from_matrix(best_pairwise_dir / "motif_induced_pairwise_NP.csv")
    tuned_eight = pd.read_csv(args.base_dir / "tuned_stringency_8_discriminative_motifs.csv")

    with PdfPages(args.output) as pdf:
        add_text_page(
            pdf,
            "Underrepresented Motif Analysis: Melanoma NN/NP",
            [
                "Scope",
                "This report summarizes the two-tailed underrepresented motif analysis for the melanoma NN/NP cohort.",
                "The analysis excludes cSCC and non-NN/NP melanoma classes.",
                "",
                "Statistical definition",
                "FANMOD Z-scores were converted to normal two-tailed p-values:",
                "p_two_tailed = erfc(abs(Z) / sqrt(2))",
                "Rows were retained when p_two_tailed < 0.05 and z_score < 0.",
                "CISM feature value: depletion = Mean_Freq - observed_Freq.",
                "",
                "Cohort and motif table",
                f"Labels: {', '.join(run_summary['summary']['labels'])}",
                f"Patients: {run_summary['summary']['nn_np_patients']}",
                f"FANMOD files scanned: {run_summary['summary']['fanmod_files_scanned']}",
                f"FANMOD rows after label filter: {run_summary['summary']['rows_after_label_filter']:,}",
                f"Underrepresented motif rows retained: {run_summary['summary']['absence_rows']:,}",
                f"Unique underrepresented motifs retained: {run_summary['summary']['unique_absence_motifs']:,}",
            ],
        )
        add_text_page(
            pdf,
            "Discriminative CISM Results",
            [
                "Optuna tuned shared_percentage from scratch while max_class_features was fixed at 30.",
                f"Best shared_percentage: {run_summary['summary']['feature_conf']['shared_percentage']:.6g}",
                f"ROC AUC: {run_summary['metrics']['roc_auc']:.6g}",
                f"Accuracy: {run_summary['metrics']['acc']:.6g}",
                f"Precision: {run_summary['metrics']['precision']:.6g}",
                f"Recall: {run_summary['metrics']['recall']:.6g}",
                f"F1 score: {run_summary['metrics']['f1_score']:.6g}",
                "",
                f"At tuned stringency: {int(stringency_summary['candidate_unique_subgraphs']):,} unique underrepresented subgraphs.",
                f"After max_class_features cap: {int(stringency_summary['capped_selected_unique_subgraphs']):,} unique subgraphs.",
            ],
        )
        add_table_page(pdf, "Tuned-Stringency 8 Discriminative Motifs", format_tuned_eight_rows(tuned_eight))
        add_landscape_image_page(
            pdf,
            args.base_dir / "tuned_stringency_8_motif_cell_type_composition.png",
            "Cell-type composition of the 8 tuned-stringency underrepresented motifs",
        )
        add_landscape_image_page(
            pdf,
            args.base_dir / "tuned_stringency_8_motif_cell_type_fractions_panel_b.png",
            "Cell-type fractions in the 8 tuned-stringency underrepresented motifs",
        )
        add_landscape_image_page(
            pdf,
            args.base_dir / "tuned_stringency_8_motif_composition_panel_ab.png",
            "Panel A-B composition summary for the 8 depleted discriminative motifs",
        )
        add_image_page(pdf, args.base_dir / "absent_subgraphs_vs_stringency.png", "Unique underrepresented subgraphs versus stringency")
        add_image_page(pdf, args.base_dir / "absent_auc_vs_stringency.png", "LOOCV ROC AUC versus stringency")

        add_text_page(
            pdf,
            "Motif-Induced Pairwise Interactions: Stringency 0",
            [
                "CISM feature extraction was rerun with shared_percentage = 0.",
                "Motif edges were weighted by mean depletion frequency and each matrix was normalized to sum to 1.",
                f"Selected unique motifs across LOOCV folds: {pairwise_summary['all_selected_unique_motifs']:,}",
                f"NN-associated selected motifs: {pairwise_summary['selected_motifs_by_class']['NN']:,}",
                f"NP-associated selected motifs: {pairwise_summary['selected_motifs_by_class']['NP']:,}",
                f"Mean fold feature count: {pairwise_summary['mean_cfeatures']:.4g}",
            ],
        )
        add_image_page(pdf, pairwise_dir / "motif_induced_pairwise_NN.png", "NN motif-induced pairwise interactions")
        add_image_page(pdf, pairwise_dir / "motif_induced_pairwise_NP.png", "NP motif-induced pairwise interactions")
        add_image_page(pdf, pairwise_dir / "motif_induced_pairwise_NN_minus_NP.png", "NN minus NP motif-induced pairwise interactions")

        add_text_page(
            pdf,
            "Motif-Induced Pairwise Interactions: Optimal Stringency",
            [
                "This section uses the union of discriminative motif IDs selected across all LOOCV folds.",
                f"Shared-percentage stringency: {best_pairwise_summary['shared_percentage']:.6g}",
                f"Selected unique motifs across LOOCV folds: {best_pairwise_summary['all_selected_unique_motifs']:,}",
                f"NN-associated selected motifs: {best_pairwise_summary['selected_motifs_by_class']['NN']:,}",
                f"NP-associated selected motifs: {best_pairwise_summary['selected_motifs_by_class']['NP']:,}",
                f"Mean fold feature count: {best_pairwise_summary['mean_cfeatures']:.4g}",
                "",
                "The next page visualizes every selected discriminative underrepresented motif.",
                "Blue titles indicate NN-associated depleted motifs; red titles indicate NP-associated depleted motifs.",
                f"Full motif legend CSV: {best_pairwise_dir / 'optimal_stringency_discriminative_motifs.csv'}",
            ],
        )
        add_landscape_image_page(
            pdf,
            best_pairwise_dir / "optimal_stringency_discriminative_motif_subgraphs.png",
            "Discriminative underrepresented motif subgraphs at optimal stringency",
        )
        add_image_page(pdf, best_pairwise_dir / "motif_induced_pairwise_NN.png", "Optimal-stringency NN motif-induced pairwise interactions")
        add_image_page(pdf, best_pairwise_dir / "motif_induced_pairwise_NP.png", "Optimal-stringency NP motif-induced pairwise interactions")
        add_image_page(pdf, best_pairwise_dir / "motif_induced_pairwise_NN_minus_NP.png", "Optimal-stringency NN minus NP motif-induced pairwise interactions")
        add_table_page(pdf, "Top Optimal-Stringency Pairwise Interactions: NN", format_interaction_rows(best_nn_top.head(12)))
        add_table_page(pdf, "Top Optimal-Stringency Pairwise Interactions: NP", format_interaction_rows(best_np_top.head(12)))

        add_text_page(
            pdf,
            "Motifs Depleted in at Least 90% of Class Patients",
            [
                "Motifs were selected independently per class, ignoring whether they are also absent in the other class.",
                f"NN motifs depleted in at least 90% of NN patients: {class_summary['n_motifs_by_class']['NN']:,}",
                f"NP motifs depleted in at least 90% of NP patients: {class_summary['n_motifs_by_class']['NP']:,}",
                f"Total class-motif rows: {class_summary['n_report_rows']:,}",
                "",
                "The tabular report includes within-class absence rate, observed appearance mean/std,",
                "and whether the motif is also absent in at least 90% of the other class.",
            ],
        )
        add_table_page(
            pdf,
            "Top NN 90%+ Depleted Motifs",
            format_motif_rows(class_report[class_report["patient_class"] == "NN"].head(10)),
        )
        add_table_page(
            pdf,
            "Top NP 90%+ Depleted Motifs",
            format_motif_rows(class_report[class_report["patient_class"] == "NP"].head(10)),
        )

        add_text_page(
            pdf,
            "Pairwise Heatmaps for 90% Class-Depleted Motif Sets",
            [
                f"NN-depleted motifs: {class_pairwise_summary['set_summary']['NN_depleted']['motif_count']:,}",
                f"NP-depleted motifs: {class_pairwise_summary['set_summary']['NP_depleted']['motif_count']:,}",
                "NN/NP intersection depleted motifs: "
                f"{class_pairwise_summary['set_summary']['NN_NP_intersection_depleted']['motif_count']:,}",
                "",
                "Subtraction maps remove the shared NN/NP depleted intersection from each class-specific map.",
                "All matrices are weighted by mean depletion frequency and normalized before subtraction.",
            ],
        )
        add_image_page(pdf, class_pairwise_dir / "NN_depleted_pairwise.png", "Pairwise map induced by NN-depleted motifs")
        add_image_page(pdf, class_pairwise_dir / "NP_depleted_pairwise.png", "Pairwise map induced by NP-depleted motifs")
        add_image_page(pdf, class_pairwise_dir / "NN_NP_intersection_depleted_pairwise.png", "Pairwise map induced by shared depleted motifs")
        add_image_page(pdf, class_pairwise_dir / "NN_depleted_minus_intersection_pairwise.png", "NN-depleted map minus shared depleted intersection")
        add_image_page(pdf, class_pairwise_dir / "NP_depleted_minus_intersection_pairwise.png", "NP-depleted map minus shared depleted intersection")

        add_table_page(pdf, "Top Pairwise Interactions: NN-Depleted Set", format_interaction_rows(nn_top.head(12)))
        add_table_page(pdf, "Top Pairwise Interactions: NP-Depleted Set", format_interaction_rows(np_top.head(12)))
        add_table_page(pdf, "Top Pairwise Interactions: Shared Depleted Intersection", format_interaction_rows(intersection_top.head(12)))

        add_text_page(
            pdf,
            "Primary Output Locations",
            [
                f"Markdown report: {args.base_dir / 'absent_motif_analysis_report.md'}",
                f"PDF report: {args.output}",
                f"Two-tailed absent CISM object: {args.base_dir / 'Melanoma_NN_NP_absent_cism_ready.pkl'}",
                f"90% class-depleted motif report: {class_report_dir / 'class_absent_motifs_90pct.csv'}",
                f"Class-depleted pairwise heatmaps: {class_pairwise_dir}",
                f"Stringency-0 motif-induced pairwise outputs: {pairwise_dir}",
                f"Optimal-stringency motif-induced pairwise outputs: {best_pairwise_dir}",
                f"Tuned-stringency 8-motif composition matrix: {args.base_dir / 'tuned_stringency_8_motif_cell_type_composition.csv'}",
                f"Tuned-stringency 8-motif cell-type fractions: {args.base_dir / 'tuned_stringency_8_motif_cell_type_fractions.csv'}",
            ],
        )

    print(args.output)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top")
    y = 0.88
    for line in lines:
        if not line:
            y -= 0.025
            continue
        wrapped = textwrap.wrap(line, width=92) or [""]
        for part in wrapped:
            fig.text(0.08, y, part, fontsize=10.5, va="top")
            y -= 0.022
        if y < 0.08:
            break
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.96, title, fontsize=16, weight="bold", va="top")
    image = mpimg.imread(image_path)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_landscape_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.965, title, fontsize=15, weight="bold", va="top")
    image = mpimg.imread(image_path)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, rows: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.94, title, fontsize=16, weight="bold", va="top")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.82])
    ax.axis("off")
    table = ax.table(cellText=rows.values, colLabels=rows.columns, loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def format_motif_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ID",
        "absence_rate_within_class",
        "observed_appearances_mean",
        "observed_appearances_std",
        "other_class_absence_rate",
        "absent_in_other_class_90pct",
    ]
    out = df[cols].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: f"{value:.4g}")
    return out.astype(str)


def format_tuned_eight_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["patient_class", "ID", "patient_count", "patient_percentage"]].copy()
    out["patient_percentage"] = out["patient_percentage"].map(lambda value: f"{value:.4g}")
    return out.astype(str)


def format_interaction_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["source_cell_type", "target_cell_type", "weight"]
    out = df[cols].copy()
    out["weight"] = out["weight"].map(lambda value: f"{value:.4g}")
    return out.astype(str)


def top_interactions_from_matrix(path: Path, top_n: int = 25) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    stacked = matrix.stack().reset_index()
    stacked.columns = ["source_cell_type", "target_cell_type", "weight"]
    return stacked.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    main()
