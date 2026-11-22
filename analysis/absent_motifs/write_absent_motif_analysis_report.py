from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = _REPO_ROOT
DEFAULT_BASE_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
)
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "absent_motif_analysis_report.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the final two-tailed absent motif analysis report.")
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
    class_top = pd.read_csv(class_pairwise_dir / "top_interactions_all_sets.csv")
    nn_top = pd.read_csv(class_pairwise_dir / "NN_depleted_top_interactions.csv")
    np_top = pd.read_csv(class_pairwise_dir / "NP_depleted_top_interactions.csv")
    intersection_top = pd.read_csv(class_pairwise_dir / "NN_NP_intersection_depleted_top_interactions.csv")
    best_nn_top = top_interactions_from_matrix(best_pairwise_dir / "motif_induced_pairwise_NN.csv")
    best_np_top = top_interactions_from_matrix(best_pairwise_dir / "motif_induced_pairwise_NP.csv")
    optimal_motifs = pd.read_csv(best_pairwise_dir / "optimal_stringency_discriminative_motifs.csv")
    tuned_eight = pd.read_csv(args.base_dir / "tuned_stringency_8_discriminative_motifs.csv")

    lines = [
        "# Underrepresented Motif Analysis Report: Melanoma NN/NP",
        "",
        "## Scope",
        "",
        "This report summarizes the two-tailed underrepresented motif analysis for the melanoma NN/NP cohort.",
        "The analysis intentionally excludes cSCC and non-NN/NP melanoma classes.",
        "",
        "## Statistical Definition",
        "",
        "FANMOD Z-scores were converted to normal two-tailed p-values:",
        "",
        "`p_two_tailed = erfc(abs(Z) / sqrt(2))`",
        "",
        "A motif row was retained as underrepresented when `p_two_tailed < 0.05` and `z_score < 0`.",
        "For CISM feature values, observed frequency was replaced by the positive depletion signal:",
        "",
        "`depletion = Mean_Freq - observed_Freq`",
        "",
        "## Cohort and Motif Table",
        "",
        bullet(f"Labels: {', '.join(run_summary['summary']['labels'])}"),
        bullet(f"Patients: {run_summary['summary']['nn_np_patients']}"),
        bullet(f"FANMOD files scanned: {run_summary['summary']['fanmod_files_scanned']}"),
        bullet(f"FANMOD rows after label filter: {run_summary['summary']['rows_after_label_filter']:,}"),
        bullet(f"Underrepresented motif rows retained: {run_summary['summary']['absence_rows']:,}"),
        bullet(f"Unique underrepresented motifs retained: {run_summary['summary']['unique_absence_motifs']:,}"),
        "",
        "## Discriminative CISM on Two-Tailed Absence Motifs",
        "",
        bullet("Optuna tuned only `shared_percentage`; `max_class_features` was fixed at 30."),
        bullet(f"Best shared_percentage: {run_summary['summary']['feature_conf']['shared_percentage']:.6g}"),
        bullet(f"Best ROC AUC: {run_summary['metrics']['roc_auc']:.6g}"),
        bullet(f"Accuracy: {run_summary['metrics']['acc']:.6g}"),
        bullet(f"Precision: {run_summary['metrics']['precision']:.6g}"),
        bullet(f"Recall: {run_summary['metrics']['recall']:.6g}"),
        bullet(f"F1 score: {run_summary['metrics']['f1_score']:.6g}"),
        "",
        "At the tuned stringency, the full-cohort class-associated candidate set had "
        f"{int(stringency_summary['candidate_unique_subgraphs']):,} unique underrepresented subgraphs; "
        f"after the per-class cap it had {int(stringency_summary['capped_selected_unique_subgraphs']):,}.",
        "",
        "### Tuned-Stringency 8-Motif Set",
        "",
        "The initial full-cohort tuned-stringency selection contained 8 unique underrepresented discriminative motifs.",
        "The matrix below shows the cell-type composition of each motif. Brown indicates one node of that cell type; black indicates two nodes.",
        "",
        table(tuned_eight),
        "",
        figure(
            args.base_dir / "tuned_stringency_8_motif_cell_type_composition.png",
            "Cell-type composition of the 8 tuned-stringency discriminative underrepresented motifs",
        ),
        figure(
            args.base_dir / "tuned_stringency_8_motif_cell_type_fractions_panel_b.png",
            "Cell-type fractions in the 8 tuned-stringency discriminative underrepresented motifs",
        ),
        figure(
            args.base_dir / "tuned_stringency_8_motif_composition_panel_ab.png",
            "Panel A-B composition summary for the 8 depleted discriminative motifs",
        ),
        "",
        figure(args.base_dir / "absent_subgraphs_vs_stringency.png", "Unique underrepresented subgraphs versus stringency"),
        figure(args.base_dir / "absent_auc_vs_stringency.png", "LOOCV ROC AUC versus stringency"),
        "",
        "## Motif-Induced Pairwise Interactions: Stringency 0",
        "",
        "For the first absence motif set, CISM feature extraction was rerun with `shared_percentage = 0`.",
        "Motif edges were weighted by mean depletion frequency and normalized so each pairwise matrix sums to 1.",
        "",
        bullet(f"Selected unique motifs across LOOCV folds: {pairwise_summary['all_selected_unique_motifs']:,}"),
        bullet(f"NN-associated selected motifs: {pairwise_summary['selected_motifs_by_class']['NN']:,}"),
        bullet(f"NP-associated selected motifs: {pairwise_summary['selected_motifs_by_class']['NP']:,}"),
        bullet(f"Mean fold feature count: {pairwise_summary['mean_cfeatures']:.4g}"),
        "",
        figure(pairwise_dir / "motif_induced_pairwise_NN.png", "NN motif-induced pairwise interactions"),
        figure(pairwise_dir / "motif_induced_pairwise_NP.png", "NP motif-induced pairwise interactions"),
        figure(pairwise_dir / "motif_induced_pairwise_NN_minus_NP.png", "NN minus NP motif-induced pairwise interactions"),
        "",
        "## Motif-Induced Pairwise Interactions: Optimal Stringency",
        "",
        "The optimal-stringency pairwise analysis uses the union of discriminative motif IDs selected across all LOOCV folds.",
        "",
        bullet(f"Shared-percentage stringency: {best_pairwise_summary['shared_percentage']:.6g}"),
        bullet(f"Selected unique motifs across LOOCV folds: {best_pairwise_summary['all_selected_unique_motifs']:,}"),
        bullet(f"NN-associated selected motifs: {best_pairwise_summary['selected_motifs_by_class']['NN']:,}"),
        bullet(f"NP-associated selected motifs: {best_pairwise_summary['selected_motifs_by_class']['NP']:,}"),
        bullet(f"Mean fold feature count: {best_pairwise_summary['mean_cfeatures']:.4g}"),
        "",
        "### Discriminative Underrepresented Motif Subgraphs",
        "",
        "The panel below shows every unique motif selected across the optimal-stringency LOOCV folds.",
        "Each title gives the associated class and motif ID; blue indicates NN-associated depleted motifs and red indicates NP-associated depleted motifs.",
        "",
        figure(
            best_pairwise_dir / "optimal_stringency_discriminative_motif_subgraphs.png",
            "Optimal-stringency discriminative underrepresented motif subgraphs",
        ),
        "",
        "Discriminative motif legend:",
        "",
        table(optimal_motifs[["patient_class", "ID", "node_cell_types", "n_directed_edges"]]),
        "",
        figure(best_pairwise_dir / "motif_induced_pairwise_NN.png", "Optimal-stringency NN motif-induced pairwise interactions"),
        figure(best_pairwise_dir / "motif_induced_pairwise_NP.png", "Optimal-stringency NP motif-induced pairwise interactions"),
        figure(
            best_pairwise_dir / "motif_induced_pairwise_NN_minus_NP.png",
            "Optimal-stringency NN minus NP motif-induced pairwise interactions",
        ),
        "",
        "Top optimal-stringency NN motif-induced interactions:",
        "",
        table(best_nn_top.head(10)),
        "",
        "Top optimal-stringency NP motif-induced interactions:",
        "",
        table(best_np_top.head(10)),
        "",
        "## Motifs Depleted in at Least 90% of Class Patients",
        "",
        "This analysis ignores whether the motif is also absent in the other class when selecting the class-specific set.",
        "It then reports the other-class 90% status separately.",
        "",
        bullet(f"NN motifs depleted in at least 90% of NN patients: {class_summary['n_motifs_by_class']['NN']:,}"),
        bullet(f"NP motifs depleted in at least 90% of NP patients: {class_summary['n_motifs_by_class']['NP']:,}"),
        bullet(f"Total class-motif rows: {class_summary['n_report_rows']:,}"),
        "",
        "Report columns include within-class absence rate, depleted patient count, class size, mean/std observed motif appearances, "
        "mean/std observed frequency, mean/std depletion, and whether the motif is also absent in at least 90% of the other class.",
        "",
        "### Example High-Prevalence Motifs",
        "",
        "Top NN class rows by absence prevalence and observed appearances:",
        "",
        table(
            class_report[class_report["patient_class"] == "NN"].head(10)[
                [
                    "ID",
                    "absence_rate_within_class",
                    "observed_appearances_mean",
                    "observed_appearances_std",
                    "other_class_absence_rate",
                    "absent_in_other_class_90pct",
                ]
            ]
        ),
        "",
        "Top NP class rows by absence prevalence and observed appearances:",
        "",
        table(
            class_report[class_report["patient_class"] == "NP"].head(10)[
                [
                    "ID",
                    "absence_rate_within_class",
                    "observed_appearances_mean",
                    "observed_appearances_std",
                    "other_class_absence_rate",
                    "absent_in_other_class_90pct",
                ]
            ]
        ),
        "",
        "## Pairwise Heatmaps for 90% Class-Depleted Motif Sets",
        "",
        "Three motif sets were translated into weighted pairwise interaction maps:",
        "",
        bullet(f"NN-depleted motifs: {class_pairwise_summary['set_summary']['NN_depleted']['motif_count']:,}"),
        bullet(f"NP-depleted motifs: {class_pairwise_summary['set_summary']['NP_depleted']['motif_count']:,}"),
        bullet(
            "NN/NP intersection depleted motifs: "
            f"{class_pairwise_summary['set_summary']['NN_NP_intersection_depleted']['motif_count']:,}"
        ),
        "",
        figure(class_pairwise_dir / "NN_depleted_pairwise.png", "Pairwise map induced by NN-depleted motifs"),
        figure(class_pairwise_dir / "NP_depleted_pairwise.png", "Pairwise map induced by NP-depleted motifs"),
        figure(
            class_pairwise_dir / "NN_NP_intersection_depleted_pairwise.png",
            "Pairwise map induced by motifs depleted in both NN and NP",
        ),
        "",
        "The following subtraction maps remove the shared NN/NP depleted intersection from each class-specific map:",
        "",
        figure(
            class_pairwise_dir / "NN_depleted_minus_intersection_pairwise.png",
            "NN-depleted pairwise map minus shared depleted intersection",
        ),
        figure(
            class_pairwise_dir / "NP_depleted_minus_intersection_pairwise.png",
            "NP-depleted pairwise map minus shared depleted intersection",
        ),
        "",
        "### Top Pairwise Interactions",
        "",
        "NN-depleted motif set:",
        "",
        table(nn_top.head(10)),
        "",
        "NP-depleted motif set:",
        "",
        table(np_top.head(10)),
        "",
        "NN/NP depleted intersection:",
        "",
        table(intersection_top.head(10)),
        "",
        "## Output Files",
        "",
        bullet(f"Class 90% motif report: `{class_report_dir / 'class_absent_motifs_90pct.csv'}`"),
        bullet(f"Class-depleted pairwise directory: `{class_pairwise_dir}`"),
        bullet(f"Stringency-0 motif-induced pairwise directory: `{pairwise_dir}`"),
        bullet(f"Optimal-stringency motif-induced pairwise directory: `{best_pairwise_dir}`"),
        bullet(f"Tuned-stringency 8-motif composition matrix: `{args.base_dir / 'tuned_stringency_8_motif_cell_type_composition.csv'}`"),
        bullet(f"Tuned-stringency 8-motif cell-type fractions: `{args.base_dir / 'tuned_stringency_8_motif_cell_type_fractions.csv'}`"),
        bullet(f"Two-tailed absence CISM artifact: `{args.base_dir / 'Melanoma_NN_NP_absent_cism_ready.pkl'}`"),
        "",
    ]
    args.output.write_text("\n".join(lines))
    print(args.output)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def bullet(text: str) -> str:
    return f"- {text}"


def figure(path: Path, caption: str) -> str:
    return f"![{caption}]({path})\n\n*{caption}.*"


def table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.4g}")
    formatted = formatted.astype(str)
    columns = list(formatted.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in formatted.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def top_interactions_from_matrix(path: Path, top_n: int = 25) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    stacked = matrix.stack().reset_index()
    stacked.columns = ["source_cell_type", "target_cell_type", "weight"]
    return stacked.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    main()
