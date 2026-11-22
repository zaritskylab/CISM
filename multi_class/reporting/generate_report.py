from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "multi_class" / "runtime"
REPORT_DIR = PROJECT_ROOT / "multi_class" / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
REPORT_MD = REPORT_DIR / "multi_class_melanoma_report.md"
REPORT_PDF = REPORT_DIR / "multi_class_melanoma_report.pdf"
PATIENT_CLASS_CSV = PROJECT_ROOT / "melanoma_tutorials" / "pet_data" / "Melanoma" / "patient_class.csv"

VARIANTS = ["all_classes_keep_tumor", "all_classes_remove_tumor", "without_pp_remove_tumor"]
METHODS = ["one_vs_all_union", "one_vs_one_union", "one_vs_all_positive_union"]
METHOD_LABELS = {
    "one_vs_all_union": "One-vs-all union",
    "one_vs_one_union": "One-vs-one union",
    "one_vs_all_positive_union": "One-vs-all positive-class union",
}


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def collect_outputs() -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, pd.DataFrame]], pd.DataFrame]:
    summary_rows = []
    tables: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for variant in VARIANTS:
        for method in METHODS:
            output_dir = RUNTIME_DIR / variant / method
            metrics = read_json(output_dir / "metrics.json")
            if metrics is None:
                continue
            key = (variant, method)
            row = {"variant": variant, "method": method, **metrics}
            summary_rows.append(row)
            tables[key] = {
                "tuning": read_csv(output_dir / "comparison_tuning.csv"),
                "confusion": read_csv(output_dir / "confusion_matrix.csv", index_col=0),
                "importance": read_csv(output_dir / "feature_importance.csv"),
                "selected": read_csv(output_dir / "selected_motifs.csv"),
                "predictions": read_csv(output_dir / "loocv_predictions.csv"),
            }
    summary = pd.DataFrame(summary_rows)
    patient_classes = pd.read_csv(PATIENT_CLASS_CSV, header=None, names=["patient_id", "patient_class"])
    return summary, tables, patient_classes


def status_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    completed = set(zip(summary.get("variant", []), summary.get("method", [])))
    for variant in VARIANTS:
        prep_summary = RUNTIME_DIR / variant / "artifact" / "preprocessing_summary.json"
        artifact_path = RUNTIME_DIR / variant / "artifact" / f"{variant}_cism_ready.pkl"
        prepared_dir = RUNTIME_DIR / variant / "prepared_datasets" / variant
        for method in METHODS:
            rows.append(
                {
                    "variant": variant,
                    "method": METHOD_LABELS[method],
                    "result_status": "completed" if (variant, method) in completed else "not completed",
                    "artifact": "available" if artifact_path.exists() else "not available",
                    "prepared_dataset": "available" if prepared_dir.exists() else "not needed / not available",
                    "preprocessing_summary": "available" if prep_summary.exists() else "not available",
                }
            )
    return pd.DataFrame(rows)


def create_figures(summary: pd.DataFrame, tables: dict[tuple[str, str], dict[str, pd.DataFrame]]) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    if summary.empty:
        return figures

    plot_summary = summary.copy()
    plot_summary["experiment"] = plot_summary["variant"] + "\n" + plot_summary["method"].map(METHOD_LABELS)

    metrics = ["accuracy", "balanced_accuracy", "roc_auc_ovr_macro", "f1_macro"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(plot_summary))
    width = 0.18
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, plot_summary[metric], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_summary["experiment"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Multi-class LOOCV performance")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figures["performance"] = FIGURE_DIR / "performance_comparison.png"
    fig.savefig(figures["performance"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plot_summary["experiment"], plot_summary["n_motifs"], color="#4C78A8")
    ax.set_ylabel("Selected motif count")
    ax.set_title("Final union panel size")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figures["motif_counts"] = FIGURE_DIR / "motif_counts.png"
    fig.savefig(figures["motif_counts"], dpi=220)
    plt.close(fig)

    auc_rows = []
    for _, row in summary.iterrows():
        for label in row["labels"]:
            auc_rows.append(
                {
                    "experiment": row["variant"] + "\n" + METHOD_LABELS[row["method"]],
                    "label": label,
                    "auc": row.get(f"roc_auc_ovr_{label}", np.nan),
                }
            )
    auc_df = pd.DataFrame(auc_rows)
    if not auc_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        experiments = auc_df["experiment"].unique()
        labels = auc_df["label"].unique()
        x = np.arange(len(labels))
        width = 0.8 / len(experiments)
        for i, experiment in enumerate(experiments):
            values = auc_df[auc_df["experiment"] == experiment].set_index("label").reindex(labels)["auc"]
            ax.bar(x + (i - (len(experiments) - 1) / 2) * width, values, width, label=experiment)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("One-vs-rest ROC-AUC")
        ax.set_title("Per-class ROC-AUC")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        figures["per_class_auc"] = FIGURE_DIR / "per_class_auc.png"
        fig.savefig(figures["per_class_auc"], dpi=220)
        plt.close(fig)

    for key, experiment_tables in tables.items():
        variant, method = key
        short = f"{variant}_{method}"
        confusion = experiment_tables["confusion"]
        if not confusion.empty:
            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            image = ax.imshow(confusion.to_numpy(), cmap="Blues")
            for i in range(confusion.shape[0]):
                for j in range(confusion.shape[1]):
                    ax.text(j, i, str(confusion.iloc[i, j]), ha="center", va="center", color="black")
            ax.set_xticks(range(confusion.shape[1]))
            ax.set_xticklabels([c.replace("pred_", "") for c in confusion.columns])
            ax.set_yticks(range(confusion.shape[0]))
            ax.set_yticklabels([i.replace("true_", "") for i in confusion.index])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion matrix\n{variant} / {METHOD_LABELS[method]}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            figures[f"confusion_{short}"] = FIGURE_DIR / f"confusion_{short}.png"
            fig.savefig(figures[f"confusion_{short}"], dpi=220)
            plt.close(fig)

        tuning = experiment_tables["tuning"]
        if not tuning.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].bar(tuning["comparison"], tuning["stringency"], color="#72B7B2")
            axes[0].set_ylim(0, 1)
            axes[0].set_ylabel("Selected stringency")
            axes[0].set_title("Best stringency per comparison")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].grid(axis="y", alpha=0.25)
            axes[1].bar(tuning["comparison"], tuning["roc_auc"], color="#F58518")
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("Binary LOOCV ROC-AUC")
            axes[1].set_title("Tuning objective at selected stringency")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].grid(axis="y", alpha=0.25)
            fig.suptitle(f"Tuning summary: {variant} / {METHOD_LABELS[method]}")
            fig.tight_layout()
            figures[f"tuning_{short}"] = FIGURE_DIR / f"tuning_{short}.png"
            fig.savefig(figures[f"tuning_{short}"], dpi=220)
            plt.close(fig)

        importance = experiment_tables["importance"]
        if not importance.empty:
            top = importance.head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(top["ID"].astype(str), top["mean_importance"], color="#54A24B")
            ax.set_xlabel("Mean RF impurity importance")
            ax.set_ylabel("Motif ID")
            ax.set_title(f"Top motif importances\n{variant} / {METHOD_LABELS[method]}")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            figures[f"importance_{short}"] = FIGURE_DIR / f"importance_{short}.png"
            fig.savefig(figures[f"importance_{short}"], dpi=220)
            plt.close(fig)
    return figures


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    table_df = df.copy()
    def stringify(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(map(str, value))
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    for column in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[column]):
            table_df[column] = table_df[column].map(lambda value: format(value, floatfmt) if pd.notna(value) else "")
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


def generate_markdown(
    summary: pd.DataFrame,
    tables: dict[tuple[str, str], dict[str, pd.DataFrame]],
    patient_classes: pd.DataFrame,
    figures: dict[str, Path],
) -> str:
    completed = len(summary)
    status = status_table(summary)
    cohort = patient_classes["patient_class"].value_counts().sort_index().reset_index()
    cohort.columns = ["class", "patients"]
    metric_cols = [
        "variant",
        "method",
        "n_patients",
        "n_motifs",
        "accuracy",
        "balanced_accuracy",
        "roc_auc_ovr_macro",
        "f1_macro",
        "f1_weighted",
    ]
    metrics_table = summary[metric_cols].copy() if not summary.empty else pd.DataFrame()
    if not metrics_table.empty:
        metrics_table["method"] = metrics_table["method"].map(METHOD_LABELS)

    lines = [
        "# Multi-Class Melanoma Discriminative Motif Analysis Report",
        "",
        f"Generated from saved outputs under `{RUNTIME_DIR}`.",
        "",
        "## Completion Status",
        "",
        f"Completed experiment outputs found: **{completed}**.",
        "",
        dataframe_to_markdown(status),
        "",
        "## Experiment Settings",
        "",
        "- Classes: NN, NP, PN, PP for `all_classes_keep_tumor` and `all_classes_remove_tumor`; NN, NP, PN for `without_pp_remove_tumor`.",
        "- Tumor handling: no tumor removal for `all_classes_keep_tumor`; tumor alpha-shape removal from PN and PP for `all_classes_remove_tumor`; tumor alpha-shape removal from PN for `without_pp_remove_tumor`.",
        "- Tumor removal method: built-in CISM graph preparation path with `exclude_cell_type='tumor'`, alpha shape parameter 0.01, buffer 0, max distance 100.",
        "- Motif extraction: union of tuned binary discriminative motif panels.",
        "- Selection methods: one-vs-all union, one-vs-one union, and one-vs-all positive-class union.",
        "- One-vs-all positive-class union keeps only motifs associated with the held-out positive class in each binary comparison.",
        "- Stringency selection: binary RF LOOCV ROC-AUC maximized per comparison; balanced accuracy and selected motif count used as tie-breakers.",
        "- Final classifier: patient-level RandomForestClassifier LOOCV using the final union motif-frequency matrix.",
        "",
        "## Cohort",
        "",
        dataframe_to_markdown(cohort),
        "",
        "## Results Summary",
        "",
        dataframe_to_markdown(metrics_table),
        "",
    ]

    for figure_key in ["performance", "motif_counts", "per_class_auc"]:
        if figure_key in figures:
            lines.extend(["", f"![{figure_key}]({figures[figure_key].relative_to(REPORT_DIR)})", ""])

    if not summary.empty:
        best = summary.sort_values("balanced_accuracy", ascending=False).iloc[0]
        other = summary.sort_values("balanced_accuracy", ascending=False).iloc[-1]
        missing = status[status["result_status"] != "completed"]
        missing_text = (
            "Incomplete experiment outputs at report time: "
            + "; ".join(f"{row.variant} / {row.method}" for row in missing.itertuples())
            + "."
            if not missing.empty
            else "All configured experiment outputs were present at report time."
        )
        without_pp = summary[summary["variant"] == "without_pp_remove_tumor"]
        without_pp_text = ""
        if not without_pp.empty:
            top_without_pp = without_pp.sort_values("balanced_accuracy", ascending=False).iloc[0]
            without_pp_text = (
                f"- The completed tumor-removed NN/NP/PN run (`{METHOD_LABELS[top_without_pp['method']]}`) reached "
                f"balanced accuracy {top_without_pp['balanced_accuracy']:.3f}, accuracy {top_without_pp['accuracy']:.3f}, "
                f"and macro OvR ROC-AUC {top_without_pp['roc_auc_ovr_macro']:.3f} using {int(top_without_pp['n_motifs'])} motifs."
            )
        comparison_sentence = (
            f"- Compared with `{other['variant']}` / `{METHOD_LABELS[other['method']]}`, the best run used "
            f"{int(best['n_motifs'])} motifs versus {int(other['n_motifs'])} and improved balanced accuracy by "
            f"{best['balanced_accuracy'] - other['balanced_accuracy']:.3f}."
            if len(summary) > 1
            else "- Only one completed run was available, so cross-experiment ranking is provisional."
        )
        lines.extend(
            [
                "## Insights",
                "",
                f"- The strongest completed run is `{best['variant']}` / `{METHOD_LABELS[best['method']]}` with balanced accuracy {best['balanced_accuracy']:.3f}, accuracy {best['accuracy']:.3f}, and macro OvR ROC-AUC {best['roc_auc_ovr_macro']:.3f}.",
                comparison_sentence,
                without_pp_text,
                "- NN remains the easiest class across completed outputs. In the completed NN/NP/PN tumor-removed run, the remaining errors are NP/NN swaps and PN predicted as NP.",
                f"- {missing_text}",
                "",
            ]
        )

    lines.extend(["## Per-Experiment Details", ""])
    for (variant, method), experiment_tables in tables.items():
        metrics = summary[(summary["variant"] == variant) & (summary["method"] == method)].iloc[0]
        lines.extend(
            [
                f"### {variant} / {METHOD_LABELS[method]}",
                "",
                f"- Patients: {int(metrics['n_patients'])}",
                f"- Final motifs: {int(metrics['n_motifs'])}",
                f"- Accuracy: {metrics['accuracy']:.3f}",
                f"- Balanced accuracy: {metrics['balanced_accuracy']:.3f}",
                f"- Macro OvR ROC-AUC: {metrics['roc_auc_ovr_macro']:.3f}",
                "",
                "Best stringency per binary comparison:",
                "",
            ]
        )
        tuning_cols = ["comparison", "stringency", "roc_auc", "accuracy", "balanced_accuracy", "mean_selected_motifs"]
        lines.extend([dataframe_to_markdown(experiment_tables["tuning"][tuning_cols]), ""])
        for prefix in ["confusion", "tuning", "importance"]:
            key = f"{prefix}_{variant}_{method}"
            if key in figures:
                lines.extend(["", f"![{key}]({figures[key].relative_to(REPORT_DIR)})", ""])
        importance = experiment_tables["importance"]
        if not importance.empty:
            top_cols = ["ID", "mean_importance", "comparison_provenance", "comparison_label_provenance"]
            lines.extend(["Top 10 motifs by mean RF importance:", "", dataframe_to_markdown(importance[top_cols].head(10)), ""])

    preprocessing_rows = []
    for variant in VARIANTS:
        prep = read_json(RUNTIME_DIR / variant / "artifact" / "preprocessing_summary.json")
        if prep:
            preprocessing_rows.append(prep)
    if preprocessing_rows:
        lines.extend(["## Preprocessing Summaries", "", dataframe_to_markdown(pd.DataFrame(preprocessing_rows)), ""])

    return "\n".join(lines) + "\n"


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top")
    wrapped = []
    for paragraph in body.split("\n"):
        if not paragraph.strip():
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(paragraph, width=92))
    fig.text(0.08, 0.88, "\n".join(wrapped[:42]), fontsize=10, va="top", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 24) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
    table_df = df.head(max_rows).copy()
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda value: f"{value:.3f}")
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
    ax.imshow(image)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf(
    summary: pd.DataFrame,
    tables: dict[tuple[str, str], dict[str, pd.DataFrame]],
    patient_classes: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    status = status_table(summary)
    cohort = patient_classes["patient_class"].value_counts().sort_index().reset_index()
    cohort.columns = ["class", "patients"]
    metrics_cols = [
        "variant",
        "method",
        "n_patients",
        "n_motifs",
        "accuracy",
        "balanced_accuracy",
        "roc_auc_ovr_macro",
        "f1_macro",
    ]
    metrics = summary[metrics_cols].copy() if not summary.empty else pd.DataFrame()
    if not metrics.empty:
        metrics["method"] = metrics["method"].map(METHOD_LABELS)

    with PdfPages(REPORT_PDF) as pdf:
        body = (
            "This PDF was generated from saved multi_class/runtime outputs. "
            "Only completed experiment outputs are interpreted; incomplete methods are listed in the status table.\n\n"
            "Key settings: RF LOOCV classifier, motif-frequency features, binary comparison stringency selected by ROC-AUC, "
            "and final multi-class evaluation by accuracy, balanced accuracy, macro F1, macro OvR ROC-AUC, confusion matrix, and RF feature importance."
        )
        add_text_page(pdf, "Multi-Class Melanoma Discriminative Motif Analysis", body)
        add_table_page(pdf, "Completion Status", status)
        add_table_page(pdf, "Cohort Counts", cohort)
        if not metrics.empty:
            add_table_page(pdf, "Results Summary", metrics)
        for key in ["performance", "motif_counts", "per_class_auc"]:
            if key in figures:
                add_image_page(pdf, key.replace("_", " ").title(), figures[key])
        for (variant, method), experiment_tables in tables.items():
            tuning_cols = ["comparison", "stringency", "roc_auc", "accuracy", "balanced_accuracy", "mean_selected_motifs"]
            add_table_page(pdf, f"Tuning: {variant} / {METHOD_LABELS[method]}", experiment_tables["tuning"][tuning_cols])
            for prefix in ["confusion", "tuning", "importance"]:
                key = f"{prefix}_{variant}_{method}"
                if key in figures:
                    add_image_page(pdf, f"{prefix.title()}: {variant} / {METHOD_LABELS[method]}", figures[key])
            importance = experiment_tables["importance"]
            if not importance.empty:
                add_table_page(
                    pdf,
                    f"Top Feature Importances: {variant} / {METHOD_LABELS[method]}",
                    importance[["ID", "mean_importance", "comparison_provenance", "comparison_label_provenance"]].head(12),
                    max_rows=12,
                )
        if not summary.empty:
            best = summary.sort_values("balanced_accuracy", ascending=False).iloc[0]
            without_pp = summary[summary["variant"] == "without_pp_remove_tumor"]
            without_pp_sentence = ""
            if not without_pp.empty:
                top_without_pp = without_pp.sort_values("balanced_accuracy", ascending=False).iloc[0]
                without_pp_sentence = (
                    f"The completed NN/NP/PN tumor-removed run was {METHOD_LABELS[top_without_pp['method']]} with "
                    f"balanced accuracy {top_without_pp['balanced_accuracy']:.3f}, accuracy {top_without_pp['accuracy']:.3f}, "
                    f"macro OvR ROC-AUC {top_without_pp['roc_auc_ovr_macro']:.3f}, and {int(top_without_pp['n_motifs'])} motifs.\n\n"
                )
            insight_body = (
                f"Best completed run: {best['variant']} / {METHOD_LABELS[best['method']]}; "
                f"balanced accuracy {best['balanced_accuracy']:.3f}, accuracy {best['accuracy']:.3f}, "
                f"macro OvR ROC-AUC {best['roc_auc_ovr_macro']:.3f}.\n\n"
                f"{without_pp_sentence}"
                "NN remains the easiest class across completed outputs. In the completed NN/NP/PN tumor-removed run, "
                "the remaining errors are NP/NN swaps and PN predicted as NP. Any missing methods are listed in the completion status table."
            )
            add_text_page(pdf, "Insights", insight_body)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary, tables, patient_classes = collect_outputs()
    figures = create_figures(summary, tables)
    markdown = generate_markdown(summary, tables, patient_classes, figures)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    generate_pdf(summary, tables, patient_classes, figures)
    print(f"Markdown report: {REPORT_MD}")
    print(f"PDF report: {REPORT_PDF}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
