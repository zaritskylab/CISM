#!/usr/bin/env python3
"""Run tutorial 03b soft motif selection as a script.

The script mirrors ``03b_soft_motif_selection.ipynb`` and saves tabular
artifacts plus all figures/intermediate visual summaries under
``melanoma_tutorials/runtime/soft_motif_selection``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_curve
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:
    sns = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism import (  # noqa: E402
    MotifSelectionWeights,
    SoftMotifSelectionConfig,
    StabilityGateConfig,
    evaluate_soft_motif_selection_loocv,
    optimize_soft_motif_weights,
    score_soft_motifs,
    score_soft_motifs_from_discriminator,
)
from cism.helpers import string_base64_pickle  # noqa: E402
from cism.motif_selection import evaluate_soft_motif_score_cutoff_sweep_loocv  # noqa: E402


if sns is not None:
    sns.set_theme(style="whitegrid")
else:
    plt.style.use("ggplot")


def parse_args() -> argparse.Namespace:
    default_runtime = SCRIPT_DIR / "melanoma_tutorials" / "runtime"
    parser = argparse.ArgumentParser(
        description="Run tutorial 03b soft motif selection and save outputs."
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=default_runtime / "serialized" / "Melanoma_cism_ready.pkl",
        help="Path to the serialized tutorial 02 artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_runtime / "soft_motif_selection",
        help="Directory where CSVs and images will be saved.",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=["NN", "NP"],
        help="The two labels to compare, in order.",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Number of motifs to select.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to add to the study.",
    )
    parser.add_argument(
        "--weight-low",
        type=float,
        default=0.0,
        help="Lower bound for Optuna weight search.",
    )
    parser.add_argument(
        "--weight-high",
        type=float,
        default=1.0,
        help="Upper bound for Optuna weight search.",
    )
    parser.add_argument(
        "--optuna-storage",
        type=Path,
        default=None,
        help="SQLite file used to persist and resume Optuna studies.",
    )
    parser.add_argument(
        "--optuna-study-name",
        default=None,
        help="Optuna study name. Defaults to a label/top-k-specific name.",
    )
    parser.add_argument(
        "--fresh-optuna-study",
        action="store_true",
        help="Fail if the named Optuna study already exists instead of resuming it.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument(
        "--optimize-gate",
        action="store_true",
        help="Let Optuna tune the stability gate tau and gamma in addition to additive weights.",
    )
    parser.add_argument(
        "--tau-range",
        nargs=2,
        type=float,
        default=(0.6, 1.0),
        metavar=("LOW", "HIGH"),
        help="Tau search range when --optimize-gate is used.",
    )
    parser.add_argument(
        "--gamma-range",
        nargs=2,
        type=float,
        default=(2.0, 10.0),
        metavar=("LOW", "HIGH"),
        help="Gamma search range when --optimize-gate is used.",
    )
    parser.add_argument("--fanmod-p-value-threshold", type=float, default=0.05)
    parser.add_argument(
        "--final-score-cutoff",
        type=float,
        default=None,
        help="Minimum final score required for motif selection in normal scoring/LOOCV.",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Use equal additive weights instead of running Optuna.",
    )
    parser.add_argument(
        "--skip-loocv",
        action="store_true",
        help="Skip leakage-safe LOOCV validation and dependent plots.",
    )
    parser.add_argument(
        "--run-cutoff-sweep",
        action="store_true",
        help="Run leakage-safe LOOCV final-score cutoff sensitivity analysis.",
    )
    parser.add_argument(
        "--cutoff-min",
        type=float,
        default=0.0,
        help="Minimum final-score cutoff for --run-cutoff-sweep.",
    )
    parser.add_argument(
        "--cutoff-max",
        type=float,
        default=0.8,
        help="Maximum final-score cutoff for --run-cutoff-sweep.",
    )
    parser.add_argument(
        "--cutoff-steps",
        type=int,
        default=25,
        help="Number of evenly spaced final-score cutoffs for --run-cutoff-sweep.",
    )
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip final LOOCV selection-stability scoring. Useful for quick script checks.",
    )
    parser.add_argument(
        "--max-motifs",
        type=int,
        default=None,
        help="Debug only: restrict scoring to the first N unique motif IDs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def save_table_image(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    max_rows: int = 20,
    max_cols: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_df = df.head(max_rows).iloc[:, :max_cols].copy()
    table_df = table_df.reset_index() if table_df.index.name or not isinstance(table_df.index, pd.RangeIndex) else table_df
    if table_df.empty:
        table_df = pd.DataFrame({"message": ["No rows"]})
    if hasattr(table_df, "map"):
        table_df = table_df.map(format_cell)
    else:
        table_df = table_df.applymap(format_cell)

    fig_width = max(8, min(24, 1.6 * len(table_df.columns)))
    fig_height = max(2.5, min(18, 0.45 * (len(table_df) + 2)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=13)
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 50 else f"{text[:47]}..."


def save_json(data: dict[str, Any], path: Path) -> None:
    serializable = {
        key: value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else str(value)
        for key, value in data.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved image: {path}")


def plot_bar(
    df: pd.DataFrame,
    *,
    x: str | None = None,
    y: str,
    hue: str | None = None,
    color: str = "#4C78A8",
    horizontal: bool = False,
) -> None:
    if sns is not None:
        if horizontal:
            sns.barplot(data=df, y=x, x=y, hue=hue, color=None if hue else color)
        else:
            sns.barplot(data=df, x=x, y=y, hue=hue, color=None if hue else color)
        return

    ax = plt.gca()
    if hue is None:
        labels = df[x].astype(str) if x is not None else df.index.astype(str)
        values = df[y].astype(float)
        positions = np.arange(len(df))
        if horizontal:
            ax.barh(positions, values, color=color)
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
        else:
            ax.bar(positions, values, color=color)
            ax.set_xticks(positions, labels)
        return

    pivot = df.pivot(index=x, columns=hue, values=y)
    pivot.plot(kind="bar", ax=ax)


def plot_heatmap(df: pd.DataFrame, *, title: str) -> None:
    if sns is not None:
        sns.heatmap(df, annot=True, fmt=".0f", cmap="Blues", cbar=False)
        plt.title(title)
        return

    ax = plt.gca()
    data = df.to_numpy(dtype=float)
    image = ax.imshow(data, cmap="Blues")
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            ax.text(col, row, f"{data[row, col]:.0f}", ha="center", va="center", color="black")
    ax.set_xticks(np.arange(len(df.columns)), df.columns)
    ax.set_yticks(np.arange(len(df.index)), df.index)
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def draw_motif_to_file(motif: Any, cells_type: dict[int, str] | list[str], path: Path, title: str) -> None:
    if isinstance(motif, str):
        motif = string_base64_pickle(motif)

    labels = {}
    node_types = nx.get_node_attributes(motif, name="type")
    for node, cell_type_id in node_types.items():
        labels[node] = cells_type[int(cell_type_id)]

    graph = motif.to_undirected()
    pos = nx.spring_layout(graph, seed=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(
        graph,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_size=4000,
        node_color="#BFD7EA",
        edge_color="#4A4A4A",
        font_color="black",
        font_size=9,
        ax=ax,
    )
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {path}")


def extract_positive_probabilities(results: pd.DataFrame, positive_label: str) -> tuple[list[str], list[float]]:
    true_labels = results["class"].tolist()
    probabilities = []
    for _, row in results.iterrows():
        classes = np.asarray(row["classes"])
        prob = np.asarray(row["prob"])
        positive_index = np.where(classes == positive_label)[0][0]
        probabilities.append(float(prob[0][positive_index]))
    return true_labels, probabilities


def shap_importance_from_loocv(results: pd.DataFrame) -> pd.DataFrame | None:
    rows = []
    for _, row in results.iterrows():
        shap_values, instance = row["shape_values"]
        if shap_values is None:
            return None
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = np.abs(values).mean(axis=(0, 2))
        elif values.ndim == 2:
            values = np.abs(values).mean(axis=0)
        else:
            values = np.abs(values).reshape(-1)
        rows.append(pd.Series(values, index=instance.columns, dtype=float))

    if not rows:
        return None
    importance = pd.DataFrame(rows).fillna(0.0).mean(axis=0).sort_values(ascending=False)
    return importance.rename("mean_abs_shap").reset_index().rename(columns={"index": "ID"})


def selection_frequency_from_loocv(selected_by_patient: dict[str, list[Any]]) -> pd.DataFrame:
    counts: dict[Any, int] = {}
    for selected_ids in selected_by_patient.values():
        for motif_id in selected_ids:
            counts[motif_id] = counts.get(motif_id, 0) + 1
    n_folds = max(1, len(selected_by_patient))
    frequency_df = (
        pd.Series(counts, name="selected_fold_count")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "ID"})
    )
    frequency_df["selection_frequency"] = frequency_df["selected_fold_count"] / n_folds
    return frequency_df



def save_cutoff_sweep_plot(sweep_df: pd.DataFrame, path: Path) -> None:
    fig, ax_auc = plt.subplots(figsize=(9, 5))
    ax_count = ax_auc.twinx()

    ax_auc.plot(
        sweep_df["final_score_cutoff"],
        sweep_df["roc_auc"],
        marker="o",
        color="#4C78A8",
        label="LOOCV ROC AUC",
    )
    ax_count.plot(
        sweep_df["final_score_cutoff"],
        sweep_df["mean_selected_motifs"],
        marker="s",
        color="#F58518",
        label="Mean selected motifs",
    )

    ax_auc.set_xlabel("Final score cutoff")
    ax_auc.set_ylabel("LOOCV ROC AUC", color="#4C78A8")
    ax_count.set_ylabel("Mean selected motifs", color="#F58518")
    ax_auc.tick_params(axis="y", labelcolor="#4C78A8")
    ax_count.tick_params(axis="y", labelcolor="#F58518")
    ax_auc.set_ylim(0, 1)
    ax_auc.set_title("Soft motif score cutoff sensitivity")

    lines = ax_auc.get_lines() + ax_count.get_lines()
    labels = [line.get_label() for line in lines]
    ax_auc.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Serialized artifact: {args.artifact_path.resolve()}")
    print(f"Output directory: {output_dir}")
    print(f"Image directory: {image_dir}")

    with args.artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)

    cism = artifact["cism"]
    discriminator = artifact["discriminator"]
    common_cells_type = artifact.get("common_cells_type")

    if args.max_motifs is not None:
        if args.max_motifs <= 0:
            raise ValueError("--max-motifs must be positive when provided.")
        keep_ids = cism.motifs_dataset["ID"].drop_duplicates().head(args.max_motifs)
        cism.motifs_dataset = cism.motifs_dataset[cism.motifs_dataset["ID"].isin(keep_ids)].copy()
        print(f"Debug mode: restricted motif table to {len(keep_ids)} unique motif IDs.")

    labels = list(args.labels)
    patients_df = discriminator.get_patients_class(labels)
    print(f"Labels: {labels}")
    print(f"Patients with selected labels: {patients_df.shape[0]}")
    print(f"Motif rows: {cism.motifs_dataset.shape[0]:,}")
    print(f"Number of cell types: {len(common_cells_type)}")

    patients_df.to_csv(output_dir / "selected_label_patients.csv")
    save_table_image(
        patients_df.head(),
        image_dir / "selected_label_patients_head.png",
        title="Selected label patients",
    )

    base_selection_config = SoftMotifSelectionConfig(
        labels=labels,
        top_k=args.top_k,
        weights=MotifSelectionWeights(),
        gate=StabilityGateConfig(tau=args.tau, gamma=args.gamma),
        fanmod_p_value_threshold=args.fanmod_p_value_threshold,
        final_score_cutoff=args.final_score_cutoff,
    )

    weight_tuning_result = None
    if args.skip_optimization:
        selection_config = base_selection_config
        best_weights = base_selection_config.weights
        print("Skipping Optuna. Using equal additive weights.")
    else:
        optuna_storage_path = (args.optuna_storage or output_dir / "optuna_soft_motif_weights.db").resolve()
        optuna_storage_path.parent.mkdir(parents=True, exist_ok=True)
        optuna_storage = f"sqlite:///{optuna_storage_path}"
        optuna_study_name = args.optuna_study_name or (
            f"soft_motif_weights_{'_'.join(labels)}_top{args.top_k}"
            f"{'_gate' if args.optimize_gate else ''}"
        )
        print(f"Optuna storage: {optuna_storage_path}")
        print(f"Optuna study: {optuna_study_name}")
        print(f"Adding Optuna trials: {args.n_trials}")

        weight_tuning_result = optimize_soft_motif_weights(
            discriminator=discriminator,
            base_config=base_selection_config,
            n_trials=args.n_trials,
            metric="roc_auc",
            random_state=args.random_state,
            weight_range=(args.weight_low, args.weight_high),
            show_progress_bar=not args.no_progress,
            storage=optuna_storage,
            study_name=optuna_study_name,
            load_if_exists=not args.fresh_optuna_study,
            optimize_gate=args.optimize_gate,
            tau_range=tuple(args.tau_range),
            gamma_range=tuple(args.gamma_range),
        )
        selection_config = weight_tuning_result.best_config
        best_weights = weight_tuning_result.best_weights
        print(f"Best Optuna ROC AUC: {weight_tuning_result.best_score:.3f}")
        print(best_weights)
        print(selection_config.gate)

        trials_df = weight_tuning_result.trials_dataframe()
        trials_df.to_csv(output_dir / "soft_motif_weight_optuna_trials.csv", index=False)
        save_table_image(
            trials_df.sort_values("value", ascending=False).head(10),
            image_dir / "soft_motif_weight_optuna_trials_top10.png",
            title="Top Optuna trials",
            max_cols=12,
        )

    weight_df = (
        pd.Series(best_weights.as_dict(), name="weight")
        .reset_index()
        .rename(columns={"index": "component"})
    )
    weight_df.to_csv(output_dir / "soft_motif_selected_weights.csv", index=False)
    pd.Series(
        {
            "tau": selection_config.gate.tau,
            "gamma": selection_config.gate.gamma,
        },
        name="value",
    ).to_csv(output_dir / "soft_motif_selected_gate.csv")

    plt.figure(figsize=(7, 4))
    plot_bar(weight_df, x="component", y="weight", color="#4C78A8")
    plt.title("Optuna-selected additive score weights" if not args.skip_optimization else "Equal additive score weights")
    plt.xlabel("")
    plt.ylabel("Weight")
    plt.xticks(rotation=30, ha="right")
    save_current_figure(image_dir / "soft_motif_selected_weights.png")

    save_json(
        {
            "labels": labels,
            "top_k": args.top_k,
            "weights": best_weights.as_dict(),
            "tau": args.tau,
            "gamma": args.gamma,
            "optimized_tau": selection_config.gate.tau,
            "optimized_gamma": selection_config.gate.gamma,
            "optimize_gate": args.optimize_gate,
            "tau_range": list(args.tau_range),
            "gamma_range": list(args.gamma_range),
            "fanmod_p_value_threshold": args.fanmod_p_value_threshold,
            "final_score_cutoff": args.final_score_cutoff,
            "random_state": args.random_state,
            "n_trials": 0 if args.skip_optimization else args.n_trials,
            "run_cutoff_sweep": args.run_cutoff_sweep,
            "cutoff_min": args.cutoff_min,
            "cutoff_max": args.cutoff_max,
            "cutoff_steps": args.cutoff_steps,
        },
        output_dir / "soft_motif_selection_config.json",
    )

    if args.skip_stability:
        print("Skipping final LOOCV stability scoring. Stability values will be set to 1.0.")
        soft_result = score_soft_motifs(
            motifs_df=discriminator.cism.motifs_dataset,
            patient_class_df=patients_df,
            config=selection_config,
            stability=None,
        )
    else:
        soft_result = score_soft_motifs_from_discriminator(
            discriminator=discriminator,
            config=selection_config,
        )
    scored_motifs_df = soft_result.scores
    selected_motif_ids = soft_result.selected_motif_ids

    scored_motifs_df.to_csv(output_dir / "soft_motif_scores.csv", index=False)
    pd.Series(selected_motif_ids, name="ID").to_csv(
        output_dir / "selected_soft_motif_ids.csv",
        index=False,
    )
    save_table_image(
        scored_motifs_df.head(20),
        image_dir / "soft_motif_scores_top20.png",
        title="Top soft motif scores",
        max_cols=10,
    )

    plot_columns = [
        "effect_score",
        "abundance_score",
        "prevalence_score",
        "confidence_score",
        "dispersion_desirability_score",
        "stability_gate",
        "final_score",
    ]
    top_plot_df = scored_motifs_df.head(10).melt(
        id_vars="ID",
        value_vars=plot_columns,
        var_name="component",
        value_name="score",
    )
    plt.figure(figsize=(13, 5))
    plot_bar(top_plot_df, x="ID", y="score", hue="component")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("Top soft motif score components")
    save_current_figure(image_dir / "top_soft_motif_score_components.png")

    top3_motifs = scored_motifs_df.head(3).copy()
    top3_columns = [
        "ID",
        "final_score",
        "additive_score",
        "stability_gate",
        "effect_score",
        "overall_prevalence",
        "direction",
        "logfc_group1_vs_group0",
    ]
    top3_motifs[top3_columns].to_csv(output_dir / "top3_soft_motifs.csv", index=False)
    save_table_image(
        top3_motifs[top3_columns],
        image_dir / "top3_soft_motifs.png",
        title="Top 3 soft motifs",
        max_rows=3,
        max_cols=len(top3_columns),
    )

    motif_lookup = (
        discriminator.cism.motifs_dataset.drop_duplicates("ID").set_index("ID")["motif"]
    )
    for _, motif_row in tqdm(
        top3_motifs.iterrows(),
        total=top3_motifs.shape[0],
        desc="Drawing top motifs",
        disable=args.no_progress,
    ):
        motif_id = motif_row["ID"]
        motif_path = image_dir / f"top_motif_{motif_id}.png"
        title = (
            f"Motif {motif_id}: final={motif_row['final_score']:.3f}, "
            f"effect={motif_row['effect_score']:.3f}, direction={motif_row['direction']}"
        )
        draw_motif_to_file(motif_lookup.loc[motif_id], common_cells_type, motif_path, title)

    soft_inference_feature_conf = soft_result.to_inference_feature_config()
    save_json(
        {
            "labels": soft_inference_feature_conf.labels,
            "motifs_ids": list(soft_inference_feature_conf.motifs_ids),
        },
        output_dir / "soft_inference_feature_config.json",
    )

    if args.run_cutoff_sweep:
        if args.cutoff_steps <= 1:
            raise ValueError("--cutoff-steps must be greater than 1.")
        if args.cutoff_min < 0 or args.cutoff_max < args.cutoff_min:
            raise ValueError("--cutoff-min and --cutoff-max must define a non-negative increasing range.")

        cutoffs = np.linspace(args.cutoff_min, args.cutoff_max, args.cutoff_steps)
        cutoff_sweep_df = evaluate_soft_motif_score_cutoff_sweep_loocv(
            discriminator=discriminator,
            config=selection_config,
            score_cutoffs=cutoffs,
            random_state=args.random_state,
            show_progress=not args.no_progress,
        )
        cutoff_sweep_path = output_dir / "soft_motif_score_cutoff_sweep.csv"
        cutoff_sweep_df.to_csv(cutoff_sweep_path, index=False)
        print(f"Saved cutoff sweep: {cutoff_sweep_path}")

        save_table_image(
            cutoff_sweep_df,
            image_dir / "soft_motif_score_cutoff_sweep.png",
            title="Soft motif score cutoff sensitivity",
            max_rows=args.cutoff_steps,
            max_cols=10,
        )
        save_cutoff_sweep_plot(
            cutoff_sweep_df,
            image_dir / "soft_motif_score_cutoff_sensitivity.png",
        )

    if args.skip_loocv:
        print("Skipping LOOCV validation.")
        return

    loocv_result = evaluate_soft_motif_selection_loocv(
        discriminator=discriminator,
        config=selection_config,
        random_state=args.random_state,
        show_progress=not args.no_progress,
    )
    loocv_results = loocv_result.results
    loocv_results.to_pickle(output_dir / "soft_motif_loocv_results.pkl")
    loocv_results.drop(columns=["prob", "classes", "contributions", "shape_values"]).to_csv(
        output_dir / "soft_motif_loocv_results.csv"
    )

    loocv_metrics = loocv_result.get_metrics()
    pd.Series(loocv_metrics, name="value").to_csv(output_dir / "soft_motif_loocv_metrics.csv")
    print(f"ROC AUC: {loocv_result.get_roc_auc_score():.3f}")
    print(loocv_metrics)

    save_table_image(
        loocv_results.drop(columns=["prob", "classes", "contributions", "shape_values"]),
        image_dir / "soft_motif_loocv_results.png",
        title="Soft motif LOOCV results",
        max_rows=40,
    )
    save_table_image(
        pd.Series(loocv_metrics, name="value").to_frame(),
        image_dir / "soft_motif_loocv_metrics.png",
        title="Soft motif LOOCV metrics",
    )

    confusion = pd.DataFrame(
        [[loocv_results["TP"].sum(), loocv_results["FN"].sum()],
         [loocv_results["FP"].sum(), loocv_results["TN"].sum()]],
        index=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
        columns=[f"Predicted {labels[0]}", f"Predicted {labels[1]}"],
    )
    confusion.to_csv(output_dir / "soft_motif_loocv_confusion_matrix.csv")
    plt.figure(figsize=(5, 4))
    plot_heatmap(confusion, title="Soft motif LOOCV confusion matrix")
    save_current_figure(image_dir / "soft_motif_loocv_confusion_matrix.png")

    true_labels, probabilities = extract_positive_probabilities(loocv_results, labels[0])
    y_true = [1 if label == labels[0] else 0 for label in true_labels]
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=loocv_result.get_roc_auc_score())
    display.plot()
    plt.title("Soft motif LOOCV ROC curve")
    save_current_figure(image_dir / "soft_motif_loocv_roc_curve.png")

    feature_importance_df = shap_importance_from_loocv(loocv_results)
    if feature_importance_df is not None:
        feature_importance_df["ID"] = feature_importance_df["ID"].astype(str)
        feature_importance_df.to_csv(output_dir / "soft_motif_feature_importance.csv", index=False)
        save_table_image(
            feature_importance_df.head(20),
            image_dir / "soft_motif_feature_importance_top20.png",
            title="Random Forest motif feature importance",
        )
        plt.figure(figsize=(9, 6))
        plot_bar(
            feature_importance_df.head(20),
            x="ID",
            y="mean_abs_shap",
            color="#59A14F",
            horizontal=True,
        )
        plt.title("Random Forest motif feature importance")
        plt.xlabel("Mean absolute SHAP value")
        plt.ylabel("Motif ID")
        save_current_figure(image_dir / "soft_motif_feature_importance.png")
    else:
        frequency_df = selection_frequency_from_loocv(loocv_result.selected_features_by_patient)
        frequency_df.to_csv(output_dir / "soft_motif_loocv_selection_frequency.csv", index=False)
        save_table_image(
            frequency_df.head(20),
            image_dir / "soft_motif_loocv_selection_frequency_top20.png",
            title="Soft motif LOOCV selection frequency",
        )
        plt.figure(figsize=(9, 6))
        plot_df = frequency_df.head(20).copy()
        plot_df["ID"] = plot_df["ID"].astype(str)
        plot_bar(
            plot_df,
            x="ID",
            y="selection_frequency",
            color="#59A14F",
            horizontal=True,
        )
        plt.title("Soft motif LOOCV selection frequency")
        plt.xlabel("Selection frequency across LOOCV folds")
        plt.ylabel("Motif ID")
        save_current_figure(image_dir / "soft_motif_loocv_selection_frequency.png")


if __name__ == "__main__":
    main()
