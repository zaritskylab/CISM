from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = _REPO_ROOT
BASE = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np_3node"
COLORS = {"NN": "#3979a8", "NP": "#c85058"}


def build_table(motif_table: Path, artifact_path: Path) -> pd.DataFrame:
    selected = pd.read_csv(motif_table)
    if len(selected) != 15 or selected.ID.nunique() != 15:
        raise ValueError("Expected the 15 unique 3-node motifs from the LOOCV fold union")
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    motifs = artifact["cism"].motifs_dataset
    required = {"ID", "patient_class", "Patient_uId", "observed_Count"}
    if required.difference(motifs.columns):
        raise ValueError(f"Missing motif columns: {sorted(required.difference(motifs.columns))}")

    patient_totals = (
        motifs.loc[motifs.ID.isin(selected.ID)]
        .groupby(["ID", "patient_class", "Patient_uId"], observed=True)
        .observed_Count.sum()
        .reset_index()
    )
    class_summary = (
        patient_totals.groupby(["ID", "patient_class"], observed=True)
        .agg(patient_count=("Patient_uId", "nunique"), average_count=("observed_Count", "mean"))
        .reset_index()
    )
    table = selected.merge(class_summary, on=["ID", "patient_class"], validate="one_to_one")
    if len(table) != len(selected):
        raise ValueError("Some fold-union motifs lack associated-class patient counts")
    other_counts = {
        (int(row.ID), str(row.patient_class)): int(row.patient_count)
        for row in class_summary.itertuples(index=False)
    }
    table["other_class_patient_count"] = [
        other_counts.get((int(row.ID), "NP" if row.patient_class == "NN" else "NN"), 0)
        for row in table.itertuples(index=False)
    ]
    return table


def plot(table: pd.DataFrame, output: Path) -> None:
    table = table.copy()
    table["plot_x"] = table.patient_count.astype(float)
    for (_cohort, _count), group in table.groupby(["patient_class", "patient_count"]):
        ordered = group.sort_values("average_count")
        clusters: list[list[int]] = []
        for index in ordered.index:
            if clusters and table.loc[index, "average_count"] - table.loc[clusters[-1][-1], "average_count"] <= 0.45:
                clusters[-1].append(index)
            else:
                clusters.append([index])
        for cluster in clusters:
            offsets = np.linspace(-0.1, 0.1, len(cluster)) if len(cluster) > 1 else [0.0]
            table.loc[cluster, "plot_x"] += offsets

    fig, ax = plt.subplots(figsize=(7.1, 4.5), dpi=240)
    for cohort in ("NN", "NP"):
        group = table.loc[table.patient_class.eq(cohort)]
        ax.scatter(group.plot_x, group.average_count, s=52, color=COLORS[cohort],
                   alpha=0.88, edgecolors="white", linewidths=0.8, label=cohort, zorder=3)
    stable = table.loc[table.selected_folds.eq(table.total_folds)]
    ax.scatter(stable.plot_x, stable.average_count, s=165, marker="s",
               facecolors="none", edgecolors="#333333", linewidths=1.1,
               label="Selected in all 38 folds", zorder=4)
    for row in stable.itertuples(index=False):
        ax.annotate(str(row.ID), (row.plot_x, row.average_count), xytext=(7, 5),
                    textcoords="offset points", fontsize=8, color="#333333")
    ax.set(xlabel="Patient Count", ylabel="Average of Count",
           xlim=(5.5, 9.8), ylim=(0, 37))
    ax.set_xticks([6, 7, 8, 9])
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot patient prevalence vs observed counts for 3-node anti-motifs.")
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(
        output_dir / "optimal_3node_fold_union_motifs.csv",
        args.base_dir / "Melanoma_NN_NP_absent_cism_ready.pkl",
    )
    csv_path = output_dir / "optimal_3node_motifs_patient_count_vs_average_count.csv"
    png_path = output_dir / "optimal_3node_motifs_patient_count_vs_average_count.png"
    table.to_csv(csv_path, index=False)
    plot(table, png_path)
    print(png_path)
    print(png_path.with_suffix(".pdf"))
    print(csv_path)


if __name__ == "__main__":
    main()
