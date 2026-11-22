#!/usr/bin/env python3
"""Plot SMORE discriminative motif counts per patient at a fixed stringency."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
import sys
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT))

from analysis.smore_comparison.compact_smore_results import DEFAULT_METADATA_DIR, DEFAULT_SMORE_RESULTS, load_labeled_smore, motif_stringency


OUTPUT_DIR = ROOT / "analysis" / "outputs" / "stringency_clean" / "figures"
CSV_DIR = ROOT / "analysis" / "outputs" / "stringency_clean"

COLORS = {
    "NN": "#8E44AD",
    "NP": "#8B9AF7",
}


def threshold_label(threshold: float) -> str:
    return f"{threshold:.3g}".replace(".", "p")


def natural_patient_order(value: object) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def build_counts(threshold: float) -> pd.DataFrame:
    labeled = load_labeled_smore(DEFAULT_SMORE_RESULTS, DEFAULT_METADATA_DIR)
    exclusive = motif_stringency(labeled)
    passing = (
        exclusive[exclusive["stringency"] >= threshold][["motif", "exclusive_to", "stringency"]]
        .rename(columns={"motif": "seed"})
        .copy()
    )

    patient_class = labeled[["patient number", "class"]].drop_duplicates().copy()
    patient_class["patient_order"] = patient_class["patient number"].map(natural_patient_order)
    patient_class = patient_class.sort_values(["class", "patient_order"])

    present = (
        labeled.merge(passing, on="seed", how="inner")
        .drop_duplicates(["patient number", "seed", "exclusive_to"])
        .groupby(["patient number", "exclusive_to"], observed=True)["seed"]
        .nunique()
        .reset_index(name="motif_count")
    )

    wide = (
        present[present["exclusive_to"].isin(["NN", "NP"])]
        .pivot_table(
            index="patient number",
            columns="exclusive_to",
            values="motif_count",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        .reset_index()
    )
    wide = patient_class[["patient number", "class", "patient_order"]].merge(
        wide,
        on="patient number",
        how="left",
    )
    for label in ("NN", "NP"):
        if label not in wide.columns:
            wide[label] = 0
        wide[label] = wide[label].fillna(0).astype(int)
    wide["total"] = wide["NN"] + wide["NP"]
    return wide.sort_values(["class", "patient_order"]).reset_index(drop=True)


def plot_counts(counts: pd.DataFrame, threshold: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    threshold_text = threshold_label(threshold)
    csv_path = CSV_DIR / f"smore_discriminative_motifs_per_patient_stringency_{threshold_text}.csv"
    counts.to_csv(csv_path, index=False)

    x = range(len(counts))
    labels = counts["patient number"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(x, counts["NN"], color=COLORS["NN"], label="NN-exclusive motifs", width=0.82)
    ax.bar(x, counts["NP"], bottom=counts["NN"], color=COLORS["NP"], label="NP-exclusive motifs", width=0.82)

    class_boundary = counts.index[counts["class"].ne(counts["class"].shift())].tolist()
    for boundary in class_boundary[1:]:
        ax.axvline(boundary - 0.5, color="#9CA3AF", linewidth=0.9, linestyle=":")

    ymax = max(1, int(counts["total"].max()))
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("Patient")
    ax.set_ylabel("# discriminative SMORE motifs")
    ax.set_title(f"SMORE discriminative motifs per patient at stringency {threshold:g}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    for xpos, row in enumerate(counts.itertuples(index=False)):
        if row.total > 0:
            ax.text(xpos, row.total + ymax * 0.015, str(row.total), ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    stem = f"smore_discriminative_motifs_per_patient_stringency_{threshold_text}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote plots to {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stringency", type=float, default=0.3)
    args = parser.parse_args()
    plot_counts(build_counts(args.stringency), args.stringency)


if __name__ == "__main__":
    main()
