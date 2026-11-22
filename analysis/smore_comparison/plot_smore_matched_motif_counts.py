#!/usr/bin/env python3
"""Plot full-cohort class-exclusive motif counts for matched SMORE and CISM."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from analysis.smore_comparison.evaluate_smore_matched import OUTPUT, validate_manifest


ROOT = _REPO_ROOT
CISM_MOTIFS = ROOT / "analysis/outputs/stringency_clean/cism4_melanoma_discriminative_motifs.csv"
GRID = tuple(round(i * 0.05, 2) for i in range(13))


def smore_exclusive(features: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    present = features.loc[features.frequency > 0, ["patient", "feature"]].merge(
        patients, on="patient", validate="many_to_one"
    )
    counts = (
        present.groupby(["feature", "class"], observed=True)["patient"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=["NN", "NP"], fill_value=0)
    )
    counts = counts[(counts.NN == 0) | (counts.NP == 0)].copy()
    counts["class"] = counts.NN.gt(0).map({True: "NN", False: "NP"})
    class_sizes = patients.groupby("class").patient.nunique().to_dict()
    counts["stringency"] = counts.apply(
        lambda row: row[row["class"]] / class_sizes[row["class"]], axis=1
    )
    return counts.reset_index()


def count_grid(motifs: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for threshold in GRID:
        eligible = motifs[motifs.stringency >= threshold]
        counts = eligible["class"].value_counts()
        for label in ("NN", "NP"):
            rows.append({"method": method, "stringency": threshold, "class": label,
                         "motif_count": int(counts.get(label, 0))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    run_dir = OUTPUT / "runs" / args.run_tag
    manifest = pd.read_csv(OUTPUT / "cohort_fovs.csv")
    patients = validate_manifest(manifest)
    features = pd.read_csv(run_dir / "smore_matched_patient_features.csv")
    smore = smore_exclusive(features, patients)
    cism = pd.read_csv(CISM_MOTIFS).rename(columns={"motif_label": "class"})
    metadata = pd.read_csv(next(run_dir.glob("FOV*_metadata.csv"))).iloc[0]
    sampling_percent = round(float(metadata.path_probability) * 100)
    smore_method = f"SMORE GPU, {sampling_percent}% paths"
    combined = pd.concat([
        count_grid(cism, "CISM 4-node"),
        count_grid(smore, smore_method),
    ], ignore_index=True)
    combined.to_csv(run_dir / "smore_cism_motif_count_vs_stringency.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for method, color in (("CISM 4-node", "#8E44AD"),
                          (smore_method, "#6676C8")):
        for label, style in (("NN", "--"), ("NP", ":")):
            group = combined[(combined.method == method) & (combined["class"] == label)]
            positive = group[group.motif_count > 0]
            ax.plot(positive.stringency, positive.motif_count, color=color,
                    linestyle=style, marker="o", markersize=3.5, linewidth=2,
                    label=f"{'FANMOD+' if method.startswith('CISM') else 'SMORE'} {label}")
            zero = group[group.motif_count == 0]
            if not zero.empty:
                ax.scatter(zero.stringency, [0.8] * len(zero), color=color,
                           marker="x", s=24, linewidths=1.2)
    ax.set(xlim=(0, 0.6), ylim=(0.7, 20000), yscale="log",
           xlabel="Discrimination stringency", ylabel="Class-exclusive motif count")
    ax.set_xticks(GRID)
    ax.set_xlabel("Discrimination stringency", fontsize=14)
    ax.set_ylabel("Class-exclusive motif count", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=12, ncol=2)
    fig.text(0.5, 0.01, "Full 38-patient cohort; significant SMORE PWM consensus motifs; zero counts marked x.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(run_dir / f"smore_cism_motif_count_vs_stringency.{ext}",
                    dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(combined[combined.stringency.isin([0.0, 0.2, 0.3, 0.4, 0.5])].to_string(index=False))


if __name__ == "__main__":
    main()
