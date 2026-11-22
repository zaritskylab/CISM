#!/usr/bin/env python3
"""Plot the maximum instance-count x enriched-patient score per optimized trial."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_ARTIFACT, DEFAULT_OUTPUT_ROOT


def load_trial_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    selected = pd.read_csv(run_dir / "selected_motifs_by_class.csv")
    if selected.empty or selected.duplicated(["patient_class", "ID"]).any():
        raise ValueError(f"Expected nonempty, unique selected motifs in {run_dir}")
    labels = pd.read_csv(run_dir / "patient_class_permuted.csv" if run_dir.name.startswith("seed_")
                         else run_dir / "patient_class_original.csv",
                         header=None, names=["Patient_uId", "patient_class"])
    if labels.Patient_uId.duplicated().any():
        raise ValueError(f"Duplicate patient labels in {run_dir}")
    return selected, labels.set_index("Patient_uId").patient_class


def score_trial(selected: pd.DataFrame, labels: pd.Series,
                patient_motifs: pd.DataFrame, run_type: str, seed: int | None) -> pd.DataFrame:
    local = patient_motifs.loc[patient_motifs.ID.isin(selected.ID)].copy()
    local["patient_class"] = local.Patient_uId.map(labels)
    if local.patient_class.isna().any():
        raise ValueError(f"Missing patient labels for {run_type} seed {seed}")
    local = local.loc[local.enriched].merge(selected, on=["patient_class", "ID"], how="inner",
                                            validate="many_to_one")
    scored = (local.groupby(["patient_class", "ID"], as_index=False)
              .agg(mean_instances_per_enriched_patient=("total_instances", "mean"),
                   enriched_patients=("Patient_uId", "nunique")))
    if len(scored) != len(selected):
        raise ValueError(f"Some selected motifs lack enriched patients in {run_type} seed {seed}")
    scored["score"] = scored.mean_instances_per_enriched_patient * scored.enriched_patients
    scored.insert(0, "seed", seed)
    scored.insert(0, "run_type", run_type)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    args = parser.parse_args()

    trials = pd.read_csv(args.run_root / "aggregate/all_trial_summaries.csv")
    shuffled = trials.loc[trials.shuffle_labels, ["seed", "n_selected_unique_motifs"]]
    if shuffled.seed.duplicated().any() or len(trials.loc[~trials.shuffle_labels]) != 1:
        raise ValueError("Expected unique shuffle seeds and exactly one original trial")

    inputs = []
    motif_ids = set()
    for seed, expected_count in shuffled.itertuples(index=False, name=None):
        selected, labels = load_trial_inputs(args.run_root / "shuffled" / f"seed_{seed}")
        if selected.ID.nunique() != expected_count:
            raise ValueError(f"Selected motif count differs from summary for seed {seed}")
        inputs.append(("shuffle", int(seed), selected, labels))
        motif_ids.update(selected.ID)
    original_dir = args.run_root / "original" / "from_artifact"
    original_selected, original_labels = load_trial_inputs(original_dir)
    inputs.append(("original", None, original_selected, original_labels))
    motif_ids.update(original_selected.ID)

    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    motifs = artifact["discriminator"].cism.motifs_dataset
    motif_rows = motifs.loc[motifs.ID.isin(motif_ids),
                            ["ID", "Patient_uId", "Count", "p_value"]].copy()
    motif_rows["enriched"] = motif_rows.p_value < 0.05
    patient_motifs = (motif_rows.groupby(["ID", "Patient_uId"], observed=True, as_index=False)
                      .agg(total_instances=("Count", "sum"), enriched=("enriched", "any")))
    del artifact, motifs, motif_rows

    motif_scores = pd.concat(
        [score_trial(selected, labels, patient_motifs, run_type, seed)
         for run_type, seed, selected, labels in inputs], ignore_index=True)
    winners = (motif_scores.sort_values("score", ascending=False)
               .drop_duplicates(["run_type", "seed"]).sort_values(["run_type", "seed"]))
    if len(winners) != len(inputs):
        raise ValueError("Expected one maximum per trial")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motif_scores.to_csv(args.output_dir / "optimal_motif_instance_patient_scores.csv", index=False)
    winners.to_csv(args.output_dir / "optimal_max_motif_instance_patient_score_by_trial.csv", index=False)

    shuffle_winners = winners.loc[winners.run_type == "shuffle"].sort_values("seed")
    original_score = float(winners.loc[winners.run_type == "original", "score"].iloc[0])
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        bin_width = 250
        bins = np.arange(0, shuffle_winners.score.max() + bin_width * 2, bin_width)
        ax.hist(shuffle_winners.score, bins=bins, color="#587fae", edgecolor="white",
                linewidth=0.7, label=f"Shuffled trials (n={len(shuffle_winners)})")
        ax.axvline(original_score, color="#c14536", linewidth=1.8,
                   label=f"Original maximum = {original_score:,.1f}")
        ax.set(xlabel="Maximum motif score per trial", ylabel="Number of shuffled seeds",
               title="Optimized stringency: maximum motif score across shuffles",
               xlim=(0, bins[-1]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d7dee8", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"optimal_max_motif_instance_patient_score_by_trial.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Shuffles: {len(shuffle_winners)}; original maximum: {original_score:.3f}; "
          f"shuffle maximum range: {shuffle_winners.score.min():.3f} to {shuffle_winners.score.max():.3f}")


if __name__ == "__main__":
    main()
