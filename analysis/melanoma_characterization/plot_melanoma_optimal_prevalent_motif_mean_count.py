#!/usr/bin/env python3
"""Plot mean instance count for the most patient-prevalent optimized motif per trial."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    args = parser.parse_args()

    scores = pd.read_csv(args.output_dir / "optimal_motif_instance_patient_scores.csv")
    required = {"run_type", "seed", "patient_class", "ID", "enriched_patients",
                "mean_instances_per_enriched_patient"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Missing motif-score columns: {sorted(required - set(scores.columns))}")
    winners = (scores.sort_values(["enriched_patients", "mean_instances_per_enriched_patient", "ID"],
                                  ascending=[False, False, True])
               .drop_duplicates(["run_type", "seed"])
               .sort_values(["run_type", "seed"]))
    shuffled = winners.loc[winners.run_type == "shuffle"]
    original = winners.loc[winners.run_type == "original"]
    if len(original) != 1 or len(shuffled) != scores.loc[scores.run_type == "shuffle", "seed"].nunique():
        raise ValueError("Expected one original and one maximum per shuffled seed")
    winners.to_csv(args.output_dir / "optimal_prevalent_motif_max_mean_count_by_trial.csv", index=False)

    original_value = float(original.mean_instances_per_enriched_patient.iloc[0])
    values = shuffled.mean_instances_per_enriched_patient.to_numpy()
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        bin_width = 10
        bins = np.arange(0, values.max() + bin_width * 2, bin_width)
        ax.hist(values, bins=bins, color="#587fae", edgecolor="white", linewidth=0.7,
                label=f"Shuffled trials (n={len(values)})")
        ax.axvline(original_value, color="#c14536", linewidth=1.8,
                   label=f"Original = {original_value:.2f} ({int(original.enriched_patients.iloc[0])} patients)")
        ax.set(xlabel="Highest mean instance count among most prevalent motifs",
               ylabel="Number of shuffled seeds",
               title="Optimized stringency: instance count at maximum patient prevalence",
               xlim=(0, bins[-1]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d7dee8", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"optimal_prevalent_motif_max_mean_count_by_trial.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"n={len(values)}; original={original_value:.6f} across "
          f"{int(original.enriched_patients.iloc[0])} patients; "
          f"shuffle range={values.min():.6f} to {values.max():.6f}")


if __name__ == "__main__":
    main()
