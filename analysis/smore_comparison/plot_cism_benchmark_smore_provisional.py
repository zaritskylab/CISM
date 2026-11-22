#!/usr/bin/env python3
"""Plot verified CISM benchmark AUCs against provisional SMORE seed AUCs."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from analysis.smore_comparison.build_clean_stringency_csvs import build_smore_auc
from analysis.smore_comparison.compact_smore_results import (
    DEFAULT_METADATA_DIR,
    DEFAULT_SMORE_RESULTS,
    load_labeled_smore,
)


OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "stringency_clean"
THRESHOLDS = [0.20, 0.30, 0.40, 0.42, 0.44, 0.46]
CISM_AUC = [
    0.3847222222222222,
    0.5333333333333333,
    0.7000000000000001,
    0.7819444444444444,
    0.7805555555555556,
    0.8916666666666667,
]
BENCHMARK = "Melanoma_100_distance_NN_NP_4_benchmark.ipynb"
DEPTH4 = "Melanoma_100_distance_NN_NP_depth_4_nodes.ipynb"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled = load_labeled_smore(DEFAULT_SMORE_RESULTS, DEFAULT_METADATA_DIR)
    cohort = labeled[["patient number", "class"]].drop_duplicates()
    class_counts = cohort.groupby("class")["patient number"].nunique().to_dict()
    if class_counts != {"NN": 20, "NP": 18}:
        raise ValueError(f"Expected 20 NN and 18 NP patients; found {class_counts}")
    smore = build_smore_auc(labeled, ["NN", "NP"], THRESHOLDS, 30, 0)
    cism = pd.DataFrame(
        {
            "method": "CISM 4-node benchmark",
            "stringency": THRESHOLDS,
            "auc": CISM_AUC,
            "valid_folds": 38,
            "skipped_folds": 0,
            "source": [BENCHMARK] * 5 + [DEPTH4],
        }
    )
    smore = smore.assign(
        method="SMORE seed (provisional)",
        source="smore_results.csv; existing seed-feature LOOCV evaluator",
    )
    combined = pd.concat([cism, smore], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "cism_benchmark_smore_provisional_auc.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(cism.stringency, cism.auc, color="#733ca5", marker="o", lw=2.4,
            label="CISM 4-node benchmark (38 patients)")
    ax.plot(smore.stringency, smore.auc, color="#53679b", marker="s", lw=2.2,
            label="SMORE seeds, provisional (38 patients)")
    ax.set(xlim=(0.195, 0.505), ylim=(0.15, 0.95),
           xlabel="Discrimination stringency", ylabel="NN-NP AUC")
    ax.set_xticks([round(0.20 + 0.05 * i, 2) for i in range(7)])
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"cism_benchmark_smore_provisional_auc.{suffix}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(combined[["method", "stringency", "auc", "valid_folds", "skipped_folds"]].to_string(index=False))


if __name__ == "__main__":
    main()
