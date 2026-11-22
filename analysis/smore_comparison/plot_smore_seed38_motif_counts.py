#!/usr/bin/env python3
"""Plot CISM and provisional SMORE seed counts on the complete melanoma cohort."""

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from analysis.smore_comparison.compact_smore_results import (
    DEFAULT_METADATA_DIR,
    DEFAULT_SMORE_RESULTS,
    load_labeled_smore,
    motif_stringency,
)


OUTPUT = ANALYSIS_DIR / "outputs/stringency_clean"
GRID = tuple(round(i * 0.05, 2) for i in range(13))


def count_by_class(motifs: pd.DataFrame, method: str, label_column: str) -> pd.DataFrame:
    rows = []
    for threshold in GRID:
        eligible = motifs[motifs.stringency >= threshold]
        counts = eligible[label_column].value_counts()
        for label in ("NN", "NP"):
            rows.append({"method": method, "stringency": threshold, "class": label,
                         "motif_count": int(counts.get(label, 0)), "patients": 38})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labeled = load_labeled_smore(DEFAULT_SMORE_RESULTS, DEFAULT_METADATA_DIR)
    cohort = labeled[["patient number", "class"]].drop_duplicates()
    if cohort.groupby("class")["patient number"].nunique().to_dict() != {"NN": 20, "NP": 18}:
        raise ValueError("Expected 20 NN and 18 NP patients")
    smore_motifs = motif_stringency(labeled)
    cism_motifs = pd.read_csv(OUTPUT / "cism4_melanoma_discriminative_motifs.csv")
    cism = count_by_class(cism_motifs, "CISM 4-node", "motif_label")
    smore = count_by_class(smore_motifs, "SMORE seeds, provisional", "exclusive_to")
    combined = pd.concat([cism, smore], ignore_index=True)
    combined.to_csv(OUTPUT / "cism4_smore_seed38_motif_count_vs_stringency.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for method, color in (("CISM 4-node", "#733ca5"),
                          ("SMORE seeds, provisional", "#53679b")):
        for label, linestyle in (("NN", "--"), ("NP", ":")):
            group = combined[(combined.method == method) & (combined["class"] == label)]
            positive = group[group.motif_count > 0]
            ax.plot(positive.stringency, positive.motif_count, color=color,
                    linestyle=linestyle, marker="o", markersize=3.5, linewidth=1.9,
                    label=f"{method} {label}")
            zero = group[group.motif_count == 0]
            if not zero.empty:
                ax.scatter(zero.stringency, [0.8] * len(zero), marker="x",
                           s=24, linewidths=1.2, color=color)
    ax.set(xlim=(0, 0.6), ylim=(0.7, 20000), yscale="log",
           xlabel="Discrimination stringency", ylabel="Class-exclusive motif count")
    ax.set_xticks([round(i * 0.05, 2) for i in range(13)])
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"cism4_smore_seed38_motif_count_vs_stringency.{suffix}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(combined[combined.stringency.isin([0.0, 0.2, 0.3, 0.4, 0.5])].to_string(index=False))


if __name__ == "__main__":
    main()
