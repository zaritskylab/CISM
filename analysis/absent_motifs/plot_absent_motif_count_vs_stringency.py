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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = _REPO_ROOT
DEFAULT_BASE_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot unique absent motif count against stringency.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    selected = pd.read_csv(
        args.base_dir / "absent_discriminative_motifs.csv",
        usecols=["ID", "patient_percentage"],
    )
    summary = json.loads((args.base_dir / "absent_stringency_0364_subgraph_count.json").read_text())
    best = float(summary["stringency"])

    # A motif survives when at least one class-specific row exceeds the threshold.
    maximum_prevalence = selected.groupby("ID", observed=True)["patient_percentage"].max()
    thresholds = np.r_[0.0, np.sort(maximum_prevalence.unique())]
    counts = np.array([(maximum_prevalence > threshold).sum() for threshold in thresholds])
    curve = pd.DataFrame({"stringency": thresholds, "unique_absent_motifs": counts})
    curve.to_csv(args.base_dir / "absent_motif_count_vs_stringency_exact.csv", index=False)

    best_count = int((maximum_prevalence > best).sum())
    if best_count != int(summary["candidate_unique_subgraphs"]):
        raise ValueError("Optimal motif count does not match the saved summary")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    fig.patch.set_facecolor("white")
    visible = curve["unique_absent_motifs"].where(curve["unique_absent_motifs"] > 0)
    ax.step(curve["stringency"], visible, where="post", color="#236b8e", linewidth=2.2)
    ax.scatter(best, best_count, s=55, color="#c02c32", edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate(
        f"Optimal: {best:.3f}, {best_count} motifs",
        xy=(best, best_count),
        xytext=(best - 0.14, best_count * 3),
        arrowprops={"arrowstyle": "-", "color": "#c02c32", "lw": 1},
        color="#a9272c",
        fontsize=9,
    )
    ax.axvline(best, color="#c02c32", linestyle="--", linewidth=0.9, alpha=0.65)
    ax.axvspan(maximum_prevalence.max(), 0.5, color="#efefef", alpha=0.7)
    ax.text(0.468, 2, "0 motifs", ha="center", va="center", fontsize=8, color="#666666")
    ax.set(xlim=(0, 0.5), ylim=(0.8, 10000), xlabel="Shared patient fraction (stringency)", ylabel="Unique under-represented motifs")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", color="#e5e5e5", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output = args.base_dir / "absent_motif_count_vs_stringency_exact.png"
    fig.savefig(output)
    plt.close(fig)
    print(output)
    print(f"Optimal count: {best_count}; total unique motifs at zero stringency: {len(maximum_prevalence)}")


if __name__ == "__main__":
    main()
