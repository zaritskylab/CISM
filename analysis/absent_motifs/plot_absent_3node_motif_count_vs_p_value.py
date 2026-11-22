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
sys.path.insert(0, str(ROOT))
BASE = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np_3node"


def build_counts(motifs: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    if bin_width <= 0 or not np.isclose(0.05 / bin_width, round(0.05 / bin_width)):
        raise ValueError("Bin width must divide 0.05 evenly")
    required = {"ID", "patient_class", "p_value_normal_two_tailed", "p_value_direction"}
    missing = required.difference(motifs.columns)
    if missing:
        raise ValueError(f"Missing motif columns: {sorted(missing)}")
    depleted = motifs.loc[
        motifs.patient_class.isin(["NN", "NP"])
        & motifs.p_value_direction.eq("absence")
    ]
    minimum_p = depleted.groupby("ID", observed=True).p_value_normal_two_tailed.min()
    if minimum_p.empty or not np.isfinite(minimum_p).all():
        raise ValueError("No finite p-values for depleted NN/NP motifs")
    if (minimum_p < 0).any() or (minimum_p >= 0.05).any():
        raise ValueError("Expected significant two-tailed p-values in [0, 0.05)")
    edges = np.linspace(0, 0.05, int(round(0.05 / bin_width)) + 1)
    counts, _ = np.histogram(minimum_p.to_numpy(), bins=edges)
    return pd.DataFrame({
        "p_value_left": edges[:-1],
        "p_value_right": edges[1:],
        "p_value_mid": (edges[:-1] + edges[1:]) / 2,
        "unique_motif_count": counts,
    })


def plot(counts: pd.DataFrame, output: Path) -> None:
    visible = counts.loc[counts.unique_motif_count.gt(0)]
    fig, ax = plt.subplots(figsize=(6.2, 4.1), dpi=240)
    ax.scatter(visible.p_value_mid, visible.unique_motif_count, marker="_", s=145,
               color="#646b70", linewidths=1.5)
    ax.set(xlim=(0, 0.05), xlabel="P-value", ylabel="Count")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=5)
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3-node depleted motif count by two-tailed p-value.")
    parser.add_argument("--artifact", type=Path, default=BASE / "Melanoma_NN_NP_absent_cism_ready.pkl")
    parser.add_argument("--output-dir", type=Path, default=BASE / "figures")
    parser.add_argument("--bin-width", type=float, default=0.001)
    args = parser.parse_args()
    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    if artifact["cism"].motif_size != 3:
        raise ValueError("Expected a 3-node CISM artifact")
    counts = build_counts(artifact["cism"].motifs_dataset, args.bin_width)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "absent_3node_motif_count_vs_p_value"
    counts.to_csv(stem.with_suffix(".csv"), index=False)
    plot(counts, stem.with_suffix(".png"))
    print(f"Unique motifs: {counts.unique_motif_count.sum()}")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
