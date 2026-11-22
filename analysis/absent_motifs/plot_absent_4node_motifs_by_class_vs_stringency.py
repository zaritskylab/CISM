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


REPO_ROOT = _REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

from cism.helpers import string_base64_pickle  # noqa: E402


DEFAULT_BASE_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
)
OPTIMAL_STRINGENCY = 0.3642694428844888
CLASS_COLORS = {"NN": "#2776a3", "NP": "#c94d57"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 4-node discriminative absent motif counts by class and stringency."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--optimal-stringency", type=float, default=OPTIMAL_STRINGENCY)
    args = parser.parse_args()

    artifact_path = args.artifact or args.base_dir / "Melanoma_NN_NP_absent_cism_ready.pkl"
    selected = pd.read_csv(
        args.base_dir / "absent_discriminative_motifs.csv",
        usecols=["patient_class", "ID", "patient_percentage"],
    )
    selected = selected[selected["patient_class"].isin(CLASS_COLORS)]

    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    motif_lookup = (
        artifact["cism"].motifs_dataset
        .loc[lambda table: table["ID"].isin(selected["ID"])]
        .drop_duplicates("ID")
        .set_index("ID")["motif"]
    )
    if len(motif_lookup) != selected["ID"].nunique():
        raise ValueError("Some selected motif graphs are missing from the CISM artifact")
    four_node_ids = set()
    for motif_id, obj in motif_lookup.items():
        graph = string_base64_pickle(obj) if isinstance(obj, str) else obj
        if graph.number_of_nodes() == 4:
            four_node_ids.add(motif_id)
    selected = selected[selected["ID"].isin(four_node_ids)]

    prevalence = (
        selected.groupby(["patient_class", "ID"], observed=True)["patient_percentage"]
        .max()
        .reset_index()
    )
    thresholds = np.r_[0.0, np.sort(prevalence["patient_percentage"].unique())]
    curve = pd.DataFrame({"stringency": thresholds})
    for label in CLASS_COLORS:
        class_prevalence = prevalence.loc[
            prevalence["patient_class"] == label, "patient_percentage"
        ].to_numpy()
        curve[f"{label}_motifs"] = [int(np.count_nonzero(class_prevalence > x)) for x in thresholds]

    csv_path = args.base_dir / "absent_4node_motif_count_by_class_vs_stringency.csv"
    png_path = args.base_dir / "absent_4node_motif_count_by_class_vs_stringency.png"
    curve.to_csv(csv_path, index=False)
    draw_plot(curve, args.optimal_stringency, png_path)
    print(png_path)
    for label in CLASS_COLORS:
        count = int((prevalence.loc[prevalence.patient_class == label, "patient_percentage"] > args.optimal_stringency).sum())
        print(f"{label}: {count} motifs at optimal stringency")


def draw_plot(curve: pd.DataFrame, optimum: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=220)
    fig.patch.set_facecolor("white")
    for label, color in CLASS_COLORS.items():
        values = curve[f"{label}_motifs"]
        changed = values.ne(values.shift(fill_value=-1))
        anchors = curve.loc[changed & values.gt(0), ["stringency", f"{label}_motifs"]].copy()
        count_at_optimum = int(
            values.iloc[np.searchsorted(curve["stringency"].to_numpy(), optimum, side="right") - 1]
        )
        anchors = pd.concat(
            [
                anchors,
                pd.DataFrame(
                    {
                        "stringency": [optimum],
                        f"{label}_motifs": [count_at_optimum],
                    }
                ),
            ],
            ignore_index=True,
        ).sort_values("stringency").drop_duplicates("stringency")
        ax.plot(
            anchors["stringency"],
            anchors[f"{label}_motifs"],
            color=color,
            lw=2.4,
            label=f"{label} (4 nodes)",
        )
        ax.scatter([optimum], [count_at_optimum], s=42, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(
            f"{label}: {count_at_optimum}",
            (optimum, count_at_optimum),
            xytext=(8, 6 if label == "NP" else -17),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    ax.axvline(optimum, color="#777777", linestyle="--", lw=1.0, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.8, 6000)
    ax.set_xlabel("Fraction of patients sharing the motif (stringency)")
    ax.set_ylabel("Number of discriminative anti-motifs")
    ax.grid(axis="y", which="both", color="#e7e7e7", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
