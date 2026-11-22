#!/usr/bin/env python3
"""Plot statistically enriched per-patient counts for one CRC motif."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.crc_characterization.plot_crc_motifs_vs_stringency import LegacyReader


ROOT = _REPO_ROOT
DEFAULT_COUNTS = Path("/home/omriavit/research/discriminative_motifs_counts_CRC_4.csv")
DEFAULT_ARTIFACT = Path("/home/omriavit/research/cism_CRC_4.pkl")
DEFAULT_LABELS = Path("/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv")
DEFAULT_OUTPUT = ROOT / "analysis/outputs/crc_clr_motif_panels"
COHORT_TEXT_COLORS = {"CLR": "#2166AC", "DII": "#B33A3A"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif-id", type=int, required=True)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patient-class", choices=("all", "CLR", "DII"), default="all",
                        help="Cohort filter; default includes significant patients from both cohorts")
    parser.add_argument("--bar-color", default="#29A326",
                        help="Single color used for this motif's bars")
    parser.add_argument("--output-stem", default=None,
                        help="Figure filename without extension")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Read the underlying per-patient statistics directly.  A raw nonzero
    # occurrence is not sufficient for inclusion: only p < 0.05 is plotted.
    with args.artifact.open("rb") as handle:
        motif_data = LegacyReader(handle).load().motifs_dataset
    values = motif_data.loc[
        motif_data.ID.astype(int).eq(args.motif_id) & motif_data.p_value.lt(0.05),
        ["Patient", "Count", "p_value"],
    ].copy()
    if values.empty:
        raise ValueError(f"Motif {args.motif_id} has no patient-level calls with p < 0.05")

    values["Patient"] = values.Patient.astype(int)
    values = values.groupby("Patient", as_index=False).agg(Count=("Count", "sum"),
                                                            p_value=("p_value", "min"))
    labels = pd.read_csv(args.labels, usecols=["Patient", "Group"]).dropna()
    labels["Patient"] = labels.Patient.astype(int)
    labels["patient_class"] = labels.Group.astype(int).map({1: "CLR", 2: "DII"})
    values = values.merge(labels[["Patient", "patient_class"]], on="Patient", validate="one_to_one")
    values = values.loc[values.Count.gt(0)].copy()
    if args.patient_class != "all":
        values = values.loc[values.patient_class.eq(args.patient_class)].copy()
    values["cohort_order"] = values.patient_class.map({"CLR": 0, "DII": 1})
    values = values.sort_values(["cohort_order", "Patient"]).reset_index(drop=True)

    # The compact CSV used elsewhere must exactly match the significant rows.
    saved = pd.read_csv(args.counts)
    saved = saved.loc[saved.ID.astype(int).eq(args.motif_id), ["Patient", "Count"]]
    saved = saved.groupby("Patient", as_index=False).Count.sum()
    check = values[["Patient", "Count"]].merge(saved, on="Patient", how="outer",
                                                  suffixes=("_artifact", "_saved"), indicator=True)
    if args.patient_class == "all" and not (
        check._merge.eq("both").all()
        and check.Count_artifact.eq(check.Count_saved).all()
    ):
        raise ValueError("Saved motif-count rows do not match the artifact's p < 0.05 calls")
    values.to_csv(
        args.output_dir / f"crc_motif_{args.motif_id}_nonzero_patient_counts_by_class.csv",
        columns=["Patient", "patient_class", "Count", "p_value"], index=False,
    )

    n_clr = int(values.patient_class.eq("CLR").sum())
    x = np.arange(len(values), dtype=float)
    if args.patient_class == "all" and 0 < n_clr < len(values):
        x[n_clr:] += 0.65

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(10, 3.25))
        ax.bar(x, values.Count, width=0.78,
               color=args.bar_color, edgecolor="none")
        ax.set_yscale("log")
        ax.set_ylim(0.7, max(100, float(values.Count.max()) * 1.6))
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlabel("Patient", fontsize=11)
        ax.set_xticks(x, values.Patient.astype(str))
        ticks = [1, 10, 100, 1000, 10000]
        ax.set_yticks([tick for tick in ticks if tick < ax.get_ylim()[1]])
        ax.set_yticklabels([f"{tick:,}" for tick in ticks if tick < ax.get_ylim()[1]])
        ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(axis="both", length=0, labelsize=9, pad=4)
        if args.patient_class == "all" and 0 < n_clr < len(values):
            boundary = (x[n_clr - 1] + x[n_clr]) / 2
            ax.axvline(boundary, color="#6b7280", linestyle=(0, (4, 4)), linewidth=1)
            label_y = ax.get_ylim()[1] / 1.35
            ax.text(float(np.mean(x[:n_clr])), label_y, "CLR", ha="center", va="top",
                    fontsize=11, color=COHORT_TEXT_COLORS["CLR"], fontweight="bold")
            ax.text(float(np.mean(x[n_clr:])), label_y, "DII", ha="center", va="top",
                    fontsize=11, color=COHORT_TEXT_COLORS["DII"], fontweight="bold")
        fig.tight_layout()
        output_stem = args.output_stem or f"crc_motif_{args.motif_id}_patient_counts"
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"{output_stem}.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(values.drop(columns="cohort_order").to_string(index=False))
    print(f"Significant patients: {len(values)}; all p < 0.05; "
          f"CLR={n_clr}, DII={int(values.patient_class.eq('DII').sum())}")


if __name__ == "__main__":
    main()
