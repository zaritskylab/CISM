#!/usr/bin/env python3
"""Plot CLR discriminative motifs by patient prevalence and average count."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = _REPO_ROOT
DEFAULT_COUNTS = Path("/home/omriavit/research/discriminative_motifs_counts_CRC_4.csv")
DEFAULT_SELECTED = ROOT / "analysis/outputs/crc_pairwise/selected_motifs.csv"
DEFAULT_OUTPUT = ROOT / "analysis/outputs/crc_clr_motif_panels"
SELECTED_COLORS = {
    538983436: "#E1C400",   # Four-B-cell motif: matches yellow edges and bars.
    1616384140: "#29A326",  # B-cell/DC motif: matches green edges and bars.
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--color-mode", choices=("muted", "all"), default="muted")
    parser.add_argument("--output-stem", default="panel_C_CLR_motif_scatter_color_matched")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(args.counts)
    selected = pd.read_csv(args.selected)
    clr_ids = selected.loc[selected["patient_class"].eq("CLR"), "ID"].astype(int).unique()
    clr = counts[counts["patient_class"].eq("CLR") & counts["ID"].isin(clr_ids)].copy()
    if clr.empty:
        raise ValueError("No CLR motif counts matched the optimal-stringency CLR motif set")

    summary = (clr.groupby("ID", as_index=False)
               .agg(patient_count=("Patient", "nunique"),
                    average_count=("Count", "mean"),
                    total_count=("Count", "sum"))
               .sort_values(["total_count", "patient_count", "ID"],
                            ascending=[False, False, True])
               .reset_index(drop=True))
    summary["pooled_count_rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(args.output_dir / "panel_C_CLR_motif_scatter_data.csv", index=False)

    # The selected motifs always retain their cross-panel semantic colors.
    # Other motifs can either be muted or assigned stable categorical colors.
    if args.color_mode == "all":
        base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
        colors = [SELECTED_COLORS.get(
            int(motif_id), base[(int(motif_id) * 2654435761) % len(base)])
            for motif_id in summary["ID"]]
    else:
        colors = [SELECTED_COLORS.get(int(motif_id), "#A9ADB3")
                  for motif_id in summary["ID"]]

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(4.2, 3.35))
        ax.scatter(summary["patient_count"], summary["average_count"],
                   c=colors, s=24, edgecolors="white", linewidths=.45,
                   alpha=.9, zorder=3)
        for motif_id in SELECTED_COLORS:
            highlighted = summary.loc[summary["ID"].eq(motif_id)]
            if len(highlighted) != 1:
                raise ValueError(f"Expected one row for motif {motif_id}, found {len(highlighted)}")
            ax.scatter(highlighted["patient_count"], highlighted["average_count"],
                       s=135, marker="s", facecolors="none", edgecolors="black",
                       linewidths=.9, linestyles=(0, (2.2, 2.2)), zorder=4)

        ax.set_xlabel("Patient count", fontsize=11)
        ax.set_ylabel("Average count", fontsize=11)
        ax.set_xlim(0, 17.5)
        ax.set_ylim(0, max(600, float(summary["average_count"].max()) * 1.08))
        ax.set_xticks([0, 5, 10, 15])
        ax.tick_params(axis="both", length=0, labelsize=9, colors="#555555")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.grid(False)
        ax.text(-.22, 1.045, "C", transform=ax.transAxes, fontsize=11,
                ha="left", va="top", color="#222222")
        fig.tight_layout(pad=.8)

        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"{args.output_stem}.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    provenance = {
        "scope": "CLR patients only",
        "motifs": "CLR discriminative motifs selected at optimal stringency 0.55",
        "patient_count": "Number of unique CLR patients with a recorded motif occurrence",
        "average_count": "Arithmetic mean of Count among CLR patients with a recorded occurrence",
        "points": int(len(summary)),
        "highlighted_motifs": list(SELECTED_COLORS),
        "highlight_colors": {str(k): v for k, v in SELECTED_COLORS.items()},
        "other_point_colors": args.color_mode,
        "highlight": "Dashed black squares; no text labels",
        "counts_source": str(args.counts),
        "selected_motifs_source": str(args.selected),
    }
    (args.output_dir / f"{args.output_stem}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(summary.head(10).to_string(index=False))
    print(f"Saved {len(summary)} motif points to {args.output_dir}")


if __name__ == "__main__":
    main()
