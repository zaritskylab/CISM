#!/usr/bin/env python3
"""Stack validated CRC motif patient-count plots as two facets."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = _REPO_ROOT
INPUT = ROOT / "analysis/outputs/crc_clr_motif_panels"
OUTPUT = INPUT / "crc_selected_motif_patient_count_facets"
FACETS = [
    (538983436, "Four-B-cell motif", "#E1C400"),
    (1616384140, "B-cell/DC motif", "#29A326"),
]


def main() -> None:
    rows = []
    for motif_id, label, color in FACETS:
        path = INPUT / f"crc_motif_{motif_id}_nonzero_patient_counts_by_class.csv"
        data = pd.read_csv(path)
        if not data.p_value.lt(0.05).all():
            raise ValueError(f"{path} contains a nonsignificant patient call")
        data["cohort_order"] = data.patient_class.map({"CLR": 0, "DII": 1})
        data = data.sort_values(["cohort_order", "Patient"]).reset_index(drop=True)
        rows.append((data, label, color))

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharey=True)
        for ax, (data, label, color) in zip(axes, rows):
            n_clr = int(data.patient_class.eq("CLR").sum())
            x = np.arange(len(data), dtype=float)
            if 0 < n_clr < len(data):
                x[n_clr:] += 0.65
            ax.bar(x, data.Count, width=0.78, color=color, edgecolor="none")
            ax.set_yscale("log")
            ax.set_ylim(0.7, 5000)
            ax.set_yticks([1, 10, 100, 1000])
            ax.set_yticklabels(["1", "10", "100", "1,000"])
            ax.set_xticks(x, data.Patient.astype(int).astype(str))
            ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=0.65)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#bbbbbb")
            ax.tick_params(axis="x", length=0, labelsize=9, pad=4)
            ax.tick_params(axis="y", length=0, labelsize=13, pad=5)
        fig.supylabel("Count", x=0.015, fontsize=11)
        axes[-1].set_xlabel("Patient", fontsize=11)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.12, hspace=0.38)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300,
                        bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {OUTPUT}.png/.pdf/.svg")
    for (data, label, _color) in rows:
        print(f"{label}: {len(data)} significant patients "
              f"({data.patient_class.eq('CLR').sum()} CLR, "
              f"{data.patient_class.eq('DII').sum()} DII)")


if __name__ == "__main__":
    main()
