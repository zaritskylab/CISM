#!/usr/bin/env python3
"""Restyle the existing pooled CRC graph-cell frequencies as a paper panel."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/omriavit/research/CRC_cell_type_frequency.csv"))
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "outputs/crc_cell_type_frequency")
    parser.add_argument("--panel-label", default="B")
    args = parser.parse_args()
    data = pd.read_csv(args.source)[["Cell Type", "Frequency", "patient_class"]]
    if data.duplicated(["Cell Type", "patient_class"]).any():
        raise ValueError("Expected one frequency per class and cell type")
    if set(data.patient_class) != {"CLR", "DII"}:
        raise ValueError("Expected CLR and DII groups")
    if not data.Frequency.between(0, 1).all():
        raise ValueError("Frequencies must lie between zero and one")
    if not np.allclose(data.groupby("patient_class").Frequency.sum(), 1):
        raise ValueError("Frequencies must sum to one within each class")
    table = data.pivot(index="Cell Type", columns="patient_class", values="Frequency")
    if table.isna().any().any():
        raise ValueError("Each cell type must be represented in both groups")
    order = table.mean(axis=1).sort_values(ascending=False, kind="stable").index
    table = table.loc[order, ["CLR", "DII"]]
    # User's requested class mapping, with the reference's muted hues.
    colors = {"CLR": "#589cda", "DII": "#c65356"}
    display = {"Bcell": "B cells", "CD3T": "CD3 T cells", "CD4T": "CD4 T cells",
               "CD8T": "CD8 T cells", "Treg": "Treg", "SMV": "SMV"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        x = np.arange(len(table))
        width = 0.38
        for offset, group in [(-width / 2, "CLR"), (width / 2, "DII")]:
            ax.bar(x + offset, table[group], width=width, color=colors[group],
                   edgecolor="none", label=group)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xticks(x, [display.get(name, name) for name in table.index],
                      rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_xlim(-0.7, len(table) - 0.3)
        ax.set_ylim(0, np.ceil(table.to_numpy().max() * 1.18 / 0.05) * 0.05)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#dddddd")
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="both", length=0, colors="#888888", pad=4)
        ax.tick_params(axis="y", labelsize=8)
        legend = ax.legend(loc="upper right", frameon=False, fontsize=10,
                           handlelength=2, handletextpad=0.6, borderaxespad=0)
        for text, group in zip(legend.get_texts(), ["CLR", "DII"]):
            text.set_color(colors[group])
        if args.panel_label:
            fig.text(0.025, 0.96, args.panel_label, fontsize=13, ha="left", va="top")
        fig.subplots_adjust(left=0.12, right=0.98, top=0.87, bottom=0.31)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(args.output_dir / f"crc_cell_type_frequency.{extension}", dpi=300)
        plt.close(fig)
    table.to_csv(args.output_dir / "crc_cell_type_frequency.csv")
    metadata = {"source": str(args.source.resolve()),
                "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "aggregation": "Existing pooled graph-cell frequencies within each class; not patient means",
                "order": "Descending unweighted mean of CLR and DII frequencies",
                "colors": colors, "panel_label": args.panel_label,
                "frequency_values": "Unchanged from source"}
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(table.to_string())
    print(f"Saved figure and table to {args.output_dir}")


if __name__ == "__main__":
    main()
