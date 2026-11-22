#!/usr/bin/env python3
"""Plot cell-type distributions induced by CRC discriminative motif templates."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import base64
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.crc_characterization.plot_crc_motifs_vs_stringency import LegacyReader


ROOT = _REPO_ROOT
DEFAULT_ARTIFACT = Path("/home/omriavit/research/cism_CRC_4.pkl")
DEFAULT_SELECTED = ROOT / "analysis/outputs/crc_pairwise/selected_motifs.csv"
DEFAULT_OUTPUT = ROOT / "analysis/outputs/crc_motif_cell_type_distribution"
CELL_TYPES = [
    "Bcell", "CD3T", "CD4T", "CD8T", "DC", "Endothelial", "Fibroblast",
    "Lymphatic", "Macrophage", "Neuron", "Neutrophil", "Plasma", "SMV",
    "Stroma", "Treg", "Tumor", "Unidentified",
]
DISPLAY = {
    "Bcell": "B cells", "CD3T": "CD3 T cells", "CD4T": "CD4 T cells",
    "CD8T": "CD8 T cells",
}
COLORS = {"CLR": "#579bd3", "DII": "#c74f52"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected)
    required = {"ID", "patient_class", "mean_patient_frequency"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Selected-motif table is missing {sorted(required - set(selected.columns))}")
    if set(selected["patient_class"]) != {"CLR", "DII"}:
        raise ValueError("Expected both CLR and DII motif assignments")

    with args.artifact.open("rb") as handle:
        motifs = LegacyReader(handle).load().motifs_dataset
    representatives = motifs[motifs["ID"].isin(selected["ID"])].drop_duplicates("ID").set_index("ID")
    missing = set(selected["ID"]) - set(representatives.index)
    if missing:
        raise ValueError(f"Artifact lacks {len(missing)} selected motifs")

    weighted_nodes = {group: np.zeros(len(CELL_TYPES), dtype=float) for group in ("CLR", "DII")}
    for row in selected.itertuples(index=False):
        graph = pickle.loads(base64.b64decode(representatives.loc[row.ID, "motif"]))
        if graph.number_of_nodes() != 4:
            raise ValueError(f"Motif {row.ID} has {graph.number_of_nodes()} nodes; expected 4")
        for _, attributes in graph.nodes(data=True):
            weighted_nodes[row.patient_class][int(attributes["type"])] += row.mean_patient_frequency

    frequencies = {
        group: values / values.sum()
        for group, values in weighted_nodes.items()
    }
    if not all(np.isclose(values.sum(), 1) for values in frequencies.values()):
        raise AssertionError("Class distributions must each sum to one")

    table = pd.DataFrame({
        "cell_type": CELL_TYPES,
        "display_name": [DISPLAY.get(value, value) for value in CELL_TYPES],
        "CLR": frequencies["CLR"],
        "DII": frequencies["DII"],
    })
    table["mean_frequency"] = table[["CLR", "DII"]].mean(axis=1)
    table = table.sort_values(["mean_frequency", "cell_type"], ascending=[False, True]).reset_index(drop=True)
    table.to_csv(args.output_dir / "crc_motif_induced_cell_type_distribution.csv", index=False)

    x = np.arange(len(table))
    width = .38
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(11.2, 4.3))
        ax.bar(x - width / 2, table["CLR"], width, color=COLORS["CLR"],
               edgecolor="none", label="CLR")
        ax.bar(x + width / 2, table["DII"], width, color=COLORS["DII"],
               edgecolor="none", label="DII")
        ax.set_ylabel("Frequency", fontsize=13)
        ax.set_xticks(x, table["display_name"], rotation=45, ha="right", fontsize=9)
        ymax = max(.30, float(table[["CLR", "DII"]].to_numpy().max()) * 1.18)
        ax.set_ylim(0, ymax)
        ax.set_yticks(np.arange(0, ymax + .001, .05))
        ax.tick_params(axis="y", length=0, labelsize=9, colors="#777777")
        ax.tick_params(axis="x", length=0, colors="#777777")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        legend = ax.legend(loc="upper right", frameon=False, fontsize=11)
        for text, group in zip(legend.get_texts(), ("CLR", "DII")):
            text.set_color(COLORS[group])
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"crc_motif_induced_cell_type_distribution.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    metadata = {
        "motif_size": 4,
        "stringency": 0.55,
        "selected_motifs": int(len(selected)),
        "selected_by_class": selected["patient_class"].value_counts().to_dict(),
        "motif_weight": "Mean patient frequency from the established CRC motif-induced pairwise analysis",
        "calculation": "For each motif, add its weight once for every template node of that cell type; normalize each class to sum to one",
        "ordering": "Descending mean of CLR and DII frequencies",
        "artifact": str(args.artifact),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "selected_motifs_source": str(args.selected),
        "colors": COLORS,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(table.to_string(index=False))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
