#!/usr/bin/env python3
"""Illustrate the paper's 14 contact-analysis groups versus CISM's 17 labels."""

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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = _REPO_ROOT
OUT = ROOT / "analysis/outputs/crc_cell_type_hierarchy"
STEM = OUT / "crc_paper14_to_cism17_mapping"

ROWS = [
    ("B cells", "Bcell", "direct"),
    ("CD11c⁺ DCs", "DC", "broader"),
    ("CD4⁺ T cells", "CD4T", "direct"),
    ("CD8⁺ T cells", "CD8T", "direct"),
    ("Granulocytes", "Neutrophil", "broader"),
    ("Macrophages", "Macrophage", "broader"),
    ("NK cells", "No separate NK label", "absent"),
    ("Other immune cells", "CD3T (partial correspondence)", "partial"),
    ("Plasma cells", "Plasma", "direct"),
    ("Smooth muscle", "SMV", "renamed"),
    ("Stroma", "Stroma  +  Fibroblast", "split"),
    ("Tregs", "Treg", "direct"),
    ("Tumor cells", "Tumor", "direct"),
    ("Vasculature", "Endothelial  +  Lymphatic", "split"),
]

STYLE = {
    "direct": ("#E8F2FF", "#2F7FC1", "direct equivalent"),
    "renamed": ("#E8F2FF", "#2F7FC1", "renamed"),
    "broader": ("#FFF3DB", "#D97706", "broader/nearest label"),
    "partial": ("#FFF3DB", "#D97706", "partial correspondence"),
    "split": ("#E8F7ED", "#2E8B57", "represented by two labels"),
    "absent": ("#FCEBEC", "#BE3E42", "not represented separately"),
}


def box(ax, xy, width, height, face, edge, text, fontsize=10,
        weight="normal", color="#172033"):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=color)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(13, 12.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.045, 0.965, "CRC cell-type correspondence",
                fontsize=25, fontweight="bold", color="#172033", va="top")
        ax.text(0.045, 0.925,
                "Paper's 14 groups for direct-contact analysis versus the 17 labels used by CISM",
                fontsize=13, color="#4B5870", va="top")

        box(ax, (0.06, 0.840), 0.32, 0.045, "#F2F5F8", "#B9C5D2",
            "Paper: 14 contact-analysis groups", 11, "bold")
        box(ax, (0.56, 0.840), 0.37, 0.045, "#E6F0FC", "#79A7D8",
            "Labels used by CISM: 17", 11, "bold")

        start_y = 0.790
        row_step = 0.0435
        height = 0.031
        for index, (paper, ours, status) in enumerate(ROWS):
            y = start_y - index * row_step
            face, edge, description = STYLE[status]
            box(ax, (0.075, y), 0.285, height, "#FFFFFF", "#AEB9C7",
                paper, 9.7, "normal")
            box(ax, (0.575, y), 0.335, height, face, edge, ours, 9.7,
                "bold" if status in {"direct", "renamed"} else "normal",
                color=edge if status != "direct" else "#2F7FC1")
            ax.add_patch(FancyArrowPatch(
                (0.37, y + height / 2), (0.565, y + height / 2),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.25,
                color=edge,
            ))
            ax.text(0.467, y + height / 2, description, ha="center", va="center",
                    fontsize=7.8, color=edge,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

        bottom = 0.105
        extra_height = 0.078
        box(ax, (0.075, bottom), 0.835, extra_height, "#F6F3FA", "#8C69AD",
            "Additional CISM labels without a one-to-one counterpart in the paper's 14-group contact scheme:\n"
            "Neuron  •  Unidentified\n"
            "(Neuron reflects a finer structural annotation; Unidentified was explicitly omitted from the paper's contact analysis.)",
            9.5, "normal", color="#5E4675")

        legend_y = bottom - 0.052
        ax.text(0.075, legend_y + 0.022, "Key", fontsize=10.5,
                fontweight="bold", color="#172033")
        legend_items = [
            ("#2F7FC1", "direct/renamed"),
            ("#D97706", "broader or partial"),
            ("#2E8B57", "split across CISM labels"),
            ("#BE3E42", "not represented separately"),
        ]
        x = 0.14
        for color, label in legend_items:
            ax.scatter([x], [legend_y + 0.025], s=55, color=color)
            ax.text(x + 0.014, legend_y + 0.025, label, va="center",
                    fontsize=8.8, color="#4B5870")
            x += 0.205

        foot_y = 0.018
        ax.text(
            0.045, foot_y,
            "Category-level correspondence based on published labels and the current CRC_cells.csv vocabulary. "
            "The repository does not preserve an exact cell-by-cell 28→14→17 crosswalk.",
            fontsize=8.5, color="#667085", va="bottom",
        )

        for extension in ("png", "pdf", "svg"):
            fig.savefig(STEM.with_suffix(f".{extension}"), dpi=300,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"Saved {STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
