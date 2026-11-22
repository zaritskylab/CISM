#!/usr/bin/env python3
"""Build a PDF report for the SMORE/CISM4 stringency analysis."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import os
import textwrap
from pathlib import Path

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = _REPO_ROOT
OUT_DIR = ROOT / "analysis" / "outputs" / "stringency_clean"
FIG_DIR = OUT_DIR / "figures"
PDF_PATH = OUT_DIR / "smore_cism_stringency_report.pdf"

PLOTS = [
    (
        "SMORE and CISM4 Stringency Comparison",
        FIG_DIR / "seed_smore_cism_stringency_side_by_side.png",
    ),
    (
        "SMORE Per-Patient Discriminative Motifs, Stringency 0.30",
        FIG_DIR / "smore_discriminative_motifs_per_patient_stringency_0p3.png",
    ),
    (
        "SMORE Per-Patient Discriminative Motifs, Stringency 0.35",
        FIG_DIR / "smore_discriminative_motifs_per_patient_stringency_0p35.png",
    ),
]


def add_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold", va="top")

    y = 0.87
    for heading, body in sections:
        fig.text(0.08, y, heading, fontsize=12, weight="bold", va="top")
        y -= 0.03
        wrapped = []
        for paragraph in body.strip().split("\n"):
            if paragraph.strip():
                wrapped.extend(textwrap.wrap(paragraph.strip(), width=92))
            else:
                wrapped.append("")
        for line in wrapped:
            fig.text(0.08, y, line, fontsize=9.5, va="top")
            y -= 0.019
        y -= 0.02

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_plot_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = Image.open(image_path)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.04, 0.965, title, fontsize=15, weight="bold", va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(PDF_PATH) as pdf:
        add_text_page(
            pdf,
            "SMORE vs CISM4 Melanoma Stringency Analysis",
            [
                (
                    "Scope",
                    "Comparison of SMORE seed-based motif results with CISM 4-node melanoma motif results. "
                    "The report covers discriminative motif counts vs stringency, LOOCV AUC vs stringency, "
                    "per-patient SMORE motif counts, and interpretation of failed SMORE LOOCV folds.",
                ),
                (
                    "Methodological Notes",
                    "SMORE motif identity is handled using exact seed values, not motifNumber. motifNumber behaves "
                    "like a local FOV/rank identifier, whereas seed maps exactly to ordered motif composition. "
                    "The SMORE-to-CISM adapter uses id_col='seed' and canonicalize_id=False. The adapter still "
                    "sets p_value_default=0.01 only as a compatibility placeholder so CISM does not discard "
                    "adapted SMORE rows via its p_value < 0.05 filter.",
                ),
                (
                    "Main Results",
                    "At stringency 0.30, CISM4 has 37 NN and 39 NP discriminative motifs; SMORE has 56 NN and "
                    "11 NP motifs. At stringency 0.40, CISM4 has 5 NN and 4 NP motifs; SMORE has 7 NN and 5 NP "
                    "motifs. At stringency 0.50, CISM4 has 1 NN and 3 NP motifs; SMORE has 0 NN and 1 NP motif.",
                ),
                (
                    "AUC Summary",
                    "Seed-based SMORE AUC values from the CISM-adapter run are 0.6265 at stringency 0.20, "
                    "0.6853 at 0.25, and 0.1882 at 0.30. CISM4 AUC values on the clean grid are 0.6667 at "
                    "0.20, 0.6708 at 0.30, 0.5917 at 0.40, and 0.8639 at 0.50.",
                ),
            ],
        )

        for title, image_path in PLOTS:
            add_plot_page(pdf, title, image_path)

        add_text_page(
            pdf,
            "Interpretation and Failed-Fold Notes",
            [
                (
                    "Per-Patient SMORE Counts",
                    "At stringency 0.30, NN patients have mean 18.45 passing motifs per patient "
                    "(median 18, range 3-35), while NP patients have mean 4.35 (median 4, range 0-10). "
                    "At stringency 0.35, NN patients have mean 9.15 (median 9.5, range 2-16), while NP "
                    "patients remain at mean 4.35 (median 4, range 0-10). Patient 5 has zero passing "
                    "motifs at both thresholds and is shown explicitly in the plots.",
                ),
                (
                    "Why Folds Fail",
                    "The per-patient bar plots use the full-data discriminative seed list. LOOCV failure is "
                    "different: after holding out a patient, CISM reselects features from the remaining training "
                    "set. At stringency 0.30 there are no empty-feature folds. At 0.35, five NP-held-out folds "
                    "collapse to zero selected features: SMORE12, SMORE13, SMORE14, SMORE15, and SMORE20.",
                ),
                (
                    "Caution",
                    "CISM uses patient_percentage > shared_percentage, not >=. Borderline features can therefore "
                    "drop out when a single patient is removed. The SMORE AUC comparison should mention that it "
                    "uses a seed-ID CISM adapter and becomes fold-sparse at higher stringency.",
                ),
                (
                    "Generated Assets",
                    "The Markdown report, CSVs, and figures are stored under analysis/outputs/stringency_clean. "
                    "The primary side-by-side figure is figures/seed_smore_cism_stringency_side_by_side.pdf.",
                ),
            ],
        )

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
