from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


ROOT = _REPO_ROOT
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "melanoma_tutorials/melanoma_tutorials/runtime"
    / "absent_discriminative_cism_nn_np_3node/figures"
)
ANTI_COLOR = "#3979a8"
MOTIF_COLOR = "#c85058"


def draw(output: Path, alpha: float) -> None:
    cutoff = float(norm.ppf(1 - alpha / 2))
    x_max = max(4.2, cutoff + 1.5)
    x = np.linspace(-x_max, x_max, 1400)
    density = norm.pdf(x)

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.fill_between(x, density, where=x <= -cutoff, color=ANTI_COLOR, alpha=0.88)
    ax.fill_between(x, density, where=x >= cutoff, color=MOTIF_COLOR, alpha=0.88)
    ax.plot(x, density, color="#363d41", linewidth=2.1)
    for boundary in (-cutoff, cutoff):
        ax.plot([boundary, boundary], [0, norm.pdf(boundary)],
                color="#687076", linewidth=1.0, linestyle=(0, (3, 3)))

    ax.text(-3.0, 0.19, "Anti-motifs", ha="center", va="center", color=ANTI_COLOR,
            fontsize=13, fontweight="bold")
    ax.text(3.0, 0.19, "Motifs", ha="center", va="center", color=MOTIF_COLOR,
            fontsize=13, fontweight="bold")
    ax.text(-2.9, 0.155, "Under-represented", ha="center", color="#364c5b", fontsize=9)
    ax.text(2.9, 0.155, "Over-represented", ha="center", color="#6c3e43", fontsize=9)

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(0, 0.45)
    ax.set_xticks([-cutoff, 0, cutoff])
    ax.set_xticklabels([f"-{cutoff:.2f}", "0", f"+{cutoff:.2f}"])
    ax.set_yticks([])
    ax.set_xlabel("Z-score: observed motif frequency vs randomized expectation", labelpad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#535b60")
    ax.tick_params(axis="x", length=5, color="#535b60")
    ax.text(0, 0.425, f"Two-tailed normal test  |  p < {alpha:g}",
            ha="center", va="bottom", fontsize=11, color="#3a4348")
    fig.tight_layout(pad=1.3)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw two-tailed motif significance schematic.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "two_tailed_motif_significance_schematic.png"
    draw(output, args.alpha)
    print(output)
    print(output.with_suffix(".pdf"))
    print(output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
