#!/usr/bin/env python3
"""Plot the melanoma fixed-0.46 rerun with the CRC/TNBC histogram style."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "fixed_046_reassessment_historical" / "all_fixed_046_aucs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "report" / "fixed_046_reassessment_panels",
    )
    args = parser.parse_args()

    trials = pd.read_csv(args.results)
    original = trials.loc[trials.seed.isna()]
    shuffled = trials.loc[trials.seed.notna()].copy()
    if len(original) != 1 or shuffled.empty or shuffled.fixed_auc.isna().any():
        raise ValueError("Expected one original and complete shuffled AUCs")
    if shuffled.seed.nunique() != len(shuffled):
        raise ValueError("Permutation seeds must be unique")
    reference = float(original.fixed_auc.iloc[0])
    exceedances = int(shuffled.fixed_auc.ge(reference).sum())
    p_value = (exceedances + 1) / (len(shuffled) + 1)
    affected = int(shuffled.no_feature_folds.gt(0).sum())

    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(5.5, 3.7))
        ax.hist(shuffled.fixed_auc, bins=np.linspace(0, 1, 21),
                color="#24558c", edgecolor="white", linewidth=0.6)
        ax.axvline(reference, color="#df5948", lw=1.5,
                   label=f"Original = {reference:.3f}")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlim=(0, 1), xlabel="AUC at fixed stringency 0.46",
               ylabel="Number of shuffles")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(
            f"Permutation p = {p_value:.4g}\n"
            f"{affected}/{len(shuffled)} shuffles with no-motif folds",
            fontsize=10,
        )
        fig.tight_layout()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"panel_A_fixed_046_crc_style.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Original AUC: {reference:.6f}; shuffles: {len(shuffled)}; "
          f"exceedances: {exceedances}; permutation p: {p_value:.6f}; "
          f"with no-motif folds: {affected}")


if __name__ == "__main__":
    main()
