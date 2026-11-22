"""Paired Wilcoxon tests of motif-mean density versus its point-null mean."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "analysis" / "reports" / "motif_density_edge_corrected"
RADIUS_SETTINGS = [(100, 50), (200, 100)]
COLORS = {"NN": "#5B7DB1", "NP": "#D88452"}


def paired_motif_means(observed: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    point_null = nulls[nulls["null_type"] == "random_point"].copy()
    keys = ["motif_id", "motif_association"]
    if observed.motif_id.nunique() != 19:
        raise ValueError("Expected exactly 19 observed motifs")
    if observed.instance_id.duplicated().any():
        raise ValueError("Observed instance IDs are not unique")
    expected = observed[keys + ["instance_id"]].drop_duplicates()
    actual = point_null.groupby("instance_id", observed=True).size()
    if set(expected.instance_id) != set(actual.index):
        raise ValueError("Point-null samples do not cover every observed instance")
    if actual.nunique() != 1:
        raise ValueError("Point-null instances have unequal sample counts")

    frames = []
    for radius_pixels, radius_microns in RADIUS_SETTINGS:
        metric = f"radius_{radius_pixels}_density"
        obs = observed.groupby(keys, observed=True)[metric].mean().reset_index(name="observed_mean")
        null = point_null.groupby(keys, observed=True)[metric].mean().reset_index(name="point_null_mean")
        paired = obs.merge(null, on=keys, how="outer", validate="one_to_one", indicator=True)
        if len(paired) != 19 or not paired["_merge"].eq("both").all():
            raise ValueError(f"Not every motif is paired for {metric}")
        paired = paired.drop(columns="_merge")
        paired["radius_pixels"] = radius_pixels
        paired["radius_microns"] = radius_microns
        paired["n_instances"] = paired.motif_id.map(observed.groupby("motif_id").size())
        paired["n_null_samples"] = paired.motif_id.map(point_null.groupby("motif_id").size())
        paired["delta_observed_minus_null"] = paired.observed_mean - paired.point_null_mean
        frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for i in range(len(values) - 1, -1, -1):
        running = min(running, values[order[i]] * len(values) / (i + 1))
        adjusted[order[i]] = running
    return pd.Series(adjusted, index=p_values.index)


def test_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius_pixels, radius_microns in RADIUS_SETTINGS:
        at_radius = pairs[pairs.radius_pixels == radius_pixels]
        for group in ("NN", "NP", "All motifs"):
            selected = at_radius if group == "All motifs" else at_radius[at_radius.motif_association == group]
            differences = selected.delta_observed_minus_null.to_numpy(dtype=float)
            nonzero = differences[differences != 0]
            if len(nonzero) < 2:
                raise ValueError(f"Too few nonzero paired differences for {group} at {radius_microns} microns")
            result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto")
            ranks = rankdata(np.abs(nonzero))
            positive_rank_sum = float(ranks[nonzero > 0].sum())
            negative_rank_sum = float(ranks[nonzero < 0].sum())
            rows.append({
                "radius_pixels": radius_pixels,
                "radius_microns": radius_microns,
                "group": group,
                "n_motif_pairs": len(selected),
                "n_positive_differences": int((differences > 0).sum()),
                "n_negative_differences": int((differences < 0).sum()),
                "observed_mean": float(selected.observed_mean.mean()),
                "point_null_mean": float(selected.point_null_mean.mean()),
                "mean_paired_difference": float(differences.mean()),
                "median_paired_difference": float(np.median(differences)),
                "wilcoxon_w": float(result.statistic),
                "wilcoxon_p_two_sided": float(result.pvalue),
                "rank_biserial": (positive_rank_sum - negative_rank_sum) / (positive_rank_sum + negative_rank_sum),
            })
    tests = pd.DataFrame(rows)
    tests["bh_q_across_six_tests"] = bh_adjust(tests.wilcoxon_p_two_sided)
    return tests


def draw_delta_plot(pairs: pd.DataFrame, tests: pd.DataFrame, radius_microns: int, output_dir: Path, pdf: PdfPages) -> None:
    selected = pairs[pairs.radius_microns == radius_microns]
    values = [selected.loc[selected.motif_association == group, "delta_observed_minus_null"].to_numpy() for group in ("NN", "NP")]
    fig, ax = plt.subplots(figsize=(7.2, 5.1), facecolor="white")
    boxes = ax.boxplot(
        values, tick_labels=["NN-associated motifs", "NP-associated motifs"],
        patch_artist=True, showfliers=False, whis=(0, 100), widths=0.45,
        boxprops={"edgecolor": "#666666"},
        medianprops={"color": "#555555", "linewidth": 1.3},
        whiskerprops={"color": "#999999"}, capprops={"color": "#999999"},
    )
    for box, group in zip(boxes["boxes"], ("NN", "NP")):
        box.set_facecolor(COLORS[group])
        box.set_alpha(0.92)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
    ax.grid(axis="y", color="#D0D0D0", alpha=0.5)
    ax.set_title(f"Motif-mean density minus matched point null | {radius_microns} micron radius")
    ax.set_ylabel("Paired difference (cells / micron²)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    at_radius = tests[tests.radius_microns == radius_microns].set_index("group")
    note = (
        f"Two-sided Wilcoxon: NN p={at_radius.loc['NN', 'wilcoxon_p_two_sided']:.3g}"
        f"   NP p={at_radius.loc['NP', 'wilcoxon_p_two_sided']:.3g}"
        f"   All 19 p={at_radius.loc['All motifs', 'wilcoxon_p_two_sided']:.3g}"
    )
    fig.text(0.5, 0.035, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(output_dir / f"paired_point_null_difference_{radius_microns}um.png", dpi=220)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.source_dir
    observed = pd.read_csv(args.source_dir / "motif_density_instance_metrics_edge_corrected.csv")
    nulls = pd.read_csv(args.source_dir / "motif_density_null_samples_edge_corrected.csv")
    pairs = paired_motif_means(observed, nulls)
    tests = test_groups(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / "motif_point_null_paired_means.csv", index=False)
    tests.to_csv(output_dir / "motif_point_null_wilcoxon_tests.csv", index=False)
    with PdfPages(output_dir / "motif_point_null_paired_differences.pdf") as pdf:
        for _, radius_microns in RADIUS_SETTINGS:
            draw_delta_plot(pairs, tests, radius_microns, output_dir, pdf)
    print(tests.to_string(index=False))
    print(f"Wrote paired Wilcoxon results to {output_dir}")


if __name__ == "__main__":
    main()
