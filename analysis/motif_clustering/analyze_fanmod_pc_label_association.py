#!/usr/bin/env python3
"""Associate individual motif-frequency PCs with NN/NP labels."""

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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = ANALYSIS_DIR / "outputs" / "fanmod_patient_motif_clustering_NN_NP_tutorial_artifact"
COLORS = {"NN": "#2F6C9E", "NP": "#D77A3D"}


def pc_ticks(k: int) -> list[int]:
    return sorted({1, k, *range(5, k + 1, 5)})


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranks = np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate((p_values[order] * len(p_values) / ranks)[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def analyze(k: int, n_permutations: int, input_dir: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(input_dir / "patient_motif_matrix_cism_frequency.csv", index_col=0)
    assignments = pd.read_csv(input_dir / "selected_cluster_assignments.csv").set_index("patient_id")
    if not matrix.index.is_unique or not matrix.index.isin(assignments.index).all():
        raise ValueError("Patient IDs in the motif matrix and label file do not align")
    labels = assignments.loc[matrix.index, "class_label"]
    if set(labels) != {"NN", "NP"} or matrix.shape[0] != 38:
        raise ValueError("Expected the 38 NN/NP patients from the melanoma tutorial artifact")
    if not 2 <= k <= min(matrix.shape[0] - 1, matrix.shape[1]):
        raise ValueError("k must be between 2 and 37")
    if n_permutations < 100:
        raise ValueError("Use at least 100 permutations")

    scaled = StandardScaler().fit_transform(np.log1p(matrix.to_numpy(dtype=float) * 1_000_000))
    pca = PCA(n_components=k, random_state=0)
    scores = pca.fit_transform(scaled)
    y = (labels.to_numpy() == "NP").astype(float)
    centered_y = y - y.mean()
    normalized_y = centered_y / np.linalg.norm(centered_y)
    normalized_scores = scores / np.linalg.norm(scores, axis=0)
    r = normalized_y @ normalized_scores

    # Each permutation keeps the correlated PC axes intact, allowing a max-statistic correction.
    rng = np.random.default_rng(0)
    permuted_y = np.array([rng.permutation(normalized_y) for _ in range(n_permutations)])
    null_abs_r = np.abs(permuted_y @ normalized_scores)
    abs_r = np.abs(r)
    permutation_p = (1 + (null_abs_r >= abs_r).sum(axis=0)) / (n_permutations + 1)
    familywise_p = (1 + (null_abs_r.max(axis=1)[:, None] >= abs_r).sum(axis=0)) / (n_permutations + 1)

    auc_np_higher = []
    rank_p = []
    for i in range(k):
        result = mannwhitneyu(scores[y == 1, i], scores[y == 0, i], alternative="two-sided")
        auc_np_higher.append(result.statistic / (sum(y == 1) * sum(y == 0)))
        rank_p.append(result.pvalue)
    auc_np_higher = np.asarray(auc_np_higher)
    rank_p = np.asarray(rank_p)
    explained = pca.explained_variance_ratio_
    summary = pd.DataFrame(
        {
            "PC": np.arange(1, k + 1),
            "variance_explained_pct": explained * 100,
            "cumulative_variance_pct": np.cumsum(explained) * 100,
            "NN_mean_score": scores[y == 0].mean(axis=0),
            "NP_mean_score": scores[y == 1].mean(axis=0),
            "point_biserial_r": r,
            "permutation_p_two_sided": permutation_p,
            "BH_q": bh_adjust(permutation_p),
            "maxT_familywise_p": familywise_p,
            "AUC_NP_higher": auc_np_higher,
            "AUC_best_direction_descriptive": np.maximum(auc_np_higher, 1 - auc_np_higher),
            "Mann_Whitney_p_two_sided": rank_p,
            "Mann_Whitney_BH_q": bh_adjust(rank_p),
        }
    )
    coordinates = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, k + 1)])
    coordinates.insert(0, "class_label", labels.to_numpy())
    coordinates.insert(0, "patient_id", matrix.index.to_numpy())
    return summary, coordinates


def figure_overview(summary: pd.DataFrame, scores: pd.DataFrame) -> plt.Figure:
    k = len(summary)
    fig = plt.figure(figsize=(11.7, 8.3))
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.96, top=0.79, bottom=0.10, hspace=0.36, wspace=0.27)
    ax_var = fig.add_subplot(grid[0, 0])
    ax_r = fig.add_subplot(grid[0, 1])
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_class = fig.add_subplot(grid[1, 1])

    fig.text(0.08, 0.955, "FANMOD+ motif PCs and NN/NP labels", fontsize=17, weight="bold")
    fig.text(0.08, 0.920, "38 patients (20 NN, 18 NP) | log1p(frequency x 10^6), feature standardization, PCA", fontsize=10)
    fig.text(0.08, 0.879, f"First {k} PCs explain {summary.cumulative_variance_pct.iloc[-1]:.1f}% of motif variance. Each PC is tested independently against class.", fontsize=10)
    if k >= 7:
        rank_q = summary.loc[summary.PC == 7, "Mann_Whitney_BH_q"].iloc[0]
        fig.text(0.08, 0.846, f"PC7 passes the mean-score test; rank-based BH q={rank_q:.3f}. NN and NP scores overlap.", fontsize=10)

    x = summary.PC.to_numpy()
    ax_var.bar(x, summary.variance_explained_pct, color="#637783")
    ax_var.set(xlabel="Principal component", ylabel="Variance explained (%)", title="Variance represented by each PC")
    ax_var.set_xlim(0.4, k + 0.6)
    ax_var.set_xticks(pc_ticks(k))
    ax_var.grid(axis="y", alpha=0.2)

    ax_r.bar(x, np.abs(summary.point_biserial_r), color=["#D77A3D" if q < .05 else "#8B979D" for q in summary.BH_q])
    ax_r.set(xlabel="Principal component", ylabel="|point-biserial r|", title="Association with NN/NP")
    ax_r.set_ylim(0, max(0.55, np.abs(summary.point_biserial_r).max() + .05))
    ax_r.set_xticks(pc_ticks(k))
    ax_r.grid(axis="y", alpha=0.2)
    for row in summary.itertuples():
        if row.BH_q < .05:
            ax_r.annotate(f"PC{row.PC}\nq={row.BH_q:.3f}", (row.PC, abs(row.point_biserial_r)),
                          xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

    strongest_two = summary.iloc[np.argsort(-np.abs(summary.point_biserial_r.to_numpy()))[:2]]
    x_pc, y_pc = [int(pc) for pc in strongest_two.PC]
    pc_columns = [f"PC{pc}" for pc in summary.PC]
    clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(scores[pc_columns])
    for cluster, marker in ((0, "o"), (1, "^")):
        for label in ("NN", "NP"):
            part = scores[(scores.class_label == label) & (clusters == cluster)]
            ax_scatter.scatter(part[f"PC{x_pc}"], part[f"PC{y_pc}"], s=53, alpha=.85,
                               marker=marker, color=COLORS[label], edgecolor="white", linewidth=.5)
    ax_scatter.set(xlabel=f"PC{x_pc} ({strongest_two.variance_explained_pct.iloc[0]:.1f}% variance)",
                   ylabel=f"PC{y_pc} ({strongest_two.variance_explained_pct.iloc[1]:.1f}% variance)",
                   title="Two most class-associated PCs (exploratory)")
    ax_scatter.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS["NN"], markersize=7, label="NN"),
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS["NP"], markersize=7, label="NP"),
        Line2D([0], [0], marker="o", linestyle="none", color="#555555", markersize=7, label="cluster 0"),
        Line2D([0], [0], marker="^", linestyle="none", color="#555555", markersize=7, label="cluster 1"),
    ], frameon=False, loc="upper right", ncol=2, fontsize=8)
    ax_scatter.axhline(0, lw=.5, color="#CCCCCC")
    ax_scatter.axvline(0, lw=.5, color="#CCCCCC")

    highlight = 7 if k >= 7 else int(summary.iloc[np.abs(summary.point_biserial_r).argmax()].PC)
    rng = np.random.default_rng(1)
    for position, label in enumerate(("NN", "NP")):
        values = scores.loc[scores.class_label == label, f"PC{highlight}"].to_numpy()
        ax_class.scatter(np.full(len(values), position) + rng.uniform(-.10, .10, len(values)), values,
                         color=COLORS[label], s=38, alpha=.8, edgecolor="white", linewidth=.5)
        ax_class.plot([position - .18, position + .18], [np.median(values)] * 2, color="#222222", lw=2)
    ax_class.set(xticks=[0, 1], xticklabels=["NN", "NP"], ylabel=f"PC{highlight} score", title=f"PC{highlight}: class overlap (bars = medians)")
    ax_class.grid(axis="y", alpha=.2)
    return fig


def association_colors(summary: pd.DataFrame) -> tuple[LinearSegmentedColormap, TwoSlopeNorm, np.ndarray]:
    cmap = LinearSegmentedColormap.from_list("nn_neutral_np", [COLORS["NN"], "#BEC6C8", COLORS["NP"]])
    limit = max(.5, np.ceil(np.abs(summary.point_biserial_r).max() * 10) / 10)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    return cmap, norm, cmap(norm(summary.point_biserial_r.to_numpy()))


def add_association_colorbar(fig: plt.Figure, cmap: LinearSegmentedColormap, norm: TwoSlopeNorm) -> None:
    colorbar_ax = fig.add_axes([.89, .27, .022, .47])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    ticks = np.linspace(norm.vmin, norm.vmax, 5)
    colorbar.set_ticks(ticks, labels=[f"{value:+.2f}" if value else "0.00" for value in ticks])
    colorbar.ax.tick_params(labelsize=8, length=0, pad=6)
    colorbar.outline.set_visible(False)
    fig.text(.89, .78, "r (NP=1)", fontsize=9, weight="bold")


def figure_variance(summary: pd.DataFrame, *, color_by_class: bool, sort_by_abs_r: bool = False) -> plt.Figure:
    """Plot per-PC motif variance, optionally encoding signed class association."""
    k = len(summary)
    ordered = summary.iloc[np.argsort(-np.abs(summary.point_biserial_r.to_numpy()))] if sort_by_abs_r else summary
    fig, ax = plt.subplots(figsize=(11.7, 6.2))
    fig.subplots_adjust(left=.09, right=.84 if color_by_class else .94, top=.77, bottom=.24 if sort_by_abs_r else .19)
    title = "Variance explained, ranked by NN/NP association" if sort_by_abs_r else "Variance explained by principal component"
    fig.text(.09, .95, title, fontsize=18, weight="bold")
    fig.text(.09, .89, f"First {k} PCs retain {summary.cumulative_variance_pct.iloc[-1]:.1f}% of the patient-by-motif matrix variance.", fontsize=10)
    if color_by_class:
        subtitle = "Bars ranked by |r|; height = motif variance; color = signed r with NN/NP labels." if sort_by_abs_r else "Bar height = motif variance; color = association with true NN/NP labels."
        fig.text(.09, .84, subtitle, fontsize=10)
        cmap, norm, bar_colors = association_colors(ordered)
    else:
        fig.text(.09, .84, "PCA was fitted without NN/NP labels; these percentages describe motif variability.", fontsize=10)
        bar_colors = "#858E92"

    x = np.arange(1, k + 1) if sort_by_abs_r else summary.PC.to_numpy()
    heights = ordered.variance_explained_pct.to_numpy()
    ax.bar(x, heights, width=.76, color=bar_colors, edgecolor="white", linewidth=.8, zorder=2)
    ax.set_xlim(.35, k + .65)
    ax.set_ylim(0, heights.max() * 1.19)
    if sort_by_abs_r:
        ax.set_xticks(x, [f"PC{int(pc)}" for pc in ordered.PC], rotation=55, ha="right")
    else:
        ax.set_xticks(pc_ticks(k))
    ax.set_xlabel("Principal component (ranked by |r|)" if sort_by_abs_r else "Principal component", labelpad=10)
    ax.set_ylabel("Motif variance explained (%)", labelpad=10)
    ax.grid(axis="y", color="#DDE3E5", linewidth=.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    if color_by_class:
        add_association_colorbar(fig, cmap, norm)
        fig.text(.09, .055, "Blue (negative r): NN higher; orange (positive r): NP higher. PC signs are arbitrary.", fontsize=9)
        fig.text(.09, .025, "Color intensity describes association strength; it is not a significance test.", fontsize=9)
    else:
        fig.text(.09, .06, "Variance explained is not the percentage of NN/NP label variation explained.", fontsize=9)
    return fig


def figure_r_values(summary: pd.DataFrame, *, color_by_class: bool = False) -> plt.Figure:
    """Plot absolute per-PC class correlations in PCA order."""
    k = len(summary)
    fig, ax = plt.subplots(figsize=(11.7, 6.2))
    fig.subplots_adjust(left=.09, right=.84 if color_by_class else .94, top=.77, bottom=.19)
    fig.text(.09, .95, "NN/NP association strength by principal component", fontsize=18, weight="bold")
    fig.text(.09, .89, f"First {k} PCs retain {summary.cumulative_variance_pct.iloc[-1]:.1f}% of the patient-by-motif matrix variance.", fontsize=10)
    subtitle = "Bar height = |r| with NN/NP labels; color = signed r (NP=1, NN=0)." if color_by_class else "Bar height = |point-biserial r| with true NN/NP labels."
    fig.text(.09, .84, subtitle, fontsize=10)

    x = summary.PC.to_numpy()
    values = summary.point_biserial_r.to_numpy()
    limit = max(.5, np.ceil(np.abs(values).max() * 10) / 10)
    if color_by_class:
        cmap, norm, bar_colors = association_colors(summary)
    else:
        bar_colors = "#858E92"
    ax.bar(x, np.abs(values), width=.76, color=bar_colors, edgecolor="white", linewidth=.8, zorder=2)
    ax.set_xlim(.35, k + .65)
    ax.set_ylim(0, limit * 1.13)
    ax.set_xticks(pc_ticks(k))
    ax.set_xlabel("Principal component", labelpad=10)
    ax.set_ylabel("|Point-biserial r|", labelpad=10)
    ax.grid(axis="y", color="#DDE3E5", linewidth=.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    if color_by_class:
        add_association_colorbar(fig, cmap, norm)
        fig.text(.09, .055, "Bar height shows strength; blue means NN higher, orange means NP higher. PC signs are arbitrary.", fontsize=9)
        fig.text(.09, .025, "Color intensity describes association strength; it is not a significance test.", fontsize=9)
    else:
        fig.text(.09, .06, "|r| shows association strength, not direction; PC signs are arbitrary.", fontsize=9)
    return fig


def figure_patient_scores(summary: pd.DataFrame, scores: pd.DataFrame) -> plt.Figure:
    """Show signed class association and patient overlap on the original PC axes."""
    ordered = summary.reindex(np.argsort(-np.abs(summary.point_biserial_r))).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    fig.subplots_adjust(left=.18, right=.72, top=.78, bottom=.12)
    fig.text(.08, .95, "Which disease state has the higher PC score?", fontsize=17, weight="bold")
    fig.text(.08, .912, "Blue = NN; orange = NP. Each dot is a patient; diamonds show class means.", fontsize=10)
    fig.text(.08, .88, "Scores keep their original PCA direction. Positive r means higher scores in NP; negative r means higher scores in NN.", fontsize=9.5)
    fig.text(.08, .848, "Rows are ranked by |r|. Shared blue/orange ranges show patient overlap.", fontsize=9.5)

    rng = np.random.default_rng(4)
    labels = scores.class_label.to_numpy()
    all_values = []
    for position, row in ordered.iterrows():
        pc = int(row.PC)
        raw = scores[f"PC{pc}"].to_numpy(dtype=float)
        standardized = raw / raw.std(ddof=0)
        all_values.extend(standardized)
        y = len(ordered) - 1 - position
        if row.BH_q < .05:
            ax.axhspan(y - .42, y + .42, color="#F0F2ED", zorder=0)
        for label in ("NN", "NP"):
            values = standardized[labels == label]
            jitter = rng.uniform(-.20, .20, len(values))
            ax.scatter(values, y + jitter, color=COLORS[label], alpha=.66, s=24,
                       edgecolor="white", linewidth=.3, zorder=2)
            ax.scatter(np.mean(values), y, color=COLORS[label], marker="D", s=77,
                       edgecolor="#242424", linewidth=.65, zorder=3)
        ax.text(1.025, y, f"{row.point_biserial_r:+.3f}",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5)
        ax.text(1.17, y, f"{row.BH_q:.3f}",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5,
                weight="bold" if row.BH_q < .05 else "normal")
        higher_class = "NP" if row.point_biserial_r > 0 else "NN"
        ax.text(1.32, y, higher_class, color=COLORS[higher_class],
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9,
                weight="bold")

    spread = max(3.5, np.ceil(max(abs(np.min(all_values)), abs(np.max(all_values))) * 2) / 2)
    ax.set_xlim(-spread - .2, spread + .2)
    ax.set_ylim(-.6, len(ordered) - .4)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([f"PC{int(row.PC)}  ({row.variance_explained_pct:.1f}%)"
                        for row in ordered.iloc[::-1].itertuples()], fontsize=8.5)
    ax.axvline(0, color="#626E73", linewidth=.9, zorder=1)
    ax.grid(axis="x", color="#DCE1E2", linewidth=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("PC score (standard deviations; higher scores to the right)", fontsize=10)
    ax.text(1.025, len(ordered) - .30, "r", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=9, weight="bold")
    ax.text(1.17, len(ordered) - .30, "BH q", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=9, weight="bold")
    ax.text(1.32, len(ordered) - .30, "Higher", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=9, weight="bold")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS["NN"], markersize=6, label="NN patient"),
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS["NP"], markersize=6, label="NP patient"),
        Line2D([0], [0], marker="D", linestyle="none", color="#555555", markersize=7, label="Class mean"),
    ], loc="upper center", bbox_to_anchor=(.5, 1.065), ncol=3, frameon=False, fontsize=9)
    fig.text(.08, .052, "BH q corrects the mean-score permutation tests across all shown PCs; pale shading marks q<0.05.", fontsize=8.5)
    fig.text(.08, .028, "A PC's sign is arbitrary: flipping it reverses the higher-score class but leaves separation unchanged.", fontsize=8.5)
    return fig


def figure_table(summary: pd.DataFrame, n_permutations: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    fig.text(.06, .95, "Per-PC association results", fontsize=16, weight="bold")
    fig.text(.06, .915, "Permutation tests are two-sided; BH controls FDR across the tested PCs; maxT controls familywise error.", fontsize=9)
    fig.text(.06, .89, "AUC is oriented to the better direction for description only. PCA axis signs are arbitrary.", fontsize=9)
    data = [[f"PC{int(row.PC)}", f"{row.variance_explained_pct:.2f}", f"{row.cumulative_variance_pct:.2f}",
             f"{row.point_biserial_r:+.3f}", f"{row.AUC_best_direction_descriptive:.3f}",
             f"{row.permutation_p_two_sided:.4f}", f"{row.BH_q:.4f}", f"{row.maxT_familywise_p:.4f}"]
            for row in summary.itertuples()]
    table = ax.table(cellText=data,
                     colLabels=["PC", "Var %", "Cum %", "r", "AUC*", "Perm p", "BH q", "maxT p"],
                     cellLoc="center", colLoc="center", bbox=[.05, .07, .90, .78])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE1E3")
        cell.set_linewidth(.4)
        if row == 0:
            cell.set_facecolor("#E5E9EB")
            cell.set_text_props(weight="bold")
        elif summary.iloc[row - 1].BH_q < .05:
            cell.set_facecolor("#FBEAD9")
        elif row % 2 == 0:
            cell.set_facecolor("#F6F7F7")
    fig.text(.06, .045, f"{n_permutations:,} fixed-seed label permutations. *Higher AUC after choosing the direction from these same data; not validation accuracy.", fontsize=8)
    return fig


def write_report(summary: pd.DataFrame, n_permutations: int, output: Path, n_motifs: int) -> None:
    strongest = summary.iloc[np.abs(summary.point_biserial_r).argmax()]
    significant = summary.loc[summary.maxT_familywise_p < .05, "PC"].tolist()
    lines = [
        "# Individual principal components and NN/NP class",
        "",
        "## Data and method",
        "",
        f"The melanoma tutorial FANMOD+ patient-by-motif frequency matrix contains 38 patients (20 NN, 18 NP) and {n_motifs:,} motifs. "
        "We apply log1p(frequency x 10^6), standardize each motif across patients, and fit PCA without class labels. "
        f"The first {len(summary)} PCs are tested one at a time against NN/NP; together they account for "
        f"{summary.cumulative_variance_pct.iloc[-1]:.2f}% of matrix variance. The comparison statistic is point-biserial r. "
        f"Two-sided p-values use {n_permutations:,} label permutations (seed 0). BH q-values control false discovery rate across "
        "the tested PCs; maxT p-values control familywise error over those same PCs. AUC is oriented post hoc to the "
        "higher-separation direction and is descriptive, not a cross-validated prediction estimate.",
        "",
        "## Findings",
        "",
        f"PC1 explains {summary.variance_explained_pct.iloc[0]:.2f}% and PC2 {summary.variance_explained_pct.iloc[1]:.2f}% "
        f"(combined {summary.cumulative_variance_pct.iloc[1]:.2f}%). "
        f"Their association statistics are r={summary.point_biserial_r.iloc[0]:+.3f} "
        f"(BH q={summary.BH_q.iloc[0]:.3f}) and r={summary.point_biserial_r.iloc[1]:+.3f} "
        f"(BH q={summary.BH_q.iloc[1]:.3f}), respectively.",
        "",
        f"The strongest single-PC association is PC{int(strongest.PC)}, which explains "
        f"{strongest.variance_explained_pct:.2f}% of motif variance: r={strongest.point_biserial_r:+.3f}, "
        f"permutation p={strongest.permutation_p_two_sided:.4f}, BH q={strongest.BH_q:.4f}, "
        f"maxT p={strongest.maxT_familywise_p:.4f}, and descriptive best-direction AUC="
        f"{strongest.AUC_best_direction_descriptive:.3f}. Its NN and NP scores overlap, so it does not cleanly classify patients. "
        f"A rank-based Mann-Whitney sensitivity check gives p={strongest.Mann_Whitney_p_two_sided:.4f} and "
        f"BH q={strongest.Mann_Whitney_BH_q:.4f}; this check does not pass the 0.05 FDR threshold.",
        "",
        "At maxT familywise p<0.05, associated PCs: " + (", ".join(f"PC{int(pc)}" for pc in significant) if significant else "none") + ".",
        "",
        "**Interpretation:** PC1 and PC2 do not show compelling independent NN/NP association after correction. "
        "There is evidence of an association on a later PC, so a blanket claim of no class signal would be inaccurate. "
        "The overlapping distributions and in-sample AUC do not establish validated predictive discrimination. "
        "Variance explained measures motif variability, not label variability; the percentages should not be described as "
        "variance in NN/NP explained.",
        "",
        "The overview scatterplot uses the two PCs with the largest absolute class correlations in this dataset. "
        "Blue/orange indicate NN/NP, while circles/triangles indicate the two k-means clusters computed from all tested PC scores. "
        "The axes were selected after seeing the labels, so the scatterplot is exploratory and does not validate discrimination.",
        "",
        "## Reading the patient-level plot",
        "",
        "Each row is one PC, ranked by the magnitude of its class correlation. Blue dots are NN patients and orange dots are NP patients; "
        "diamonds are the two class means. The horizontal score is in units of that PC's standard deviation. "
        "The original PCA score direction is retained: positive r and the right-hand 'Higher' column mean NP has a higher mean score; "
        "negative r means NN has a higher mean score. PC signs are arbitrary, so flipping a PC would reverse this direction "
        "without changing the strength of class separation. The farther apart the diamonds, the larger the absolute "
        "point-biserial correlation; the spread of dots shows "
        "how much the individual patients still overlap. The percentage beside each PC is motif variance explained, not class "
        "variance explained. This is a visual summary of PC scores, not a SHAP analysis or a trained classifier.",
        "",
        "See `pc_label_association_results.csv` for every tested PC and `pc_label_association_scores.csv` for patient-level scores.",
        "The PC-ordered variance charts show the same bar heights; the colored version encodes signed NN/NP association separately. "
        "A second colored variance chart ranks PCs by |r|. The two |r| charts show association strength in PC order; "
        "the colored one also encodes the sign of r. Grey bars do not encode direction.",
    ]
    (output / "pc_label_association_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=20, help="Number of leading PCs (default: 20, matching PCA-20 clustering)")
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--input-dir", type=Path, default=ROOT, help="Clustering output directory containing the patient-motif matrix.")
    args = parser.parse_args()
    output = args.input_dir / "pc_label_association"
    output.mkdir(parents=True, exist_ok=True)
    summary, scores = analyze(args.k, args.permutations, args.input_dir)
    summary.to_csv(output / "pc_label_association_results.csv", index=False, float_format="%.8g")
    scores.to_csv(output / "pc_label_association_scores.csv", index=False, float_format="%.8g")
    n_motifs = pd.read_csv(args.input_dir / "patient_motif_matrix_cism_frequency.csv", nrows=0).shape[1] - 1
    write_report(summary, args.permutations, output, n_motifs)
    overview = figure_overview(summary, scores)
    variance_plain = figure_variance(summary, color_by_class=False)
    variance_class = figure_variance(summary, color_by_class=True)
    variance_class_ranked = figure_variance(summary, color_by_class=True, sort_by_abs_r=True)
    r_values = figure_r_values(summary)
    r_values_class = figure_r_values(summary, color_by_class=True)
    patients = figure_patient_scores(summary, scores)
    table = figure_table(summary, args.permutations)
    with PdfPages(output / "pc_label_association_report.pdf") as pdf:
        pdf.savefig(overview)
        pdf.savefig(patients)
        pdf.savefig(table)
    overview.savefig(output / "pc_label_association_overview.png", dpi=220)
    for name, figure in (("pc_variance_explained", variance_plain),
                         ("pc_variance_class_association", variance_class),
                         ("pc_variance_class_association_ranked_by_abs_r", variance_class_ranked),
                         ("pc_r_values", r_values),
                         ("pc_r_values_class_association", r_values_class)):
        figure.savefig(output / f"{name}.png", dpi=220)
        figure.savefig(output / f"{name}.pdf")
    with PdfPages(output / "pc_variance_figures.pdf") as pdf:
        pdf.savefig(variance_plain)
        pdf.savefig(variance_class)
        pdf.savefig(variance_class_ranked)
        pdf.savefig(r_values)
        pdf.savefig(r_values_class)
    patients.savefig(output / "pc_label_association_patient_scores.png", dpi=220)
    patients.savefig(output / "pc_label_association_direction.png", dpi=220)
    patients.savefig(output / "pc_label_association_direction.pdf")
    table.savefig(output / "pc_label_association_table.png", dpi=180)
    plt.close(overview)
    plt.close(variance_plain)
    plt.close(variance_class)
    plt.close(variance_class_ranked)
    plt.close(r_values)
    plt.close(r_values_class)
    plt.close(patients)
    plt.close(table)
    print(output / "pc_label_association_report.pdf")
    print(summary.loc[summary.maxT_familywise_p < .05, ["PC", "variance_explained_pct", "point_biserial_r", "maxT_familywise_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
