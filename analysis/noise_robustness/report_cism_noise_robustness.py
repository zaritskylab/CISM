#!/usr/bin/env python3
"""Plots and PDF/Markdown reporting for the complete CISM noise experiment."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from analysis.noise_robustness.motif_noise_summary import CELL_ORDER, dataframe_to_markdown
from analysis.noise_robustness.create_motif_noise_pdf_report import add_table_page


def add_image_page(pdf, title, image_path, caption):
    image = plt.imread(image_path)
    portrait = image.shape[0] > image.shape[1] * .85
    fig = plt.figure(figsize=(8.3, 11.7) if portrait else (11.7, 8.3))
    fig.text(.045, .965, "\n".join(textwrap.wrap(title, width=67 if portrait else 92)),
             fontsize=15, weight="bold", va="top")
    ax = fig.add_axes([.035, .11, .93, .77])
    ax.imshow(image)
    ax.axis("off")
    fig.text(.045, .07, "\n".join(textwrap.wrap(caption, width=100 if portrait else 140)),
             fontsize=8.5, va="top")
    pdf.savefig(fig)
    plt.close(fig)


COLORS = ["#555555", "#238b45", "#74a9cf", "#225ea8", "#d95f0e", "#af2763"]


def condition_order(table):
    return table[["condition", "noise_rate"]].drop_duplicates().sort_values(["noise_rate", "condition"])["condition"].tolist()


def boxplot(ax, table, value, order, ylabel, show_n=False):
    groups = [table.loc[table["condition"] == cond, value].dropna().to_numpy() for cond in order]
    ticks = [c.replace("noise_", "") + (f"\nn={len(group)}" if show_n else "") for c, group in zip(order, groups)]
    boxes = ax.boxplot(groups, tick_labels=ticks,
                       patch_artist=True, showfliers=False, widths=.55,
                       medianprops={"color": "black", "linewidth": 1.4})
    for patch, color in zip(boxes["boxes"], (COLORS * (len(order) // len(COLORS) + 1))):
        patch.set_facecolor(color)
        patch.set_alpha(.55)
    rng = np.random.default_rng(0)
    for i, group in enumerate(groups, 1):
        if len(group):
            if len(group) > 160:
                group = group[np.linspace(0, len(group) - 1, 160).astype(int)]
            ax.scatter(i + rng.uniform(-.13, .13, len(group)), group, s=5, alpha=.23, color="black", rasterized=True)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=28)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def save_figure(fig, path):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics, order, output_dir):
    optimal = metrics[metrics["mode"] == "optimal"]
    for value, title, ylabel, name in [
        ("stringency", "Optimal Stringency by Noise Condition", "Stringency", "optimal_stringency_by_noise.png"),
        ("auc", "Best LOOCV AUC by Noise Condition", "Tuned LOOCV AUC", "auc_by_noise.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        boxplot(ax, optimal, value, order, ylabel)
        ax.set_title(title)
        if value == "auc":
            ax.set_ylim(0, 1.02)
            ax.axhline(.5, color="#777777", linestyle="--", linewidth=.8)
        save_figure(fig, output_dir / name)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, mode in zip(axes, ["fixed", "optimal"]):
        boxplot(ax, metrics[metrics["mode"] == mode], "n_motifs", order, "Fold-union motif count")
        ax.set_title("Fixed stringency 0.46" if mode == "fixed" else "Run-specific optimal stringency")
    save_figure(fig, output_dir / "motif_counts_fixed_vs_optimal.png")


def plot_curves(curves, order, output_dir):
    fig, (ax, eligible_ax) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
    for condition, color in zip(order, COLORS * (len(order) // len(COLORS) + 1)):
        local = curves[curves["condition"] == condition].groupby("stringency")["auc"].agg(["mean", "std", "count", "size"])
        ax.plot(local.index, local["mean"], marker=".", label=condition, color=color)
        ax.fill_between(local.index, local["mean"] - local["std"].fillna(0),
                        local["mean"] + local["std"].fillna(0), alpha=.1, color=color)
        eligible_ax.plot(local.index, 100 * local["count"] / local["size"], color=color, marker=".")
    ax.set(ylabel="LOOCV AUC", ylim=(0, 1.02), title="AUC vs Stringency: Mean and Run SD")
    ax.axhline(.5, color="#777777", linestyle="--", linewidth=.8)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    eligible_ax.set(xlabel="Stringency", ylabel="RF-eligible trials (%)", ylim=(-2, 102),
                    xlim=(float(curves["stringency"].min()), float(curves["stringency"].max())))
    eligible_ax.grid(alpha=.2)
    save_figure(fig, output_dir / "auc_vs_stringency_by_noise.png")


def plot_distances(table, kind, order, output_dir, within=False, common=False, centroids=None):
    local = table[table["kind"] == kind]
    if not within:
        local = local[local["reference"] == ("raw_fixed_0.46" if common else "raw_same_mode")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for row, mode in enumerate(["fixed", "optimal"]):
        for col, label in enumerate(["NN", "NP"]):
            ax = axes[row, col]
            selected = local[(local["mode"] == mode) & (local["label"] == label)]
            boxplot(ax, selected, "tv", order, "Total variation distance", show_n=True)
            if not within and centroids is not None:
                ref_name = "raw_fixed_0.46" if common else "raw_same_mode"
                centroid = centroids[(centroids["kind"] == kind) & (centroids["mode"] == mode)
                                     & (centroids["label"] == label) & (centroids["reference"] == ref_name)]
                means = []
                for cond in order:
                    value = centroid.loc[centroid["condition"] == cond, "tv"]
                    means.append(value.iloc[0] if len(value) else np.nan)
                ax.scatter(np.arange(1, len(order) + 1), means, marker="D", color="#af2763", s=28,
                           zorder=5, label="Distance of condition mean")
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc="upper left")
            ax.set_ylim(-.015, 1.015)
            setting = "fixed 0.46" if mode == "fixed" else "optimal"
            ax.set_title(f"{label} | {setting}")
    subject = "Cell-Type" if kind == "cell" else "Pairwise"
    target = "Between Trials Within Each Condition" if within else "to Original Raw Distribution"
    if common:
        target += " (Common Fixed Reference)"
    fig.suptitle(f"{subject} Distribution Distances {target}", fontsize=14)
    prefix = "within_condition" if within else ("distance_common_reference" if common else "distance_to_original")
    name = f"{kind}_{prefix}.png"
    save_figure(fig, output_dir / name)
    return name


def summarize_distributions(distributions, results, reference_run):
    reference = next(r for r in results if r["condition"] == "raw" and r["run_id"] == reference_run)
    rows, centroid_rows = [], []
    for (condition, noise, mode, label, kind), group in distributions.groupby(["condition", "noise_rate", "mode", "label", "kind"]):
        valid = [np.asarray(v) for v in group["vector"] if v is not None]
        if not valid:
            continue
        vectors = np.stack(valid)
        mean = vectors.mean(axis=0)
        sd = vectors.std(axis=0, ddof=1) if len(vectors) > 1 else np.full_like(mean, np.nan)
        names = CELL_ORDER if kind == "cell" else [f"{a} / {b}" for i, a in enumerate(CELL_ORDER) for b in CELL_ORDER[i:]]
        for name, avg, std in zip(names, mean, sd):
            rows.append(dict(condition=condition, noise_rate=noise, mode=mode, label=label, kind=kind,
                             component=name, mean_probability=avg, sd_probability=std, n_valid_runs=len(valid)))
        for ref_mode, ref_name in [(mode, "raw_same_mode"), ("fixed", "raw_fixed_0.46")]:
            ref = reference["selections"][ref_mode]["distributions"][label][kind]
            distance = .5 * np.abs(mean - np.asarray(ref)).sum() if ref is not None else np.nan
            centroid_rows.append(dict(condition=condition, noise_rate=noise, mode=mode, label=label,
                                      kind=kind, reference=ref_name, tv=distance, n_valid_runs=len(valid)))
    return pd.DataFrame(rows), pd.DataFrame(centroid_rows)


def plot_undefined_distributions(summary, order, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    x = np.arange(len(order))
    peak = float(summary["fraction_undefined"].max()) * 100
    for ax, label in zip(axes, ["NN", "NP"]):
        for mode, offset, color in [("fixed", -.18, "#5b9bd5"), ("optimal", .18, "#b66378")]:
            local = summary[(summary["kind"] == "cell") & (summary["label"] == label) & (summary["mode"] == mode)]
            values = [float(local.loc[local["condition"] == c, "fraction_undefined"].iloc[0]) * 100 for c in order]
            ax.bar(x + offset, values, width=.34, color=color, label=mode)
            for ix, value in enumerate(values):
                if value:
                    ax.text(ix + offset, value + max(peak * .025, .3), f"{value:.1f}", fontsize=7, ha="center")
        ax.set_xticks(x, [c.replace("noise_", "") for c in order], rotation=28)
        ax.set_title(label)
        ax.set_ylabel("Trials with undefined distribution (%)")
        ax.set_ylim(0, max(10, peak * 1.2))
        ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.suptitle("Zero Motif-Induced Mass in a Disease Class")
    save_figure(fig, output_dir / "undefined_distribution_fraction_by_noise.png")


def plot_mean_distributions(distributions, kind, order, output_dir):
    local = distributions[distributions["kind"] == kind]
    matrices = {}
    for mode in ["fixed", "optimal"]:
        for label in ["NN", "NP"]:
            columns = []
            for cond in order:
                vectors = local[(local["mode"] == mode) & (local["label"] == label) & (local["condition"] == cond)]["vector"]
                valid = [np.asarray(v) for v in vectors if v is not None]
                columns.append(np.mean(valid, axis=0) if valid else np.full(15 if kind == "cell" else 120, np.nan))
            matrices[(mode, label)] = np.column_stack(columns)
    if kind == "cell":
        names, keep = CELL_ORDER, np.arange(len(CELL_ORDER))
    else:
        pairs = [(a, b) for i, a in enumerate(CELL_ORDER) for b in CELL_ORDER[i:]]
        keep = np.argsort(-np.nanmean(np.stack(list(matrices.values())), axis=(0, 2)))[:20]
        names = [f"{pairs[ix][0]} / {pairs[ix][1]}" for ix in keep]
    vmax = max(float(np.nanmax(mat[keep])) for mat in matrices.values())
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for row, mode in enumerate(["fixed", "optimal"]):
        for col, label in enumerate(["NN", "NP"]):
            ax = axes[row, col]
            im = ax.imshow(matrices[(mode, label)][keep], aspect="auto", vmin=0, vmax=vmax,
                           cmap="Blues" if label == "NN" else "Reds")
            ax.set_xticks(np.arange(len(order)), [c.replace("noise_", "") for c in order], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(names)), names, fontsize=8)
            ax.set_title(f"{label} | {'fixed 0.46' if mode == 'fixed' else 'optimal'}")
            fig.colorbar(im, ax=ax, fraction=.045)
    fig.suptitle("Mean Motif-Induced " + ("Cell-Type Fractions" if kind == "cell" else "Pair Probabilities (Top 20)"))
    save_figure(fig, output_dir / f"average_{kind}_distributions_all_noise.png")


def mean_composition(summary, condition, mode, label, kind):
    local = summary[(summary["condition"] == condition) & (summary["mode"] == mode)
                    & (summary["label"] == label) & (summary["kind"] == kind)]
    names = CELL_ORDER if kind == "cell" else [
        f"{a} / {b}" for i, a in enumerate(CELL_ORDER) for b in CELL_ORDER[i:]]
    values = local.set_index("component")["mean_probability"].reindex(names).to_numpy()
    counts = local["n_valid_runs"].unique()
    if len(counts) != 1 or not np.isfinite(values).all() or not np.isclose(values.sum(), 1):
        raise ValueError(f"Incomplete mean composition: {condition}/{mode}/{label}/{kind}")
    return values, int(counts[0])


def pair_probability_matrix(values):
    """Display each undirected pair once; the unrepresented triangle stays masked."""
    values = np.asarray(values, dtype=float)
    indices = np.triu_indices(len(CELL_ORDER))
    if values.shape != indices[0].shape:
        raise ValueError("Expected all 120 undirected pair probabilities")
    matrix = np.full((len(CELL_ORDER), len(CELL_ORDER)), np.nan)
    matrix[indices] = values
    return matrix


def plot_faceted_mean_distributions(summary, order, output_dir):
    titles = {}
    for condition in order:
        rate = float(summary.loc[summary["condition"] == condition, "noise_rate"].iloc[0])
        titles[condition] = "Original (0% noise)" if condition == "raw" else f"{100 * rate:g}% noise"
        if condition.endswith("_matrix"):
            titles[condition] += " (matrix-labelled)"
    ncols = min(3, len(order))
    nrows = (len(order) + ncols - 1) // ncols
    cell_max = float(summary.loc[summary["kind"] == "cell", "mean_probability"].max()) * 1.08
    pair_max = float(summary.loc[summary["kind"] == "pair", "mean_probability"].max())
    captions = {}
    for mode in ["fixed", "optimal"]:
        setting = "Fixed stringency 0.46" if mode == "fixed" else "Run-specific optimal stringency"
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.4 * nrows),
                                 squeeze=False, layout="constrained")
        for ax, condition in zip(axes.flat, order):
            counts = {}
            for label, offset, color in [("NN", -.19, "#5b9bd5"), ("NP", .19, "#c95050")]:
                values, counts[label] = mean_composition(summary, condition, mode, label, "cell")
                ax.barh(np.arange(len(CELL_ORDER)) + offset, values, height=.36, color=color, label=label)
            count_label = f"n={counts['NN']} runs/class" if counts["NN"] == counts["NP"] else (
                f"NN n={counts['NN']}; NP n={counts['NP']}")
            ax.set_title(f"{titles[condition]}\n{count_label}", fontsize=11)
            ax.set_yticks(np.arange(len(CELL_ORDER)), CELL_ORDER, fontsize=9)
            ax.set_ylim(len(CELL_ORDER) - .5, -.5)
            ax.set_xlim(0, cell_max)
            ax.set_xlabel("Mean fraction")
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat[len(order):]:
            ax.set_visible(False)
        axes.flat[0].legend(loc="lower right", frameon=False)
        fig.suptitle(f"Mean Motif-Induced Cell-Type Distributions\n{setting}", fontsize=16)
        name = f"faceted_cell_distributions_{mode}.png"
        captions[name] = ("Each facet is a noise condition. Blue: NN; red: NP. Arithmetic means of normalized "
                          "run distributions, with equal weight per defined run. All 15 cell types are shown; "
                          "the same fraction axis is used across both selection policies. n counts defined runs.")
        save_figure(fig, output_dir / name)
        for label, cmap in [("NN", "Blues"), ("NP", "Reds")]:
            fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 6.4 * nrows),
                                     squeeze=False, layout="constrained")
            for ax, condition in zip(axes.flat, order):
                values, count = mean_composition(summary, condition, mode, label, "pair")
                matrix = pair_probability_matrix(values)
                im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=pair_max,
                               interpolation="nearest")
                ax.set_title(f"{titles[condition]}\nn={count} runs", fontsize=11)
                ax.set_xticks(np.arange(len(CELL_ORDER)), CELL_ORDER, rotation=90, fontsize=8)
                ax.set_yticks(np.arange(len(CELL_ORDER)), CELL_ORDER, fontsize=8)
                ax.tick_params(length=0)
                ax.spines[:].set_visible(False)
            for ax in axes.flat[len(order):]:
                ax.set_visible(False)
            fig.colorbar(im, ax=list(axes.flat[:len(order)]), shrink=.75, fraction=.025, pad=.02,
                         label="Mean pair probability")
            fig.suptitle(f"Mean Motif-Induced Pairwise Distributions: {label}\n{setting}", fontsize=16)
            name = f"faceted_pair_distributions_{mode}_{label}.png"
            captions[name] = (f"{label}: each facet is a noise condition. All 120 undirected pair categories, "
                              "including same-type pairs, are shown once in the upper triangle. The lower triangle "
                              "is not represented. Arithmetic means give equal weight to each defined normalized "
                              "run; n counts these runs. The probability scale is shared across all conditions, "
                              "both disease classes, and both selection policies.")
            save_figure(fig, output_dir / name)
    return captions


def bootstrap_effects(distances, seed=0):
    rng = np.random.default_rng(seed)
    local = distances[distances["reference"] == "raw_same_mode"]
    rows = []
    for (mode, label, kind), block in local.groupby(["mode", "label", "kind"]):
        raw = block[block["condition"] == "raw"][["run_id", "tv"]].rename(columns={"tv": "raw_tv"})
        for condition, group in block[block["condition"] != "raw"].groupby("condition"):
            paired = group.merge(raw, on="run_id").dropna(subset=["tv", "raw_tv"])
            if paired.empty:
                continue
            differences = (paired["tv"] - paired["raw_tv"]).to_numpy()
            bootstrap = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
            rows.append(dict(condition=condition, mode=mode, label=label, kind=kind,
                             n_paired_runs=len(differences), mean_excess_tv=float(differences.mean()),
                             ci_low=float(np.quantile(bootstrap, .025)), ci_high=float(np.quantile(bootstrap, .975))))
    return pd.DataFrame(rows)


def text_pages(pdf, title, paragraphs):
    lines = []
    for paragraph in paragraphs:
        lines.extend(textwrap.wrap(paragraph, width=109) or [""])
        lines.append("")
    chunks = [lines[i:i + 27] for i in range(0, len(lines), 27)] or [[]]
    for ix, chunk in enumerate(chunks):
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.055, .94, title + (" (continued)" if ix else ""), fontsize=19, weight="bold", va="top")
        fig.text(.055, .86, "\n".join(chunk), fontsize=10.7, linespacing=1.55, va="top")
        pdf.savefig(fig)
        plt.close(fig)


def generate_report(output_dir, results, inventory, metrics, curves, distributions, distances, within, reference_run):
    order = condition_order(metrics)
    reference = next(r for r in results if r["condition"] == "raw" and r["run_id"] == reference_run)
    references_equal = all(
        reference["selections"]["fixed"]["distributions"][label][kind]
        == reference["selections"]["optimal"]["distributions"][label][kind]
        for label in ["NN", "NP"] for kind in ["cell", "pair"])
    (output_dir / "reference_policy_comparison.json").write_text(json.dumps(
        dict(reference_run=reference_run, fixed_stringency=reference["selections"]["fixed"]["stringency"],
             optimal_stringency=reference["selections"]["optimal"]["stringency"],
             identical_reference_distributions=references_equal), indent=2) + "\n")
    tie_rows = []
    for run in results:
        best_auc = run["selections"]["optimal"]["auc"]
        tied = [r["stringency"] for r in run["curves"] if r["auc"] == best_auc]
        tie_rows.append(dict(condition=run["condition"], run_id=run["run_id"],
                             chosen_stringency=run["selections"]["optimal"]["stringency"],
                             n_tied_optima=len(tied), lowest_tied=min(tied), highest_tied=max(tied),
                             tied_thresholds=";".join(f"{v:g}" for v in tied)))
    ties = pd.DataFrame(tie_rows)
    ties.to_csv(output_dir / "tied_optimal_thresholds_by_run.csv", index=False)
    composition_summary, centroids = summarize_distributions(distributions, results, reference_run)
    composition_summary.to_csv(output_dir / "average_distributions_by_condition.csv", index=False)
    centroids.to_csv(output_dir / "condition_mean_distances_to_original.csv", index=False)
    undefined = distributions[distributions["vector"].map(lambda value: value is None)].drop(columns="vector")
    undefined.to_csv(output_dir / "undefined_distributions.csv", index=False)
    availability = distributions.assign(undefined=distributions["vector"].map(lambda value: value is None)).groupby(
        ["condition", "noise_rate", "mode", "label", "kind"], as_index=False).agg(
            n_runs=("undefined", "size"), n_undefined=("undefined", "sum"), fraction_undefined=("undefined", "mean"))
    availability.to_csv(output_dir / "distribution_availability_summary.csv", index=False)
    plot_metrics(metrics, order, output_dir)
    plot_curves(curves, order, output_dir)
    curve_summary = curves.groupby(["condition", "noise_rate", "stringency"], as_index=False).agg(
        n_runs=("auc", "size"), n_rf_admissible=("auc", "count"), mean_auc=("auc", "mean"), sd_auc=("auc", "std"))
    curve_summary.to_csv(output_dir / "auc_by_stringency_summary.csv", index=False)
    figure_names = ["optimal_stringency_by_noise.png", "auc_by_noise.png", "motif_counts_fixed_vs_optimal.png",
                    "auc_vs_stringency_by_noise.png"]
    plot_undefined_distributions(availability, order, output_dir)
    if not undefined.empty:
        figure_names.append("undefined_distribution_fraction_by_noise.png")
    for kind in ["pair", "cell"]:
        figure_names.append(plot_distances(distances, kind, order, output_dir, centroids=centroids))
        if not within.empty:
            figure_names.append(plot_distances(within, kind, order, output_dir, within=True))
        common_figure = plot_distances(distances, kind, order, output_dir, common=True, centroids=centroids)
        if not references_equal:
            figure_names.append(common_figure)
        plot_mean_distributions(distributions, kind, order, output_dir)
        figure_names.append(f"average_{kind}_distributions_all_noise.png")
    facet_captions = plot_faceted_mean_distributions(composition_summary, order, output_dir)
    figure_names.extend(facet_captions)
    metric_summary = metrics.groupby(["condition", "noise_rate", "mode"], as_index=False).agg(
        n_runs=("run_id", "count"), n_rf_admissible=("auc", "count"), mean_stringency=("stringency", "mean"), sd_stringency=("stringency", "std"),
        mean_auc=("auc", "mean"), sd_auc=("auc", "std"), mean_motifs=("n_motifs", "mean"), sd_motifs=("n_motifs", "std"),
        median_motifs=("n_motifs", "median"), mean_empty_folds=("empty_folds", "mean")).sort_values(["noise_rate", "condition", "mode"])
    metric_summary.to_csv(output_dir / "condition_metrics_summary.csv", index=False)
    same = distances[distances["reference"] == "raw_same_mode"]
    distance_summary = same.groupby(["condition", "noise_rate", "mode", "label", "kind"], as_index=False).agg(
        n_runs=("tv", "count"), mean_tv=("tv", "mean"), sd_tv=("tv", "std"),
        median_tv=("tv", "median"), q25=("tv", lambda s: s.quantile(.25)), q75=("tv", lambda s: s.quantile(.75)))
    distance_summary.to_csv(output_dir / "distance_summary.csv", index=False)
    effects = bootstrap_effects(distances)
    effects.to_csv(output_dir / "noise_excess_distance_vs_raw_retrials.csv", index=False)
    if not within.empty:
        within_summary = within.groupby(["condition", "noise_rate", "mode", "label", "kind"], as_index=False).agg(
            n_pairs=("tv", "count"), mean_tv=("tv", "mean"), median_tv=("tv", "median"),
            q25=("tv", lambda s: s.quantile(.25)), q75=("tv", lambda s: s.quantile(.75)))
        within_summary.to_csv(output_dir / "within_condition_distance_summary.csv", index=False)
    valid_counts = {}
    for record in results:
        valid_counts[record["condition"]] = valid_counts.get(record["condition"], 0) + 1
    source_counts = inventory.groupby("condition").size().to_dict()
    exclusion_path = output_dir / "source_data_exclusions.json"
    exclusions = json.loads(exclusion_path.read_text()) if exclusion_path.exists() else []
    excluded_counts = {}
    for row in exclusions:
        excluded_counts[row["condition"]] = excluded_counts.get(row["condition"], 0) + 1
    coverage = pd.DataFrame([dict(condition=c, requested=source_counts.get(c, 0), completed=valid_counts.get(c, 0),
                                 excluded=excluded_counts.get(c, 0),
                                 pending=source_counts.get(c, 0) - valid_counts.get(c, 0) - excluded_counts.get(c, 0))
                             for c in source_counts])
    coverage = coverage.sort_values("condition", key=lambda column: column.map({c: ix for ix, c in enumerate(order)}))
    coverage.to_csv(output_dir / "completion_inventory.csv", index=False)
    mismatches = [dict(condition=r["condition"], run_id=r["run_id"], **r["export_validation"])
                  for r in results if r["export_validation"].get("available") and not r["export_validation"].get("matched")]
    (output_dir / "export_validation_mismatches.json").write_text(json.dumps(mismatches, indent=2) + "\n")
    export_available = sum(bool(r["export_validation"].get("available")) for r in results)
    settings = results[0]["settings"]
    paragraphs = [
        f"Analyzed {len(results)} valid runs from {len(inventory)} requested runs across {len(order)} conditions; "
        f"{len(exclusions)} source-data exclusions and {int(coverage['pending'].sum())} pending runs. "
        f"Original reference: raw run {reference_run}; the reference itself is excluded from raw-retrial boxes.",
        f"Dataset: melanoma NN vs NP, {sum(v == 'NN' for v in settings['patient_classes'].values())} NN patients and "
        f"{sum(v == 'NP' for v in settings['patient_classes'].values())} NP patients, with four-node FANMOD motifs and {reference['n_fovs']} FOV tables per complete run. "
        "Noise values are cell-label flip fractions: 0.001 = 0.1%, 0.003 = 0.3%, 0.01 = 1%, and 0.03 = 3%. "
        "The matrix-labelled folder is an additional 3% condition.",
        "Complete significant FANMOD caches are used for every condition. Earlier edge-only exports therefore do not limit NN/NP distributions or tuning. "
        "This report supersedes the earlier exploratory count-score AUC estimates.",
        f"For each run and each threshold, hold out one patient, select hard-discriminative motifs using only the training patients, "
        f"and train a {settings['trees']}-tree Random Forest on CISM's per-patient motif frequencies. Aggregate held-out probabilities to calculate NN-positive ROC AUC. "
        f"The random seed is reset to {settings['seed']} for each fitted forest, standardizing classifier randomness across conditions. "
        "No SHAP computations are needed for these analyses.",
        "Selection uses p < 0.05, motifs occurring significantly in exactly one training class, and patient prevalence strictly greater than stringency. "
        "To match the exporter, take the highest-frequency 30 FOV rows per class, then deduplicate motif IDs. This may yield fewer than 30 motifs per class. "
        "Patient frequencies are sum(Count) / sum(Count/Freq) over significant FOV rows for that motif.",
        "The tested thresholds are " + ", ".join(f"{v:g}" for v in settings["thresholds"]) + ". "
        "Optimal stringency maximizes the run's LOOCV AUC; exact ties choose the smallest threshold. "
        "Patient prevalence is discrete, so multiple thresholds can select identical features or tie in AUC. "
        "All tied optima are exported separately; the chosen threshold need not be a unique optimum. "
        "AUC at that chosen threshold is a tuned descriptive result, not an independent generalization estimate. Nested CV would be needed for the latter. "
        "A threshold is eligible only if EVERY training fold selects at least one motif. RF AUC is undefined at ineligible thresholds and they cannot become the optimum. "
        "Neutral-probability fallback scores are retained only as diagnostics in run checkpoints; they do not enter AUC plots or tuning. "
        "Mean AUC curves use only eligible runs at each threshold, with their counts reported in auc_by_stringency_summary.csv.",
        "Motif counts and distribution plots use the UNION of motifs selected across LOOCV training folds. "
        "They do not require selection in every fold. Within each disease class, sum occurrences of selected motifs across its patients, "
        "using significant FOV records only; weight each motif's cell and edge multiplicities by that occurrence count, then normalize separately for cells and edges. "
        "Overlapping motif occurrences contribute repeatedly; these are motif-induced compositions, not unique-cell or unique-edge tissue abundances.",
        "All pairs are UNDIRECTED, including same-cell-type pairs; reversed edge labels are combined. "
        "This corrects the arbitrary edge-orientation treatment of the earlier summary. "
        "An empty motif-induced distribution is undefined and is recorded as missing, never silently treated as a zero distribution.",
        "AUC optimization can select motifs associated with only one disease class: the classifier can recognize the other class through their absence. "
        "Thus a valid classifier can coexist with zero motif-induced mass and no normalized distribution in one class. "
        "Undefined fractions are reported as an outcome, separately from unusable source files. Distance boxes and means are conditional on nonempty distributions; "
        "each box labels its actual sample size. Mean-composition heatmaps likewise use only defined vectors.",
        "Distance is total variation: TV(P,Q) = 0.5 * sum(abs(P-Q)), ranging from 0 (identical) to 1 (disjoint). "
        "Primary boxes compare each run with raw run " + str(reference_run) + " under the SAME policy: fixed with fixed, optimal with optimal. " +
        ("The original fixed and optimal reference distributions are identical, so these figures also use the same common fixed-0.46 original; duplicate figures are omitted. "
         if references_equal else "Additional figures use raw fixed 0.46 as a common reference across both policies. ") +
        "Within-condition boxes contain all distinct trial pairs. Those pairwise observations are dependent and are descriptive, not independent replicates.",
        "Diamonds on the reference-distance figures show distance between the condition's MEAN distribution and the original distribution, "
        "while boxes show individual trials. These answer different questions because distance of the mean need not equal mean trial distance.",
        "Noise-excess estimates subtract each matching raw retrial's distance from the noisy run's distance; "
        "95% bootstrap intervals resample matched run IDs (2,000 draws). They describe algorithm/noise variability in this patient cohort, "
        "not uncertainty across new patient cohorts. Same-index trials are matched for comparison, without assuming identical FANMOD random streams.",
        f"Fixed-stringency motif reconstruction matched {export_available - len(mismatches)} of {export_available} available exports. "
        "Any discrepancies are saved in export_validation_mismatches.json. Seed and software versions are recorded separately.",
        f"Input-quality handling: {len(exclusions)} runs were excluded because malformed/truncated caches prevent establishing complete FOV data. "
        "Even a nonsignificant malformed final record may indicate additional missing records. Such runs are not imputed or silently repaired. "
        "The exclusion audit is saved in source_data_exclusions.json; any earlier diagnostic reconstruction is quarantined and omitted from all results.",
        "Noise labels follow folder names. The current historical generator source constructs a matrix noise object and then overwrites it with prior-based noise. "
        "The saved outputs do not establish which revision generated each folder. Treat 'matrix' as a folder-labelled experimental condition, "
        "not independently verified evidence of the noise mechanism.",
    ]
    findings = []
    for _, row in metric_summary[metric_summary["mode"] == "optimal"].iterrows():
        findings.append(f"{row['condition']}: optimal stringency {row['mean_stringency']:.3f} +/- {row['sd_stringency']:.3f}; "
                        f"tuned AUC {row['mean_auc']:.3f} +/- {row['sd_auc']:.3f}; "
                        f"fold-union motifs {row['mean_motifs']:.1f} +/- {row['sd_motifs']:.1f} (mean +/- run SD, n={row['n_runs']}).")
    for kind in ["pair", "cell"]:
        for mode in ["fixed", "optimal"]:
            block = distance_summary[(distance_summary["kind"] == kind) & (distance_summary["mode"] == mode)]
            detail = [f"{r.condition} {r.label}: {r.mean_tv:.3f}" for r in block.itertuples()]
            findings.append(f"Mean {kind} TV distance to original ({mode}): " + "; ".join(detail) + ".")
    interpretations = []
    if not effects.empty:
        for kind in ["pair", "cell"]:
            for mode in ["fixed", "optimal"]:
                descriptions = []
                for label in ["NN", "NP"]:
                    block = effects[(effects["kind"] == kind) & (effects["mode"] == mode) & (effects["label"] == label)]
                    positive = block.loc[block["ci_low"] > 0, "condition"].tolist()
                    unresolved = block.loc[(block["ci_low"] <= 0) & (block["ci_high"] >= 0), "condition"].tolist()
                    text = f"{label}: excess distance above raw retrials with interval wholly above zero in " + (", ".join(positive) or "none")
                    if unresolved:
                        text += "; intervals include zero in " + ", ".join(unresolved)
                    descriptions.append(text)
                interpretations.append(f"{kind.title()} distributions, {mode} selection. " + ". ".join(descriptions) + ".")
        positive = int((effects["ci_low"] > 0).sum())
        overview = f"In {positive} of {len(effects)} noise-condition/class/distribution/policy comparisons, the mean excess TV interval lies wholly above zero relative to matched raw retrials. "
        overview += "This is descriptive evidence of distribution drift beyond raw algorithmic variability in this cohort; it is conditional on distributions being defined."
        interpretations.insert(0, overview)
        if references_equal:
            noisy = distance_summary[distance_summary["condition"] != "raw"]
            paired_modes = noisy[noisy["mode"] == "fixed"].merge(noisy[noisy["mode"] == "optimal"],
                on=["condition", "label", "kind"], suffixes=("_fixed", "_optimal"))
            increases = int((paired_modes["mean_tv_optimal"] > paired_modes["mean_tv_fixed"]).sum())
            interpretations.insert(1, f"AUC tuning increases conditional mean distance in {increases} of {len(paired_modes)} matched noise-condition/class/distribution comparisons. "
                "Optimizing classification AUC therefore does not establish compositional robustness. Undefined-distribution fractions provide an additional instability measure.")
    figure_captions = {
        "optimal_stringency_by_noise.png": "One observation per run; best threshold on the declared grid. Lowest threshold wins exact AUC ties.",
        "auc_by_noise.png": "One tuned LOOCV AUC per run, using Random Forest probabilities and fold-specific feature selection.",
        "motif_counts_fixed_vs_optimal.png": "Number of distinct motif IDs in the union across LOOCV training folds. Both settings are reconstructed from full caches.",
        "auc_vs_stringency_by_noise.png": "Top: mean/SD among eligible runs. Bottom: eligible fraction. Every fold must have a feature; gaps are ineligible. Per-threshold counts are in the CSV summary.",
        "undefined_distribution_fraction_by_noise.png": "Fraction of usable trials with zero selected-motif occurrence mass in a disease class; these have no normalized cell/pair distribution. AUC can still be computed.",
    }
    md = ["# Final Motif Noise Robustness Analysis", "", *[p + "\n" for p in paragraphs],
          "## Results", "", *[p + "\n" for p in findings],
          "## Comparison with Raw Retrial Variability", "", *[p + "\n" for p in interpretations],
          "## Completion", "", dataframe_to_markdown(coverage), "", "## Condition Summary", "", dataframe_to_markdown(metric_summary)]
    for name in figure_names:
        md.extend(["", f"![{name}]({name})"])
        if name in facet_captions:
            md.extend(["", facet_captions[name]])
    (output_dir / "motif_noise_final_report.md").write_text("\n".join(md) + "\n")
    temporary_pdf = output_dir / "motif_noise_final_report.tmp.pdf"
    with PdfPages(temporary_pdf) as pdf:
        pdf.infodict()["Title"] = "Motif-Induced Distribution Robustness to Noise"
        headline_findings = findings[:1] + findings[max(1, len(order) - 2):len(order)]
        text_pages(pdf, "Motif-Induced Distribution Robustness to Noise", paragraphs[:3] + headline_findings)
        text_pages(pdf, "Methods and Interpretation", paragraphs[3:])
        add_table_page(pdf, "Completed Experiment Inventory", coverage, "Every discovered condition is included; incomplete runs are not hidden.")
        if exclusions:
            exclusion_table = pd.DataFrame([{"Condition": row["condition"], "Run": row["run_id"],
                                             "Affected files": len({issue["path"] for issue in row["issues"]}),
                                             "Reason": "Truncated/malformed cache"} for row in exclusions])
            add_table_page(pdf, "Source Data Exclusions", exclusion_table,
                           "Excluded trials contribute no AUC, motif-count, or distribution measurements. Other trials in the condition remain usable.")
        audit_path = output_dir / "observed_graph_noise_rates.csv"
        if audit_path.exists():
            audit = pd.read_csv(audit_path)[["condition", "nominal_rate", "observed_flip_rate", "compared_fovs", "compared_nodes"]]
            audit.columns = ["Condition", "Nominal rate", "Observed rate", "FOVs", "Compared nodes"]
            add_table_page(pdf, "Input Check: Observed Graph Label Changes", audit,
                           "Run 0 graph nodes vs raw run 0. Missing values indicate unavailable saved noisy graphs, not zero noise.")
        # Keep tables readable by separating policies and using short columns.
        for mode in ["fixed", "optimal"]:
            compact = metric_summary[metric_summary["mode"] == mode][["condition", "n_runs", "n_rf_admissible", "mean_stringency", "mean_auc", "sd_auc", "mean_motifs", "sd_motifs"]].copy()
            compact.columns = ["Condition", "Trials", "RF trials", "Stringency", "Mean AUC", "SD AUC", "Mean motifs", "SD motifs"]
            add_table_page(pdf, f"Run Summary: {mode.title()} Stringency", compact,
                           "Counts use all valid trials; AUC uses only RF-admissible trials. Optimized AUC is descriptive, not independent test performance.")
        tie_summary = ties.groupby("condition", as_index=False).agg(
            mean_chosen=("chosen_stringency", "mean"), mean_highest_tied=("highest_tied", "mean"),
            mean_tied_count=("n_tied_optima", "mean"), fraction_multiple_optima=("n_tied_optima", lambda s: (s > 1).mean()))
        add_table_page(pdf, "Ambiguity in Optimal Stringency", tie_summary,
                       "Lowest threshold wins ties. Highest tied values are grid points, not guaranteed continuous optimal intervals.")
        if interpretations:
            text_pages(pdf, "Do Changes Exceed Raw Retrial Variability?", interpretations + [
                "These intervals describe the trial-generating process for this fixed cohort. They are not evidence about independent patient cohorts, "
                "and no multiple-comparison-adjusted significance claims are made. Complete effect sizes and intervals are in noise_excess_distance_vs_raw_retrials.csv."])
        if not undefined.empty:
            availability_table = availability[availability["kind"] == "cell"].copy()
            availability_table["Undefined / total"] = availability_table["n_undefined"].astype(str) + " / " + availability_table["n_runs"].astype(str)
            availability_table = availability_table.pivot(index=["condition", "mode"], columns="label", values="Undefined / total").reset_index()
            add_table_page(pdf, "Undefined Disease-Class Distributions", availability_table,
                           "Counts of zero-mass distributions among usable trials, not source-data exclusions. Cell and pair distributions share this zero-mass status.")
        for mode in ["fixed", "optimal"]:
            tv_rows = []
            for condition in order:
                row = {"Condition": condition}
                for kind, name in [("cell", "Cell"), ("pair", "Pair")]:
                    for label in ["NN", "NP"]:
                        chosen = distance_summary[(distance_summary["mode"] == mode) & (distance_summary["condition"] == condition)
                                                  & (distance_summary["kind"] == kind) & (distance_summary["label"] == label)]
                        if not chosen.empty:
                            value = chosen.iloc[0]
                            row[f"{name} {label}"] = f"{value['mean_tv']:.3f} +/- {value['sd_tv']:.3f}"
                tv_rows.append(row)
            add_table_page(pdf, f"Distance to Original: {mode.title()} Selection", pd.DataFrame(tv_rows),
                           "Mean +/- trial SD of total variation distance; same-policy raw reference, excluding its self-distance.")
        for name in figure_names:
            title = name.removesuffix(".png").replace("_", " ").title()
            if name in facet_captions:
                caption = facet_captions[name]
            elif name.startswith("average_"):
                caption = "Columns are noise conditions; entries are mean normalized within-class probabilities across valid runs. NN: blue; NP: red."
                if "pair" in name:
                    caption += " Only the top 20 pairs are displayed; all 120 pairs enter distance calculations."
            elif "within_condition" in name:
                caption = "All distinct trial pairs contribute to boxes; dots display up to 160 pairs. Pairwise observations are dependent. Whiskers: 1.5 IQR."
            elif "distance" in name:
                caption = f"Trial TV distances to raw run {reference_run}; its self-distance is excluded. Boxes: quartiles/median; whiskers: 1.5 IQR; diamonds: condition-mean distance."
            else:
                caption = figure_captions.get(name, "")
            add_image_page(pdf, title, output_dir / name, caption)
        text_pages(pdf, "Numerical Distribution Findings", findings[len(order):])
        text_pages(pdf, "Reproducibility", [
            "Run analysis/noise_robustness/run_motif_noise_summary.sh from the repository. Completed run checkpoints are reused only when source signatures and analysis settings match.",
            "Source tables: source_inventory.csv. Per-run curves: stringency_sweep.csv. Counts and selected thresholds: run_metrics.csv. "
            "Distances: distances_to_original.csv and within_condition_distances.csv. Condition summaries and bootstrap effects are in separate CSVs. "
            "Each runs/<condition>/run_NNN.json records selected motif IDs, fold selections, held-out probabilities, distributions, and source fingerprint.",
            "Reference sensitivity: primary raw run 0 distances and common-fixed-reference distances are provided. Within-condition pairwise distances show retrial variability without relying on one reference.",
        ])
    temporary_pdf.replace(output_dir / "motif_noise_final_report.pdf")
    versions = {}
    from importlib.metadata import version
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "networkx", "pyarrow"]:
        versions[package] = version(package)
    (output_dir / "software_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    if (int(coverage["pending"].sum()) == 0 and output_dir.name == "final_analysis"
            and output_dir.parent.name == "motif_noise_summary_outputs"):
        parent = output_dir.parent
        archive = parent / "legacy_export_only"
        archive.mkdir(exist_ok=True)
        for name in ["motif_noise_experiment_report.pdf", "motif_noise_summary_report.md"]:
            previous = parent / name
            if previous.exists() and not (archive / name).exists():
                shutil.copy2(previous, archive / name)
        replacement = parent / "motif_noise_experiment_report.tmp.pdf"
        shutil.copyfile(output_dir / "motif_noise_final_report.pdf", replacement)
        replacement.replace(parent / "motif_noise_experiment_report.pdf")
        (parent / "motif_noise_summary_report.md").write_text(
            "# Completed Motif Noise Analysis\n\n"
            f"Analyzed {len(results)} valid trials with {len(exclusions)} documented source-data exclusions.\n\n"
            "- [Full PDF report](final_analysis/motif_noise_final_report.pdf)\n"
            "- [Detailed report and figures](final_analysis/motif_noise_final_report.md)\n"
            "- [Condition summaries](final_analysis/condition_metrics_summary.csv)\n\n"
            "The earlier export-only report is preserved under `legacy_export_only/`. "
            "Its approximate count-score AUC estimates are superseded by the full-cache Random Forest LOOCV analysis.\n")
