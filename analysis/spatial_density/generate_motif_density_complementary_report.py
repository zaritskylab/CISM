from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import math
import os
from pathlib import Path
from typing import Any

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import entropy, mannwhitneyu, ttest_ind


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    PROJECT_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "motif_density_analysis_report_notebook03_discriminative"
)
REPORT_DIR = PROJECT_ROOT / "analysis" / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "motif_density_complementary"
FIGURE_DIR_NO_POINTS = REPORT_DIR / "figures" / "motif_density_complementary_no_points"
REPORT_PDF = REPORT_DIR / "motif_density_complementary_nn_np_null_report.pdf"
REPORT_PDF_NO_POINTS = REPORT_DIR / "motif_density_complementary_nn_np_null_report_no_points.pdf"
TESTS_CSV = REPORT_DIR / "motif_density_complementary_nn_np_null_tests.csv"
GROUPED_TESTS_CSV = REPORT_DIR / "motif_density_fov_patient_nn_np_null_tests.csv"
GROUPED_VALUES_CSV = REPORT_DIR / "motif_density_fov_patient_values.csv"

INSTANCE_CSV = SOURCE_DIR / "motif_density_instance_metrics.csv"
NULL_CSV = SOURCE_DIR / "motif_density_null_samples.csv"

COLORS = {
    "NN": "#5B7DB1",
    "NP": "#D88452",
    "NN_null": "#9DB6D8",
    "NP_null": "#E8A06F",
}
NULL_LABELS = {
    "random_point": "Point null",
    "random_connected_cellset": "Connected-set null",
}
SHORT_COMPARISON_LABELS = {
    "NN observed vs NP observed": "NN obs vs NP obs",
    "NN observed vs Point null": "NN obs vs point null",
    "NP observed vs Point null": "NP obs vs point null",
    "NN observed vs Connected-set null": "NN obs vs connected null",
    "NP observed vs Connected-set null": "NP obs vs connected null",
}
RADIUS_SETTINGS = [
    {"label": "50 micron", "pixels": 100, "metric": "radius_100_density"},
    {"label": "100 micron", "pixels": 200, "metric": "radius_200_density"},
]
AGGREGATION_SETTINGS = [
    {"label": "Per instance", "key": "instance"},
    {"label": "Per motif", "key": "motif"},
]
NULL_SETTINGS = [
    {"label": "Point null", "key": "random_point"},
    {"label": "Connected-set null", "key": "random_connected_cellset"},
]


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(INSTANCE_CSV)
    nulls = pd.read_csv(NULL_CSV)
    return observed, nulls


def smoothed_kl(values_a: np.ndarray, values_b: np.ndarray, bins: int = 8) -> tuple[float, float]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan
    combined = np.concatenate([values_a, values_b])
    if np.nanmin(combined) == np.nanmax(combined):
        return 0.0, 0.0
    hist_a, edges = np.histogram(values_a, bins=bins, range=(float(np.nanmin(combined)), float(np.nanmax(combined))))
    hist_b, _ = np.histogram(values_b, bins=edges)
    hist_a = hist_a.astype(float) + 0.5
    hist_b = hist_b.astype(float) + 0.5
    prob_a = hist_a / hist_a.sum()
    prob_b = hist_b / hist_b.sum()
    return float(entropy(prob_a, prob_b)), float(entropy(prob_b, prob_a))


def compare_distributions(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
    comparison: str,
    metric: str,
    radius_label: str,
    aggregation: str,
    null_type: str,
) -> dict[str, Any]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) == 0 or len(values_b) == 0:
        return {
            "radius": radius_label,
            "aggregation": aggregation,
            "null_type": null_type,
            "metric": metric,
            "comparison": comparison,
            "group_a": label_a,
            "group_b": label_b,
            "n_a": len(values_a),
            "n_b": len(values_b),
        }
    t_result = ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
    try:
        mw_result = mannwhitneyu(values_a, values_b, alternative="two-sided")
        mw_statistic = float(mw_result.statistic)
        mw_p_value = float(mw_result.pvalue)
    except ValueError:
        mw_statistic = np.nan
        mw_p_value = np.nan
    kl_a_to_b, kl_b_to_a = smoothed_kl(values_a, values_b)
    return {
        "radius": radius_label,
        "aggregation": aggregation,
        "null_type": null_type,
        "metric": metric,
        "comparison": comparison,
        "group_a": label_a,
        "group_b": label_b,
        "n_a": int(len(values_a)),
        "n_b": int(len(values_b)),
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "mean_difference_a_minus_b": float(np.mean(values_a) - np.mean(values_b)),
        "welch_t_statistic": float(t_result.statistic) if np.isfinite(t_result.statistic) else np.nan,
        "welch_t_p_value": float(t_result.pvalue) if np.isfinite(t_result.pvalue) else np.nan,
        "mannwhitney_u_statistic": mw_statistic,
        "mannwhitney_p_value": mw_p_value,
        "kl_a_to_b": kl_a_to_b,
        "kl_b_to_a": kl_b_to_a,
    }


def values_for_page(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    metric: str,
    aggregation: str,
    null_type: str,
) -> dict[str, np.ndarray]:
    observed_metric = observed.dropna(subset=[metric]).copy()
    null_metric = nulls[(nulls["null_type"] == null_type)].dropna(subset=[metric]).copy()

    if aggregation == "motif":
        observed_metric = (
            observed_metric.groupby(["motif_id", "motif_association"], as_index=False, observed=True)
            .agg(value=(metric, "mean"))
        )
        null_metric = (
            null_metric.groupby(["motif_id", "motif_association"], as_index=False, observed=True)
            .agg(value=(metric, "mean"))
        )
        value_col = "value"
    else:
        null_metric = (
            null_metric.groupby(["motif_id", "motif_association", "instance_id"], as_index=False, observed=True)
            .agg(value=(metric, "mean"))
        )
        value_col = metric

    return {
        "NN observed": observed_metric.loc[observed_metric["motif_association"] == "NN", value_col].to_numpy(dtype=float),
        "NP observed": observed_metric.loc[observed_metric["motif_association"] == "NP", value_col].to_numpy(dtype=float),
        "NN null": null_metric.loc[null_metric["motif_association"] == "NN", "value"].to_numpy(dtype=float)
        if aggregation == "instance"
        else null_metric.loc[null_metric["motif_association"] == "NN", value_col].to_numpy(dtype=float),
        "NP null": null_metric.loc[null_metric["motif_association"] == "NP", "value"].to_numpy(dtype=float)
        if aggregation == "instance"
        else null_metric.loc[null_metric["motif_association"] == "NP", value_col].to_numpy(dtype=float),
    }


def grouped_values(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    metric: str,
    null_type: str,
    level: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Average instances within FOVs, then FOVs within patients."""
    fov_keys = ["motif_association", "patient", "FOV"]
    patient_keys = ["motif_association", "patient"]
    observed_fov = observed.dropna(subset=[metric]).groupby(fov_keys, observed=True)[metric].mean().reset_index(name="value")
    null_by_instance = (
        nulls[nulls["null_type"] == null_type]
        .dropna(subset=[metric])
        .groupby(fov_keys + ["instance_id"], observed=True)[metric]
        .mean()
        .reset_index(name="value")
    )
    null_fov = null_by_instance.groupby(fov_keys, observed=True)["value"].mean().reset_index()
    if level == "patient":
        observed_groups = observed_fov.groupby(patient_keys, observed=True)["value"].mean().reset_index()
        null_groups = null_fov.groupby(patient_keys, observed=True)["value"].mean().reset_index()
    elif level == "fov":
        observed_groups, null_groups = observed_fov, null_fov
    else:
        raise ValueError(f"Unknown aggregation level: {level}")

    rows = []
    values = {}
    for association in ("NN", "NP"):
        for distribution, frame in (("observed", observed_groups), ("null", null_groups)):
            group = frame[frame["motif_association"] == association].copy()
            values[f"{association} {distribution}"] = group["value"].to_numpy(dtype=float)
            group["distribution"] = distribution
            group["level"] = level
            group["metric"] = metric
            group["null_type"] = null_type
            rows.append(group)
    return values, pd.concat(rows, ignore_index=True)


def build_grouped_figures(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    output_dir: Path = REPORT_DIR,
    name_prefix: str = "motif_density",
    footer_note: str = "Bounded run: one FOV per patient; FOV/patient views coincide. Tests are exploratory.",
) -> None:
    all_tests = []
    all_values = []
    for level in ("fov", "patient"):
        figure_dir = output_dir / "figures" / f"{name_prefix}_{level}_no_points"
        figure_dir.mkdir(parents=True, exist_ok=True)
        page_paths = []
        for radius in RADIUS_SETTINGS:
            for null_setting in NULL_SETTINGS:
                values, grouped = grouped_values(observed, nulls, radius["metric"], null_setting["key"], level)
                all_values.append(grouped)
                tests = page_tests(
                    values, radius["metric"], radius["label"],
                    f"Per {level.upper() if level == 'fov' else level}",
                    null_setting["label"], null_setting["key"],
                )
                all_tests.extend(tests)
                output_path = figure_dir / f"nn_np_density_{radius['pixels']}px_{null_setting['key']}_{level}.png"
                save_page_figure(
                    values, tests, radius["label"],
                    f"Per {level.upper() if level == 'fov' else level}",
                    null_setting["label"], radius["metric"], output_path,
                    show_points=False,
                    footer_note=footer_note,
                )
                page_paths.append(output_path)
        report_path = output_dir / f"{name_prefix}_{level}_nn_np_null_report_no_points.pdf"
        write_report_from_pages(page_paths, report_path)
        print(f"Wrote {report_path}")
    grouped_values_csv = output_dir / f"{name_prefix}_fov_patient_values.csv"
    grouped_tests_csv = output_dir / f"{name_prefix}_fov_patient_nn_np_null_tests.csv"
    pd.concat(all_values, ignore_index=True).to_csv(grouped_values_csv, index=False)
    pd.DataFrame(all_tests).to_csv(grouped_tests_csv, index=False)
    print(f"Wrote {grouped_values_csv}")
    print(f"Wrote {grouped_tests_csv}")


def draw_boxplot(
    ax: plt.Axes,
    values: dict[str, np.ndarray],
    title: str,
    y_label: str,
    show_points: bool,
) -> None:
    labels = ["NN observed", "NP observed", "NN null", "NP null"]
    positions = [1.0, 2.0, 3.4, 4.4]
    colors = [COLORS["NN"], COLORS["NP"], COLORS["NN_null"], COLORS["NP_null"]]
    groups = [values[label] for label in labels]
    boxplot = ax.boxplot(
        groups,
        positions=positions,
        tick_labels=[label.replace(" ", "\n") for label in labels],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#555555", "linewidth": 1.2},
        whiskerprops={"color": "#999999", "linewidth": 1.0},
        capprops={"color": "#999999", "linewidth": 1.0},
        boxprops={"edgecolor": "#666666", "linewidth": 1.0},
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
    if show_points:
        rng = np.random.default_rng(0)
        for position, group in zip(positions, groups):
            if len(group) == 0:
                continue
            jitter = rng.normal(0, 0.035, size=len(group))
            ax.scatter(
                np.full(len(group), position) + jitter,
                group,
                s=14,
                c="#2F2F2F",
                alpha=0.72,
                linewidths=0,
                zorder=3,
            )
    ax.axvline(2.7, linestyle="--", color="#A8A8A8", linewidth=1.0)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_p(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def add_tests_table(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    ax.axis("off")
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                SHORT_COMPARISON_LABELS.get(row["comparison"], row["comparison"]),
                f"{row.get('n_a', 0)} / {row.get('n_b', 0)}",
                f"{row.get('mean_a', np.nan):.4g}",
                f"{row.get('mean_b', np.nan):.4g}",
                f"{row.get('mean_difference_a_minus_b', np.nan):.4g}",
                format_p(row.get("welch_t_p_value", np.nan)),
                format_p(row.get("mannwhitney_p_value", np.nan)),
                f"{row.get('kl_a_to_b', np.nan):.3g} / {row.get('kl_b_to_a', np.nan):.3g}",
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Comparison", "n A/B", "Mean A", "Mean B", "Delta", "Welch p", "MW p", "KL A/B"],
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.09, 0.11, 0.11, 0.12, 0.09, 0.09, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EFEFEF")


def page_tests(
    values: dict[str, np.ndarray],
    metric: str,
    radius_label: str,
    aggregation_label: str,
    null_label: str,
    null_type: str,
) -> list[dict[str, Any]]:
    return [
        compare_distributions(
            values["NN observed"],
            values["NP observed"],
            "NN observed",
            "NP observed",
            "NN observed vs NP observed",
            metric,
            radius_label,
            aggregation_label,
            null_type,
        ),
        compare_distributions(
            values["NN observed"],
            values["NN null"],
            "NN observed",
            "NN null",
            f"NN observed vs {null_label}",
            metric,
            radius_label,
            aggregation_label,
            null_type,
        ),
        compare_distributions(
            values["NP observed"],
            values["NP null"],
            "NP observed",
            "NP null",
            f"NP observed vs {null_label}",
            metric,
            radius_label,
            aggregation_label,
            null_type,
        ),
    ]


def save_page_figure(
    values: dict[str, np.ndarray],
    test_rows: list[dict[str, Any]],
    radius_label: str,
    aggregation_label: str,
    null_label: str,
    metric: str,
    output_path: Path,
    show_points: bool,
    footer_note: str | None = None,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5), facecolor="white")
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.1, 1.35], hspace=0.28)
    ax_plot = fig.add_subplot(grid[0])
    ax_table = fig.add_subplot(grid[1])
    title = f"NN vs NP motif local density | {radius_label} | {aggregation_label} | {null_label}"
    y_label = "Local density (cells per micron^2)"
    if aggregation_label == "Per motif":
        y_label += "\n(mean per motif)"
    elif aggregation_label in ("Per FOV", "Per patient"):
        y_label += f"\n(mean per {aggregation_label[4:]})"
    draw_boxplot(ax_plot, values, title, y_label, show_points=show_points)
    add_tests_table(ax_table, test_rows)
    if footer_note:
        fig.text(0.5, 0.025, footer_note, ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.08)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report_from_pages(page_paths: list[Path], report_pdf: Path) -> None:
    with PdfPages(report_pdf) as pdf:
        for page_path in page_paths:
            image = plt.imread(page_path)
            fig, ax = plt.subplots(figsize=(11.0, 8.5), facecolor="white")
            ax.imshow(image)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def build_figures(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    figure_dir: Path,
    report_pdf: Path,
    show_points: bool,
    footer_note: str | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for stale in figure_dir.glob("*.png"):
        stale.unlink()

    all_test_rows: list[dict[str, Any]] = []
    page_paths: list[Path] = []

    for radius in RADIUS_SETTINGS:
        for aggregation in AGGREGATION_SETTINGS:
            for null_setting in NULL_SETTINGS:
                values = values_for_page(
                    observed,
                    nulls,
                    radius["metric"],
                    aggregation["key"],
                    null_setting["key"],
                )
                test_rows = page_tests(
                    values,
                    radius["metric"],
                    radius["label"],
                    aggregation["label"],
                    null_setting["label"],
                    null_setting["key"],
                )
                all_test_rows.extend(test_rows)
                filename = (
                    f"nn_np_density_{radius['pixels']}px_{aggregation['key']}_{null_setting['key']}.png"
                )
                if not show_points:
                    filename = filename.replace(".png", "_no_points.png")
                page_path = figure_dir / filename
                save_page_figure(
                    values,
                    test_rows,
                    radius["label"],
                    aggregation["label"],
                    null_setting["label"],
                    radius["metric"],
                    page_path,
                    show_points=show_points,
                    footer_note=footer_note,
                )
                page_paths.append(page_path)

    write_report_from_pages(page_paths, report_pdf)
    return all_test_rows, page_paths


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    observed, nulls = read_inputs()

    all_test_rows, page_paths = build_figures(
        observed,
        nulls,
        FIGURE_DIR,
        REPORT_PDF,
        show_points=True,
    )
    no_points_test_rows, no_points_page_paths = build_figures(
        observed,
        nulls,
        FIGURE_DIR_NO_POINTS,
        REPORT_PDF_NO_POINTS,
        show_points=False,
    )

    pd.DataFrame(all_test_rows).to_csv(TESTS_CSV, index=False)

    print(f"Wrote {REPORT_PDF}")
    print(f"Wrote {REPORT_PDF_NO_POINTS}")
    print(f"Wrote {TESTS_CSV}")
    print(f"Wrote {len(page_paths)} page figures to {FIGURE_DIR}")
    print(f"Wrote {len(no_points_page_paths)} no-points page figures to {FIGURE_DIR_NO_POINTS}")
    build_grouped_figures(observed, nulls)


if __name__ == "__main__":
    main()
