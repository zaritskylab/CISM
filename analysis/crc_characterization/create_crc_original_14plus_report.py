#!/usr/bin/env python3
"""Create a self-contained PDF report for the CRC 14+Unidentified analysis."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import json
import base64
import pickle
import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


ROOT = _REPO_ROOT
DATA = ROOT / "data/CRC_original_14plusUnidentified_full"
OUT = ROOT / "analysis/outputs/crc_original_14plus_unidentified"
PDF = OUT / "CRC_14plusUnidentified_full_analysis_report.pdf"

CLR = "#2D78B7"
DII = "#C85252"
GREEN = "#2CA02C"
GOLD = "#D7B600"
DARK = "#20252B"
MID = "#66707A"
LIGHT = "#EEF2F5"


def load_json(path: Path):
    return json.loads(path.read_text())


prep = load_json(DATA / "preparation_summary.json")
summaries = {
    n: load_json(OUT / f"motif_{n}/analysis_outputs/run_summary.json") for n in (3, 4)
}
BEST_N = max(summaries, key=lambda n: summaries[n]["optimal_auc"])
sweeps = {
    n: pd.read_csv(OUT / f"motif_{n}/analysis_outputs/auc_by_stringency.csv") for n in (3, 4)
}
motif_counts = {
    n: pd.read_csv(OUT / f"motif_{n}/analysis_outputs/motif_count_vs_stringency.csv") for n in (3, 4)
}
group_counts = pd.read_csv(DATA / "analysis_group_counts.csv")
removed = pd.read_csv(DATA / "alpha_shape_removed_cells.csv.gz")
mapping = pd.read_csv(DATA / "cell_type_mapping.csv")
singleton = pd.read_csv(OUT / "motif_3/analysis_outputs/singleton_loocv_predictions.csv")
pairwise = pd.read_csv(OUT / "motif_3/analysis_outputs/pairwise_loocv_predictions.csv")
mapped_cells = pd.read_csv(DATA / "mapped_cells.csv.gz", usecols=["patient_id", "cell_type"])
patient_classes = pd.read_csv(DATA / "patient_class.csv", header=None, names=["patient_uid", "patient_class"])
patient_classes["patient_id"] = patient_classes.patient_uid.str.replace("CRC", "", regex=False).astype(int)


def read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    return frame.astype(float)


general_pairwise = {
    label: read_matrix(OUT / f"motif_3/analysis_outputs/general_pairwise_{label}.csv")
    for label in ("CLR", "DII")
}
motif_pairwise = {
    (n, label): read_matrix(OUT / f"motif_{n}/analysis_outputs/motif_induced_pairwise_{label}.csv")
    for n in (3, 4) for label in ("CLR", "DII")
}

# Identify the high-abundance points in the scatterplot for the better motif size.
best_discriminative = pd.read_csv(OUT / f"motif_{BEST_N}/analysis_outputs/discriminative_motifs.csv")
best_opt = summaries[BEST_N]["optimal_stringency"]
best_discriminative = best_discriminative[best_discriminative.patient_percentage > best_opt].copy()
best_per_patient = best_discriminative.groupby(
    ["ID", "patient_class", "Patient"], observed=True
).Count.sum()
best_scatter = best_per_patient.groupby(["ID", "patient_class"], observed=True).agg(
    patient_count="size", average_count="mean", total_count="sum"
).reset_index()
highlighted_motifs = best_scatter[
    (best_scatter.average_count > 30) & (best_scatter.patient_count >= 9)
].sort_values("average_count", ascending=False).reset_index(drop=True)
if len(highlighted_motifs) != 2:
    raise RuntimeError(f"Expected two high-abundance motifs, found {len(highlighted_motifs)}")

node_type_map = pd.read_csv(DATA / "node_color_map.csv").set_index("color_id").node_color.to_dict()


def new_page(pdf: PdfPages, title: str, subtitle: str | None = None, landscape: bool = False):
    size = (11, 8.5) if landscape else (8.5, 11)
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(0.06, 0.955, title, fontsize=19, fontweight="bold", color=DARK, va="top")
    if subtitle:
        fig.text(0.06, 0.922, subtitle, fontsize=9.5, color=MID, va="top")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.902, 0.902], color="#CBD3DA", lw=1))
    fig.text(0.06, 0.025, "CRC CISM analysis | 14 paper groups + Unidentified", fontsize=7.5, color=MID)
    fig.text(0.94, 0.025, f"Generated {date.today().isoformat()}", fontsize=7.5, color=MID, ha="right")
    return fig


def finish(pdf: PdfPages, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def box(fig, xywh, title, value, color=DARK, note=None):
    x, y, w, h = xywh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.008",
                           transform=fig.transFigure, facecolor="white", edgecolor="#CED6DD", lw=1)
    fig.add_artist(patch)
    fig.text(x + 0.018, y + h - 0.028, title, fontsize=8.5, color=MID, va="top")
    fig.text(x + 0.018, y + h - 0.062, value, fontsize=17, color=color, fontweight="bold", va="top")
    if note:
        fig.text(x + 0.018, y + 0.009, note, fontsize=6.8, color=MID, va="bottom")


def paragraph(fig, x, y, text, width=95, fontsize=9.2, color=DARK, line_height=0.021, weight="normal"):
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [""])
    fig.text(x, y, "\n".join(lines), fontsize=fontsize, color=color, va="top",
             linespacing=1.35, fontweight=weight)
    return y - line_height * len(lines)


def add_table(ax, frame, col_widths=None, fontsize=8, header_color="#DDE6ED", scale_y=1.3):
    ax.axis("off")
    table = ax.table(cellText=frame.values, colLabels=frame.columns, cellLoc="center",
                     colLoc="center", colWidths=col_widths, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, scale_y)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=DARK)
        elif r % 2 == 0:
            cell.set_facecolor("#F5F7F9")
    return table


def triangular_heatmap(fig, ax, frame: pd.DataFrame, *, cmap: str, vmax: float,
                       title: str, cbar_label: str | None = None):
    values = frame.to_numpy(copy=True)
    mask = np.tril(np.ones_like(values, dtype=bool), k=-1)
    shown = np.ma.array(values, mask=mask)
    image = ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
    labels = [str(x) for x in frame.columns]
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, fontsize=6.2)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=6.2)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.025)
    cbar.ax.tick_params(labelsize=6.5)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)
    return image


def decoded_motif(motif_id: int) -> nx.Graph:
    encoded = best_discriminative.loc[best_discriminative.ID == motif_id, "motif"].iloc[0]
    directed = pickle.loads(base64.b64decode(encoded))
    graph = nx.Graph()
    for node, attrs in directed.nodes(data=True):
        graph.add_node(node, cell_type=node_type_map[int(attrs["type"])])
    graph.add_edges_from(directed.edges())
    return graph


def path_order(graph: nx.Graph) -> list:
    endpoints = sorted(node for node, degree in graph.degree if degree == 1)
    if len(endpoints) == 2:
        return nx.shortest_path(graph, endpoints[0], endpoints[1])
    return list(graph.nodes())


def draw_motif_diagram(ax, graph: nx.Graph, edge_color: str):
    order = path_order(graph)
    xvals = np.linspace(-1.35, 1.35, len(order))
    positions = {node: (x, 0) for node, x in zip(order, xvals)}
    nx.draw_networkx_edges(graph, positions, ax=ax, width=9, edge_color=edge_color, alpha=.95)
    for node in order:
        cell_type = graph.nodes[node]["cell_type"]
        is_b = cell_type == "B cells"
        outer = "#A8D7E5" if is_b else "#B9B5E4"
        inner = "#70A8BC" if is_b else "#8580C2"
        x, y = positions[node]
        ax.scatter([x], [y], s=2800, color=outer, edgecolor="#71818B", linewidth=1.2, zorder=3)
        ax.scatter([x], [y], s=1250, color=inner, edgecolor="none", zorder=4)
        label = "B cell" if is_b else "CD11c+ DC"
        ax.text(x, -.57, label, ha="center", va="top", fontsize=9, color=DARK)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-.9, .7)
    ax.axis("off")


ASSETS = OUT / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
edge_colors = [GREEN, "#A83DBA"]
motif_image_paths = []
for i, record in highlighted_motifs.iterrows():
    motif_id = int(record.ID)
    graph = decoded_motif(motif_id)
    fig_asset, ax_asset = plt.subplots(figsize=(8, 2.8), facecolor="none")
    draw_motif_diagram(ax_asset, graph, edge_colors[i])
    ax_asset.set_title(
        f"Motif {motif_id} | CLR | {int(record.patient_count)} patients | "
        f"mean count {record.average_count:.1f}", fontsize=12, fontweight="bold", color=DARK
    )
    asset_path = ASSETS / f"motif_{motif_id}.png"
    fig_asset.savefig(asset_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig_asset)
    motif_image_paths.append(asset_path)


with PdfPages(PDF) as pdf:
    metadata = pdf.infodict()
    metadata["Title"] = "CRC 14+Unidentified Full Analysis Report"
    metadata["Author"] = "CISM analysis workflow"
    metadata["Subject"] = "CRC three-node and four-node discriminative motif analysis"

    # Cover and executive summary
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.add_artist(plt.Rectangle((0, 0.80), 1, 0.20, transform=fig.transFigure, color="#173A5E"))
    fig.text(0.07, 0.935, "CRC CISM ANALYSIS", fontsize=13, color="#B9D6ED", fontweight="bold")
    fig.text(0.07, 0.875, "Full analysis report", fontsize=30, color="white", fontweight="bold")
    fig.text(0.07, 0.825, "Original CRC dataset | 14 paper groups + Unidentified", fontsize=13, color="white")
    fig.text(0.07, 0.755, "Executive results", fontsize=17, color=DARK, fontweight="bold")
    box(fig, (0.07, 0.615, 0.255, 0.11), "THREE-NODE MOTIFS", "AUC 0.742", CLR,
        "Optimal stringency 0.10")
    box(fig, (0.372, 0.615, 0.255, 0.11), "FOUR-NODE MOTIFS", "AUC 0.824", GREEN,
        "Optimal stringency 0.30")
    box(fig, (0.675, 0.615, 0.255, 0.11), "BEST BASELINE", "AUC 0.802", GOLD,
        "Singleton cell frequencies")
    y = 0.565
    y = paragraph(fig, 0.07, y,
        "The four-node motif analysis achieved the strongest observed leave-one-out ROC AUC. "
        "The three-node motif classifier did not outperform the singleton or pairwise baselines. "
        "The four-node optimum was tied at stringencies 0.25 and 0.30; the workflow selected 0.30 "
        "because it resolves ties in favor of higher stringency.", width=105, fontsize=10.2, line_height=0.025)
    y -= 0.025
    fig.text(0.07, y, "Scope", fontsize=13, fontweight="bold", color=DARK, va="top")
    y -= 0.034
    y = paragraph(fig, 0.07, y,
        "The report covers cohort preparation, dirt alpha-shape removal, graph construction, FANMOD motif "
        "enumeration, CLR-versus-DII discrimination, stringency selection, singleton and pairwise baselines, "
        "generated artifacts, and interpretation limits.", width=105)
    y -= 0.025
    fig.text(0.07, y, "Critical interpretation", fontsize=13, fontweight="bold", color=DARK, va="top")
    y -= 0.034
    y = paragraph(fig, 0.07, y,
        "The reported optimal AUC was chosen after comparing multiple stringencies on this cohort. It is an "
        "exploratory model-selection result, not an unbiased nested-cross-validation estimate or external-test "
        "result. The 1,000 FANMOD iterations randomize graph motifs; they are not patient-label permutations.",
        width=105)
    fig.text(0.07, 0.075, "Analysis completed 17 November 2026", fontsize=9, color=MID)
    finish(pdf, fig)

    # Cohort and preprocessing
    fig = new_page(pdf, "1. Cohort and preprocessing",
                   "Shared preparation performed before the two motif-size array tasks")
    box(fig, (0.06, 0.785, 0.19, 0.09), "PATIENTS", "35", DARK, "17 CLR | 18 DII")
    box(fig, (0.285, 0.785, 0.19, 0.09), "FOVs", "140", DARK, "Four per patient")
    box(fig, (0.51, 0.785, 0.19, 0.09), "SOURCE OBJECTS", "258,385", DARK)
    box(fig, (0.735, 0.785, 0.19, 0.09), "RETAINED", "249,899", GREEN)
    ax = fig.add_axes([0.08, 0.39, 0.55, 0.34])
    gc = group_counts.sort_values("cells")
    ax.barh(gc.cell_type, gc.cells, color="#5B9BD5")
    ax.set_xlabel("Retained cells")
    ax.tick_params(axis="y", labelsize=7.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E7EBEF", lw=.7)
    ax.set_axisbelow(True)
    ax.set_title("Final analysis categories", loc="left", fontsize=11, fontweight="bold")
    ax2 = fig.add_axes([0.70, 0.48, 0.23, 0.24])
    removed_counts = removed.cell_type.value_counts().sort_values().tail(8)
    ax2.barh(removed_counts.index, removed_counts.values, color="#C97B63")
    ax2.set_title("Largest alpha-shape removals", loc="left", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Biological cells", fontsize=8)
    ax2.tick_params(labelsize=6.8)
    ax2.spines[["top", "right"]].set_visible(False)
    y = 0.335
    y = paragraph(fig, 0.07, y,
        f"Dirt handling. All {prep['excluded_dirt_cells_raw']:,} dirt objects were included while constructing "
        f"the Delaunay geometry. Connected dirt components ({prep['connected_dirt_clusters']:,}) were converted "
        f"to alpha shapes using alpha={prep['dirt_alpha']} and buffer={prep['dirt_buffer_source_pixels']:.0f} pixels. "
        f"The dirt objects and {prep['alpha_shape_removed_biological_cells']:,} biological cells inside those shapes "
        "were then excluded.", width=108)
    y -= 0.015
    y = paragraph(fig, 0.07, y,
        "Graph construction. Delaunay edges longer than 125 source pixels were discarded. At 0.37744 µm per "
        "pixel, this corresponds to 47.18 µm. The resulting dataset contains 15 categories: 14 paper-level "
        "groups plus Unidentified. No duplicate coordinates were removed.", width=108)
    y -= 0.015
    paragraph(fig, 0.07, y,
        "Reproducibility note. Alpha-shape removal is a deliberate modification relative to the original paper's "
        "released contact-analysis code, which retained dirt during triangulation and later removed dirt rows and "
        "columns. This report describes the modified workflow actually run here.", width=108, color="#8A3D2F")
    finish(pdf, fig)

    # Cell-type distributions
    fig = new_page(pdf, "2. Cell-type distributions",
                   "Mean patient-level frequency after dirt and alpha-shape removal")
    patient_cell_counts = mapped_cells.groupby(["patient_id", "cell_type"], observed=True).size().unstack(fill_value=0)
    patient_cell_freq = patient_cell_counts.div(patient_cell_counts.sum(axis=1), axis=0)
    patient_cell_freq = patient_cell_freq.join(patient_classes.set_index("patient_id")["patient_class"])
    class_means = patient_cell_freq.groupby("patient_class", observed=True).mean(numeric_only=True)
    order = class_means.mean(axis=0).sort_values(ascending=False).index.tolist()
    ax = fig.add_axes([0.09, 0.42, 0.84, 0.41])
    x = np.arange(len(order))
    width = .38
    ax.bar(x - width/2, class_means.loc["CLR", order], width, color=CLR, label="CLR")
    ax.bar(x + width/2, class_means.loc["DII", order], width, color=DII, label="DII")
    ax.set_xticks(x, order, rotation=48, ha="right", fontsize=8)
    ax.set_ylabel("Mean frequency per patient")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E7EBEF", lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    differences = (class_means.loc["CLR"] - class_means.loc["DII"]).reindex(order)
    largest = differences.abs().sort_values(ascending=False).head(5).index
    diff_table = pd.DataFrame({
        "Cell type": largest,
        "CLR": [f"{class_means.loc['CLR', c]:.3f}" for c in largest],
        "DII": [f"{class_means.loc['DII', c]:.3f}" for c in largest],
        "CLR − DII": [f"{differences[c]:+.3f}" for c in largest],
    })
    tax = fig.add_axes([0.16, 0.175, 0.68, 0.16])
    add_table(tax, diff_table, col_widths=[.40, .18, .18, .24], fontsize=8.5, scale_y=1.45)
    fig.text(0.16, 0.35, "Largest absolute cohort differences", fontsize=11, fontweight="bold", color=DARK)
    paragraph(fig, 0.08, 0.12,
        "Frequencies were calculated within each patient after preprocessing and then averaged within CLR and DII. "
        "This gives every patient equal weight and is the same representation used by the singleton classifier.",
        width=108, fontsize=8.8)
    finish(pdf, fig)

    # Workflow and jobs
    fig = new_page(pdf, "3. What the two compute jobs did",
                   "Slurm array 24241624: one task per motif size")
    steps = [
        ("1", "FANMOD enumeration", "Run every one of the 140 FOV graphs with 1,000 graph-randomization iterations."),
        ("2", "Parsing and aggregation", "Parse colored motif structures and cell identities into patient/FOV motif observations."),
        ("3", "Discriminative discovery", "Associate motifs with CLR or DII using structure plus cell identities."),
        ("4", "Stringency sweep", "Evaluate thresholds 0.00–0.95 in increments of 0.05 by patient-level leave-one-out CV."),
        ("5", "Optimal model", "Choose the highest AUC; on a tie, retain the higher stringency."),
        ("6", "Secondary outputs", "Create pairwise matrices, baseline classifiers, plots, CSV files, caches, and pickle bundles."),
    ]
    y = 0.855
    for num, name, desc in steps:
        fig.add_artist(plt.Circle((0.085, y - 0.012), 0.018, transform=fig.transFigure, color="#3E6F94"))
        fig.text(0.085, y - 0.012, num, color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        fig.text(0.12, y, name, fontsize=10, color=DARK, fontweight="bold", va="top")
        paragraph(fig, 0.12, y - 0.023, desc, width=92, fontsize=8.8, line_height=0.018)
        y -= 0.102
    comparison = pd.DataFrame([
        ["Graph size", "3 cells", "4 cells"],
        ["Motif-observation rows", f"{summaries[3]['motifs_dataset_rows']:,}", f"{summaries[4]['motifs_dataset_rows']:,}"],
        ["Distinct discriminative IDs", f"{summaries[3]['discriminative_motif_ids']:,}", f"{summaries[4]['discriminative_motif_ids']:,}"],
        ["Approximate runtime", "1 h 6 min", "6 h 18 min"],
        ["Output footprint", "35 MB", "518 MB"],
    ], columns=["Property", "Array task 3", "Array task 4"])
    ax = fig.add_axes([0.10, 0.115, 0.80, 0.20])
    add_table(ax, comparison, col_widths=[.40, .30, .30], fontsize=8.8, scale_y=1.55)
    fig.text(0.10, 0.335, "Task comparison", fontsize=12, fontweight="bold", color=DARK)
    finish(pdf, fig)

    # AUC sweeps
    fig = new_page(pdf, "4. AUC versus stringency",
                   "Patient-level leave-one-out ROC AUC across discrimination stringency")
    ax = fig.add_axes([0.11, 0.49, 0.78, 0.34])
    for n, color, marker in [(3, CLR, "o"), (4, GREEN, "s")]:
        sw = sweeps[n]
        valid = sw.dropna(subset=["roc_auc"])
        ax.plot(valid.stringency, valid.roc_auc, marker=marker, color=color, lw=2,
                ms=5, label=f"{n}-node motifs")
        opt = summaries[n]["optimal_stringency"]
        auc = summaries[n]["optimal_auc"]
        ax.axvline(opt, ls="--", color=color, lw=1.1, alpha=.75)
        ax.scatter([opt], [auc], s=80, color=color, edgecolor="white", zorder=4)
        ax.annotate(f"{auc:.3f} at {opt:.2f}", (opt, auc), xytext=(8, 10),
                    textcoords="offset points", fontsize=8)
    ax.axhline(.5, ls=":", color=MID, lw=1)
    ax.set(xlabel="Discrimination stringency", ylabel="ROC AUC", ylim=(.35, .9), xlim=(-.02, .58))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E9EDF0", lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    table = pd.DataFrame([
        ["Optimal stringency", "0.10", "0.30"],
        ["Optimal AUC", "0.742", "0.824"],
        ["AUC at 0.55", "0.582", "0.441"],
        ["Mean selected features at optimum", "41.29", "47.03"],
        ["Usable thresholds", "0.00–0.55", "0.00–0.55"],
    ], columns=["Metric", "3-node", "4-node"])
    ax = fig.add_axes([0.12, 0.205, 0.76, 0.20])
    add_table(ax, table, col_widths=[.48, .26, .26], fontsize=9, scale_y=1.5)
    y = 0.165
    y = paragraph(fig, 0.08, y,
        "The four-node curve peaks at both 0.25 and 0.30 (AUC 0.824). The workflow chooses 0.30 because its "
        "sorting rule favors higher stringency when AUC is tied. At thresholds 0.60–0.95, neither job retained "
        "discriminative features; these are unavailable AUCs, not failed jobs.", width=108, fontsize=8.8)
    paragraph(fig, 0.08, y - 0.008,
        "Because the optimal threshold was selected from this same sweep, the peak AUC can be optimistic. A nested "
        "cross-validation or held-out cohort would be required for an unbiased performance estimate.",
        width=108, fontsize=8.8, color="#8A3D2F")
    finish(pdf, fig)

    # Motif availability
    fig = new_page(pdf, "5. Number of motifs versus stringency",
                   "Counts are class-associated motif IDs before the per-fold feature cap")
    ax = fig.add_axes([0.11, 0.48, 0.78, 0.35])
    for n, style, marker in [(3, "-", "o"), (4, "--", "s")]:
        frame = motif_counts[n]
        for label, color in [("CLR", CLR), ("DII", DII)]:
            p = frame[frame.patient_class == label]
            nonzero = p[p.discriminative_motifs > 0]
            ax.plot(nonzero.stringency, nonzero.discriminative_motifs, marker=marker,
                    linestyle=style, color=color, lw=2, ms=4, label=f"{n}-node {label}")
    ax.set_yscale("log")
    ax.set(xlabel="Discrimination stringency", ylabel="# discriminative motifs", xlim=(-.02, .97))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E9EDF0", lw=.7)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    rows = []
    for n in (3, 4):
        opt = summaries[n]["optimal_stringency"]
        p = motif_counts[n][motif_counts[n].stringency == opt].set_index("patient_class")
        rows.append([f"{n}-node at {opt:.2f}", int(p.loc["CLR", "discriminative_motifs"]),
                     int(p.loc["DII", "discriminative_motifs"]),
                     summaries[n]["discriminative_motif_ids"]])
    t = pd.DataFrame(rows, columns=["Selected setting", "CLR motifs", "DII motifs", "All discovered IDs"])
    ax = fig.add_axes([0.11, 0.235, 0.78, 0.15])
    add_table(ax, t, col_widths=[.34, .20, .20, .26], fontsize=9, scale_y=1.6)
    paragraph(fig, 0.08, 0.18,
        "The four-node search space is much larger: it produced approximately twelve times as many motif-observation "
        "rows and more than fifty times as many distinct discriminative IDs as the three-node analysis. The classifier "
        "limits the number of candidate features per class, so the discovered-ID totals are not the number used in "
        "every leave-one-out fold.", width=108, fontsize=8.9)
    finish(pdf, fig)

    # Motif patient prevalence versus appearance
    fig = new_page(pdf, "6. Motif appearance versus patient count",
                   f"Better-performing setting: {BEST_N}-node motifs at stringency "
                   f"{summaries[BEST_N]['optimal_stringency']:.2f}")
    disc = pd.read_csv(OUT / f"motif_{BEST_N}/analysis_outputs/discriminative_motifs.csv")
    opt = summaries[BEST_N]["optimal_stringency"]
    disc = disc[disc.patient_percentage > opt].copy()
    per_patient = disc.groupby(["ID", "patient_class", "Patient"], observed=True).Count.sum()
    scatter = per_patient.groupby(["ID", "patient_class"], observed=True).agg(
        patient_count="size", average_count="mean"
    ).reset_index()
    ax = fig.add_axes([0.12, 0.43, 0.76, 0.40])
    for label, color in [("CLR", CLR), ("DII", DII)]:
        part = scatter[scatter.patient_class == label]
        ax.scatter(part.patient_count, part.average_count, s=30, alpha=.62,
                   color=color, edgecolors="white", linewidths=.3, label=label)
    ax.set(xlabel="Patient count", ylabel="Average motif count per positive patient")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E9EDF0", lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    scatter_table = pd.DataFrame([
        [f"{BEST_N}-node", len(scatter), int(scatter.patient_count.max()),
         f"{scatter.average_count.max():.1f}", f"{summaries[BEST_N]['optimal_auc']:.3f}"],
    ], columns=["Motif size", "Plotted motifs", "Max patients", "Max mean count", "Optimal AUC"])
    tax = fig.add_axes([0.10, 0.245, 0.80, 0.13])
    add_table(tax, scatter_table, col_widths=[.17, .20, .23, .23, .17], fontsize=8.7, scale_y=1.5)
    paragraph(fig, 0.08, 0.18,
        "Patient count is the number of patients in the associated class with a significant enriched occurrence. "
        "Average count is calculated by summing motif appearances across that patient's FOVs and averaging across "
        "positive patients. Each dot represents one class-associated motif ID.", width=108, fontsize=8.8)
    finish(pdf, fig)

    # Draw the two high-abundance motifs highlighted by the scatterplot.
    fig = new_page(pdf, "7. High-abundance four-node motifs",
                   "The two CLR-associated motifs with mean count >30 and support in at least nine patients")
    topology_names = []
    for i, record in highlighted_motifs.iterrows():
        motif_id = int(record.ID)
        graph = decoded_motif(motif_id)
        dc_node = next(node for node, attrs in graph.nodes(data=True) if attrs["cell_type"] == "CD11c+ DCs")
        topology = "terminal DC" if graph.degree[dc_node] == 1 else "internal DC"
        topology_names.append(topology)
        ax = fig.add_axes([0.06 + i * .48, 0.59, 0.42, 0.22])
        draw_motif_diagram(ax, graph, edge_colors[i])
        ax.set_title(f"Motif {motif_id}: {topology}", fontsize=11, fontweight="bold", color=DARK)
        counts = best_per_patient.loc[(motif_id, record.patient_class)].sort_index()
        bax = fig.add_axes([0.09 + i * .48, 0.28, 0.35, 0.20])
        bax.bar([str(x) for x in counts.index], counts.values, color=CLR, width=.72)
        bax.set_xlabel("Patient", fontsize=8)
        bax.set_ylabel("Motif count", fontsize=8)
        bax.tick_params(labelsize=7)
        bax.spines[["top", "right"]].set_visible(False)
        bax.grid(axis="y", color="#E9EDF0", lw=.7)
        bax.set_axisbelow(True)
        fig.text(0.27 + i * .48, 0.535,
                 f"3 B cells + 1 CD11c+ DC\n{int(record.patient_count)} CLR patients | "
                 f"mean {record.average_count:.1f} | total {int(record.total_count)}",
                 ha="center", va="top", fontsize=8.8, color=DARK)
    paragraph(fig, 0.07, 0.19,
        "Both motifs have the same cellular composition and the same four-node path structure. Their distinction is "
        "the DC position: motif 202770475 places the DC at the end of the B-cell chain, whereas motif 68815915 "
        "places the DC at an internal position. Edge colors distinguish the drawings only and do not encode an "
        "interaction type or direction.", width=108, fontsize=8.8)
    finish(pdf, fig)

    # General pairwise distributions
    fig = new_page(pdf, "8. Pairwise cell-type distributions", landscape=True,
                   subtitle="All graph interactions; upper triangle displayed")
    vmax = max(frame.to_numpy().max() for frame in general_pairwise.values())
    for i, label in enumerate(("CLR", "DII")):
        ax = fig.add_axes([0.055 + i * .49, 0.15, 0.40, 0.67])
        triangular_heatmap(fig, ax, general_pairwise[label], cmap="Blues" if label == "CLR" else "Reds",
                           vmax=vmax, title=label, cbar_label="Probability" if i == 1 else None)
    fig.text(.50, .09,
             "Values are normalized interaction frequencies pooled within each cohort; a shared color scale is used.",
             ha="center", fontsize=8.5, color=MID)
    finish(pdf, fig)

    # Motif-induced pairwise distributions
    fig = new_page(pdf, "9. Discriminative motif-induced pairwise distributions", landscape=True,
                   subtitle=f"Better-performing setting: {BEST_N}-node motifs at stringency "
                            f"{summaries[BEST_N]['optimal_stringency']:.2f}; upper triangle displayed")
    vmax = max(motif_pairwise[(BEST_N, label)].to_numpy().max() for label in ("CLR", "DII"))
    for col, label in enumerate(("CLR", "DII")):
        ax = fig.add_axes([0.055 + col * .49, 0.15, 0.40, 0.67])
        triangular_heatmap(fig, ax, motif_pairwise[(BEST_N, label)],
                           cmap="Blues" if label == "CLR" else "Reds", vmax=vmax,
                           title=label, cbar_label="Normalized weight" if col == 1 else None)
    finish(pdf, fig)

    # Baselines
    fig = new_page(pdf, "10. Singleton and pairwise baselines",
                   "Random forest, 100 trees, five seeds, patient-level leave-one-out cross-validation")
    sing_auc = singleton.groupby("seed").roc_auc.first()
    pair_auc = pairwise.groupby("seed").roc_auc.first()
    means = [sing_auc.mean(), pair_auc.mean(), summaries[3]["optimal_auc"], summaries[4]["optimal_auc"]]
    labels = ["Singleton\nfrequencies", "All pairwise\ninteractions", "3-node\nmotifs", "4-node\nmotifs"]
    colors = [GOLD, "#8A9BA8", CLR, GREEN]
    ax = fig.add_axes([0.11, 0.47, 0.78, 0.36])
    bars = ax.bar(labels, means, color=colors, width=.65)
    ax.axhline(.5, color=MID, ls=":")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(.4, .9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E9EDF0", lw=.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + .012, f"{value:.3f}", ha="center",
                fontsize=10, fontweight="bold")
    ax2 = fig.add_axes([0.15, 0.245, 0.70, 0.14])
    seeds = np.arange(5)
    ax2.plot(seeds, sing_auc.values, "o-", color=GOLD, label="Singleton")
    ax2.plot(seeds, pair_auc.values, "o-", color="#657985", label="Pairwise")
    ax2.set(xlabel="Random-forest seed", ylabel="ROC AUC", xticks=seeds, ylim=(.74, .85))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, ncol=2, fontsize=8)
    y = 0.18
    y = paragraph(fig, 0.08, y,
        "Singleton features are each patient's normalized frequencies of the 15 analysis categories. Pairwise "
        "features are normalized upper-triangle cell-type interaction frequencies from the full graphs. Both "
        "baselines were repeated inside each motif task, yielding identical files and means because their inputs "
        "do not depend on motif size.", width=108, fontsize=8.8)
    paragraph(fig, 0.08, y - 0.005,
        "Observed ordering: four-node motifs (0.824) > singleton frequencies (0.802) > pairwise interactions "
        "(0.775) > three-node motifs (0.742). Differences have not been tested for statistical significance.",
        width=108, fontsize=8.8, weight="bold")
    finish(pdf, fig)

    # Full sweep table
    fig = new_page(pdf, "Appendix A. Complete stringency sweep", landscape=True,
                   subtitle="A dash indicates that no discriminative feature set survived that threshold")
    rows = []
    for s in np.round(np.arange(0, .951, .05), 2):
        row = [f"{s:.2f}"]
        for n in (3, 4):
            rec = sweeps[n].loc[np.isclose(sweeps[n].stringency, s)].iloc[0]
            row.extend([
                "—" if pd.isna(rec.roc_auc) else f"{rec.roc_auc:.3f}",
                f"{rec.mean_selected_features:.2f}",
            ])
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["Stringency", "3-node AUC", "3-node mean features",
                                        "4-node AUC", "4-node mean features"])
    ax = fig.add_axes([0.10, 0.09, 0.80, 0.77])
    add_table(ax, frame, col_widths=[.18, .18, .23, .18, .23], fontsize=7.4, scale_y=1.18)
    finish(pdf, fig)

    # Mapping and provenance
    fig = new_page(pdf, "Appendix B. Cell-type mapping and provenance", landscape=True,
                   subtitle="Source labels were consolidated into 14 paper-level groups plus Unidentified")
    half = (len(mapping) + 1) // 2
    for i, part in enumerate((mapping.iloc[:half], mapping.iloc[half:])):
        ax = fig.add_axes([0.05 + i * 0.48, 0.25, 0.43, 0.60])
        add_table(ax, part.reset_index(drop=True), col_widths=[.58, .42], fontsize=7.0, scale_y=1.25)
    fig.text(0.06, 0.19, "Input provenance", fontsize=11, fontweight="bold", color=DARK)
    paragraph(fig, 0.06, 0.16,
        f"Cell table: {prep['source']}\nSHA-256: {prep['source_sha256']}\n"
        f"Clinical table: {prep['clinical_source']}\nSHA-256: {prep['clinical_sha256']}",
        width=155, fontsize=7.5, line_height=.018)
    finish(pdf, fig)

    # Deliverables and interpretation
    fig = new_page(pdf, "Appendix C. Deliverables and interpretation limits")
    y = 0.855
    fig.text(0.07, y, "Primary artifacts", fontsize=12.5, fontweight="bold", color=DARK, va="top")
    y -= .04
    artifacts = [
        ("Prepared graphs", str(DATA)),
        ("Three-node results", str(OUT / "motif_3/analysis_outputs")),
        ("Four-node results", str(OUT / "motif_4/analysis_outputs")),
        ("Three-node CISM pickle", summaries[3]["raw_cism_artifact"]),
        ("Four-node CISM pickle", summaries[4]["raw_cism_artifact"]),
    ]
    for label, path in artifacts:
        fig.text(0.08, y, label, fontsize=8.8, fontweight="bold", color=DARK, va="top")
        y = paragraph(fig, 0.25, y, path, width=95, fontsize=7.6, color=MID, line_height=.017)
        y -= .012
    y -= .01
    fig.text(0.07, y, "Files produced per motif size", fontsize=12.5, fontweight="bold", color=DARK, va="top")
    y -= .038
    y = paragraph(fig, 0.08, y,
        "FANMOD output and parsed caches for 140 FOVs; discriminative_motifs.csv; auc_by_stringency.csv; "
        "motif_count_vs_stringency.csv; per-threshold LOOCV tables; selected motif feature and weight tables; "
        "general and motif-induced pairwise matrices for CLR and DII; singleton and pairwise predictions; "
        "summary JSON; figures; raw CISM and pipeline pickle bundles.", width=110, fontsize=8.8)
    y -= .025
    fig.text(0.07, y, "Interpretation limits", fontsize=12.5, fontweight="bold", color=DARK, va="top")
    y -= .04
    limits = [
        "Only 35 patients were available; all performance estimates therefore have substantial sampling uncertainty.",
        "Stringency was optimized on this cohort without nested CV, which can inflate the selected peak AUC.",
        "No patient-label permutation test or confidence interval was produced by this run.",
        "The FANMOD count of 1,000 refers to graph randomizations for motif enumeration, not 1,000 independent patient analyses.",
        "Alpha-shape removal excluded 1,129 biological cells and differs from the original paper's released contact-analysis workflow.",
        "The singleton and pairwise baselines were recomputed in both array tasks, but they are duplicate analyses rather than independent replications.",
        "At stringencies above 0.55 no features survived, so the absent AUCs should not be interpreted as poor classifier performance.",
    ]
    for item in limits:
        fig.text(0.085, y, "•", fontsize=10, color="#8A3D2F", va="top")
        y = paragraph(fig, 0.105, y, item, width=100, fontsize=8.6, line_height=.019)
        y -= .012
    finish(pdf, fig)

print(PDF)
