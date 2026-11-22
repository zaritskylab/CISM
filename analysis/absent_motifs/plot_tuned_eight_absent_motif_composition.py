from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


REPO_ROOT = _REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

from cism.helpers import string_base64_pickle  # noqa: E402


DEFAULT_BASE_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
)
DEFAULT_CISM_PATH = DEFAULT_BASE_DIR / "Melanoma_NN_NP_absent_cism_ready.pkl"
DEFAULT_STRINGENCY = 0.3642694428844888

CELL_TYPE_ORDER = [
    "b cell",
    "cd4 apc",
    "cd4 t cell",
    "cd8 t cell",
    "DC",
    "gc",
    "hevs",
    "mac",
    "memory%cd4 t cell",
    "nk cell",
    "neutrophil",
    "stroma",
    "unidentified",
    "blood vessels",
]

PANEL_B_ORDER = [
    "cd8 t cell",
    "mac",
    "b cell",
    "unidentified",
    "treg",
    "DC",
    "hevs",
    "cd4 t cell",
    "blood vessels",
    "cd4 apc",
    "stroma",
    "gc",
    "memory%cd4 t cell",
    "neutrophil",
    "nk cell",
]

DISPLAY_LABELS = {
    "b cell": "B cell",
    "cd4 apc": "CD4 APC",
    "cd4 t cell": "CD4 T cell",
    "cd8 t cell": "CD8 T cell",
    "DC": "DCs",
    "gc": "Germinal Centers",
    "hevs": "HEVs",
    "mac": "MAC",
    "memory%cd4 t cell": "Memory CD4 T cell",
    "nk cell": "NK cell",
    "neutrophil": "Neutrophil",
    "stroma": "Stroma",
    "unidentified": "Unidentified",
    "blood vessels": "Vessels",
    "treg": "CD4 Treg",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cell-type composition for the 8 tuned-stringency underrepresented motifs."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--cism-path", type=Path, default=DEFAULT_CISM_PATH)
    parser.add_argument("--stringency", type=float, default=DEFAULT_STRINGENCY)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BASE_DIR / "tuned_stringency_8_motif_cell_type_composition.png",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=DEFAULT_BASE_DIR / "tuned_stringency_8_motif_cell_type_composition.csv",
    )
    parser.add_argument(
        "--fractions-output",
        type=Path,
        default=DEFAULT_BASE_DIR / "tuned_stringency_8_motif_cell_type_fractions.csv",
    )
    parser.add_argument(
        "--motif-output",
        type=Path,
        default=DEFAULT_BASE_DIR / "tuned_stringency_8_discriminative_motifs.csv",
    )
    parser.add_argument(
        "--panel-b-output",
        type=Path,
        default=DEFAULT_BASE_DIR / "tuned_stringency_8_motif_cell_type_fractions_panel_b.png",
    )
    parser.add_argument(
        "--composite-output",
        type=Path,
        default=DEFAULT_BASE_DIR / "tuned_stringency_8_motif_composition_panel_ab.png",
    )
    args = parser.parse_args()

    selected = pd.read_csv(args.base_dir / "absent_discriminative_motifs.csv")
    target = selected[selected["patient_percentage"] > args.stringency].copy()
    capped = (
        target.sort_values("Freq", ascending=False)
        .groupby("patient_class", observed=True)
        .head(30)
    )
    motif_rows = (
        capped.sort_values(["patient_class", "ID"])
        .drop_duplicates(["patient_class", "ID"])
        .reset_index(drop=True)
    )

    with args.cism_path.open("rb") as handle:
        payload = pickle.load(handle)
    cism = payload["cism"]
    cell_type_by_code = {str(code): label for code, label in payload["common_cells_type"].items()}
    motif_lookup = cism.motifs_dataset.drop_duplicates("ID").set_index("ID")["motif"]

    composition_rows = []
    for row in motif_rows.itertuples(index=False):
        graph = deserialize_graph(motif_lookup.loc[row.ID])
        counts = {cell_type: 0 for cell_type in sorted(set(CELL_TYPE_ORDER + PANEL_B_ORDER))}
        for _node, data in graph.nodes(data=True):
            cell_type = cell_type_by_code.get(str(data.get("type")), str(data.get("type")))
            if cell_type in counts:
                counts[cell_type] += 1
        composition_rows.append(
            {
                "patient_class": row.patient_class,
                "ID": int(row.ID),
                "patient_count": int(row.patient_count),
                "patient_percentage": float(row.patient_percentage),
                **counts,
            }
        )

    composition = pd.DataFrame(composition_rows)
    composition.to_csv(args.matrix_output, index=False)
    fractions = create_fraction_table(composition)
    fractions.to_csv(args.fractions_output, index=False)
    composition[
        ["patient_class", "ID", "patient_count", "patient_percentage"]
    ].to_csv(args.motif_output, index=False)
    plot_composition(composition, args.output)
    plot_fraction_bars(fractions, args.panel_b_output)
    plot_panel_ab(composition, fractions, args.composite_output)
    print(args.output)
    print(args.matrix_output)
    print(args.fractions_output)
    print(args.motif_output)
    print(args.panel_b_output)
    print(args.composite_output)


def deserialize_graph(motif_obj):
    if isinstance(motif_obj, (nx.Graph, nx.DiGraph)):
        return motif_obj
    if isinstance(motif_obj, str):
        return string_base64_pickle(motif_obj)
    raise TypeError(f"Unsupported motif graph representation: {type(motif_obj)!r}")


def plot_composition(composition: pd.DataFrame, output: Path) -> None:
    composition = composition.sort_values(["patient_class", "ID"]).reset_index(drop=True)
    matrix = composition[CELL_TYPE_ORDER].to_numpy()
    row_labels = [f"{row.patient_class} | {int(row.ID)}" for row in composition.itertuples(index=False)]
    nn_count = int((composition["patient_class"] == "NN").sum())
    np_count = int((composition["patient_class"] == "NP").sum())

    cmap = ListedColormap(["#ffffff", "#a57a43", "#000000"])
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=220)
    fig.patch.set_facecolor("white")
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(CELL_TYPE_ORDER)))
    ax.set_xticklabels([DISPLAY_LABELS[cell_type] for cell_type in CELL_TYPE_ORDER], rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.tick_params(axis="both", length=0)

    ax.set_ylabel("Discriminative motifs\ncell-type composition", fontsize=10)
    ax.set_title("Tuned-stringency 8 discriminative underrepresented motifs", fontsize=12, fontweight="bold")
    ax.axhline(nn_count - 0.5, color="#5e5e5e", linewidth=1.0, linestyle=(0, (3, 3)))

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = [
        Patch(facecolor="#000000", edgecolor="#000000", label="2"),
        Patch(facecolor="#a57a43", edgecolor="#a57a43", label="1"),
    ]
    ax.legend(
        handles=legend,
        title="Cell type\ncount",
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.04),
        frameon=True,
        fancybox=False,
        edgecolor="#111111",
        framealpha=1,
        fontsize=7,
        title_fontsize=7,
    )

    bracket_x = len(CELL_TYPE_ORDER) + 0.18
    draw_group_bracket(ax, bracket_x, -0.35, nn_count - 0.65, "NN", "#3f8fd2")
    draw_group_bracket(ax, bracket_x, nn_count - 0.35, nn_count + np_count - 0.65, "NP", "#d64a4a")

    ax.set_xlim(-0.5, len(CELL_TYPE_ORDER) + 1.55)
    ax.set_ylim(len(composition) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def create_fraction_table(composition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_class, group in composition.groupby("patient_class", observed=True):
        totals = group[PANEL_B_ORDER].sum()
        denominator = float(totals.sum())
        for cell_type in PANEL_B_ORDER:
            rows.append(
                {
                    "patient_class": patient_class,
                    "cell_type": cell_type,
                    "cell_type_label": DISPLAY_LABELS[cell_type],
                    "count": int(totals[cell_type]),
                    "fraction": float(totals[cell_type] / denominator) if denominator else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_fraction_bars(fractions: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=220)
    fig.patch.set_facecolor("white")
    draw_fraction_bars(ax, fractions)
    ax.set_title("Cell-type fractions in depleted discriminative motifs", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_panel_ab(composition: pd.DataFrame, fractions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.7),
        dpi=220,
        gridspec_kw={"width_ratios": [1.22, 1.0], "wspace": 0.32},
    )
    fig.patch.set_facecolor("white")
    draw_composition_matrix(axes[0], composition, title=None, show_row_labels=False)
    draw_fraction_bars(axes[1], fractions)
    axes[0].text(-0.18, 1.04, "A", transform=axes[0].transAxes, fontsize=15, fontweight="bold")
    axes[1].text(-0.25, 1.04, "B", transform=axes[1].transAxes, fontsize=15, fontweight="bold")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def draw_composition_matrix(
    ax,
    composition: pd.DataFrame,
    title: str | None = "Tuned-stringency 8 discriminative underrepresented motifs",
    show_row_labels: bool = True,
) -> None:
    composition = composition.sort_values(["patient_class", "ID"]).reset_index(drop=True)
    matrix = composition[CELL_TYPE_ORDER].to_numpy()
    row_labels = [f"{row.patient_class} | {int(row.ID)}" for row in composition.itertuples(index=False)]
    nn_count = int((composition["patient_class"] == "NN").sum())
    np_count = int((composition["patient_class"] == "NP").sum())

    cmap = ListedColormap(["#ffffff", "#a57a43", "#000000"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(CELL_TYPE_ORDER)))
    ax.set_xticklabels([DISPLAY_LABELS[cell_type] for cell_type in CELL_TYPE_ORDER], rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels if show_row_labels else ["" for _ in row_labels], fontsize=7)
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel("Discriminative motifs\ncell-type composition", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(nn_count - 0.5, color="#5e5e5e", linewidth=1.0, linestyle=(0, (3, 3)))
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = [
        Patch(facecolor="#000000", edgecolor="#000000", label="2"),
        Patch(facecolor="#a57a43", edgecolor="#a57a43", label="1"),
    ]
    ax.legend(
        handles=legend,
        title="Cell type\ncount",
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.04),
        frameon=True,
        fancybox=False,
        edgecolor="#111111",
        framealpha=1,
        fontsize=7,
        title_fontsize=7,
    )

    bracket_x = len(CELL_TYPE_ORDER) + 0.18
    draw_group_bracket(ax, bracket_x, -0.35, nn_count - 0.65, "NN", "#3f8fd2")
    draw_group_bracket(ax, bracket_x, nn_count - 0.35, nn_count + np_count - 0.65, "NP", "#d64a4a")
    ax.set_xlim(-0.5, len(CELL_TYPE_ORDER) + 1.55)
    ax.set_ylim(len(composition) - 0.5, -0.5)


def draw_fraction_bars(ax, fractions: pd.DataFrame) -> None:
    pivot = (
        fractions.pivot(index="cell_type", columns="patient_class", values="fraction")
        .reindex(PANEL_B_ORDER)
        .fillna(0.0)
    )
    y = range(len(PANEL_B_ORDER))
    height = 0.32
    ax.barh([value - height / 2 for value in y], pivot.get("NN", 0), height=height, color="#5b9bd5", label="NN")
    ax.barh([value + height / 2 for value in y], pivot.get("NP", 0), height=height, color="#c95757", label="NP")
    ax.set_yticks(list(y))
    ax.set_yticklabels([DISPLAY_LABELS[cell_type] for cell_type in PANEL_B_ORDER], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction", fontsize=9)
    ax.set_ylabel("Cell type", fontsize=9)
    ax.set_xlim(0, max(0.32, float(pivot.max().max()) * 1.12))
    ax.legend(frameon=False, loc="center right")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_group_bracket(ax, x: float, y0: float, y1: float, label: str, color: str) -> None:
    ax.plot([x, x + 0.18, x + 0.18, x], [y0, y0, y1, y1], color=color, linewidth=1.5, clip_on=False)
    ax.text(x + 0.34, (y0 + y1) / 2, label, va="center", ha="left", fontsize=11, color="#111111")


if __name__ == "__main__":
    main()
