#!/usr/bin/env python3
"""Plot CRC class-exclusive motif counts using the existing CISM discovery rule.

Run from any directory with Python, pandas, numpy and matplotlib installed.
The default artifacts are the saved CRC FANMOD analyses (1000 iterations).
Only load trusted local pickle files.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd


class LegacyCISMArtifact:
    """State container for old CISM pickles; no old package imports required."""


class LegacyReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {("CISM.cism.cism", "CISM"), ("cism.cism", "CISM")}:
            return LegacyCISMArtifact
        return super().find_class(module, name)


def extract_motifs(artifact, labels, size):
    if artifact.motif_size != size:
        raise ValueError(f"Expected {size}-node artifact, got {artifact.motif_size}")
    data = artifact.motifs_dataset.copy()
    data["patient_class"] = data.Patient.astype(int).map(labels)
    if data.patient_class.isna().any():
        raise ValueError("Some motif patients have no CRC label")
    significant = data.loc[data.p_value < 0.05]
    # Matches TissueStateDiscriminativeMotifs._extract_discriminative with ID
    # (STRUCTURE_AND_CELL_IDENTITIES). Count patients, never FOVs or instances.
    denominators = significant.groupby("patient_class").Patient_uId.nunique()
    cohort = data.groupby("patient_class").Patient_uId.nunique()
    if not denominators.equals(cohort):
        raise ValueError("Some patients have no significant motifs; review denominator")
    exclusive_ids = significant.groupby("ID").patient_class.nunique().loc[lambda x: x == 1].index
    exclusive = significant.loc[significant.ID.isin(exclusive_ids) & (significant.nunique_colors >= 1)]
    motifs = exclusive.groupby(["ID", "patient_class"]).Patient_uId.nunique().rename("patient_count").reset_index()
    motifs["class_patient_count"] = motifs.patient_class.map(denominators)
    motifs["patient_percentage"] = motifs.patient_count / motifs.class_patient_count
    motifs.insert(0, "motif_size", size)
    return motifs, cohort.to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/groups/assafza_group/assafza/Datasets/CRC"))
    parser.add_argument("--artifact-3", type=Path, default=Path("/home/omriavit/research/cism_CRC.pkl"))
    parser.add_argument("--artifact-4", type=Path, default=Path("/home/omriavit/research/cism_CRC_4.pkl"))
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "outputs/crc_stringency")
    parser.add_argument("--curve", choices=["step", "line"], default="step",
                        help="Connect class prevalence boundaries with lines for a smoother visual")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.data_dir / "CRC_patient_class.csv"
    raw_labels = pd.read_csv(label_path)[["Patient", "Group"]].dropna()
    raw_labels = raw_labels.astype({"Patient": int, "Group": int})
    if raw_labels.Patient.duplicated().any() or not raw_labels.Group.isin([1, 2]).all():
        raise ValueError("Expected unique patient labels with groups 1=CLR, 2=DII")
    labels = raw_labels.set_index("Patient").Group.map({1: "CLR", 2: "DII"})
    sources, tables = [], []
    for size, path in [(3, args.artifact_3), (4, args.artifact_4)]:
        with path.open("rb") as handle:
            artifact = LegacyReader(handle).load()
        table, cohort = extract_motifs(artifact, labels, size)
        if set(artifact.motifs_dataset.Patient.astype(int)) != set(labels.index):
            raise ValueError("Artifact and labeled patient cohorts differ")
        tables.append(table)
        sources.append({"motif_size": size, "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "iterations": artifact.iterations, "patients": cohort})
    motifs = pd.concat(tables, ignore_index=True)
    # Include every prevalence boundary, as well as a regular grid.
    thresholds = np.unique(np.r_[np.linspace(0, 1, 101), motifs.patient_percentage])
    rows = []
    for (size, group), table in motifs.groupby(["motif_size", "patient_class"]):
        for threshold in thresholds:
            rows.append({"motif_size": size, "patient_class": group,
                         "stringency": threshold,
                         "n_discriminative_motifs": int((table.patient_percentage > threshold).sum())})
    counts = pd.DataFrame(rows)
    for _, table in counts.groupby(["motif_size", "patient_class"]):
        assert (np.diff(table.n_discriminative_motifs) <= 0).all()
        assert table.iloc[-1].n_discriminative_motifs == 0
    motifs.to_csv(args.output_dir / "crc_discriminative_motifs.csv", index=False)
    counts.to_csv(args.output_dir / "crc_motif_count_vs_stringency.csv", index=False)
    labels.rename("patient_class").to_csv(args.output_dir / "crc_patient_labels.csv")

    colors = {3: "#49ad2b", 4: "#7632a8"}
    styles = {"CLR": (0, (5, 4)), "DII": (0, (0.6, 2.1))}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.4, 4.7))
        for (size, group), table in counts.groupby(["motif_size", "patient_class"]):
            if args.curve == "line":
                # Use one point at each attainable class prevalence, avoiding
                # the artificial near-vertical segments of the dense CSV grid.
                n_patients = cohort[group]
                boundaries = np.arange(n_patients + 1) / n_patients
                prevalence = motifs.loc[(motifs.motif_size == size) &
                                        (motifs.patient_class == group), "patient_percentage"]
                table = pd.DataFrame({"stringency": boundaries,
                                      "n_discriminative_motifs": [int((prevalence > t).sum()) for t in boundaries]})
            values = table.n_discriminative_motifs.to_numpy(dtype=float)
            values[values == 0] = np.nan  # Zero is undefined on a log axis.
            ax.plot(table.stringency, values,
                    drawstyle="steps-post" if args.curve == "step" else "default", color=colors[size],
                    linestyle=styles[group], linewidth=2.7, dash_capstyle="round")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 10 ** np.ceil(np.log10(counts.n_discriminative_motifs.max() * 1.2)))
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Discrimination stringency parameter", labelpad=10)
        ax.set_ylabel("Number of discriminative motifs", labelpad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bbbbbb")
        ax.tick_params(axis="both", colors="#555555", which="both")
        size_legend = ax.legend(handles=[Line2D([], [], color=colors[s], lw=3, label=f"{s}-nodes") for s in [3, 4]],
                                loc="upper right", bbox_to_anchor=(1, 1.02), frameon=False)
        ax.add_artist(size_legend)
        ax.legend(handles=[Line2D([], [], color="#555555", lw=2.7, linestyle=styles[g],
                                  dash_capstyle="round", label=f"{g} (n={cohort[g]})") for g in ["CLR", "DII"]],
                  loc="upper right", bbox_to_anchor=(1, 0.82), frameon=False, handlelength=3)
        fig.tight_layout()
        suffix = "_smooth" if args.curve == "line" else ""
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(args.output_dir / f"crc_motifs_vs_stringency{suffix}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    metadata = {"label_source": str(label_path), "sources": sources,
                "rule": "p_value < 0.05; ID exclusive to one class; nunique_colors >= 1; patient_percentage > stringency",
                "denominator": "Unique patients with significant motifs in each class, verified equal to full cohort",
                "zero_counts": "Included in CSV; omitted from logarithmic plot",
                "curve": args.curve,
                "line_plot": "Straight lines between class prevalence boundaries; visual interpolation only"}
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# CRC discriminative motifs versus stringency\n\n"
        "Green: 3-node motifs; purple: 4-node motifs. Dashed: CLR (17 patients); dotted: DII (18 patients). "
        "Each curve counts distinct motif IDs (structure and cell identities), significant at p < 0.05 in "
        "at least one patient and significant exclusively in that class. Stringency is the fraction of "
        "patients in the class: a motif is retained only when its patient prevalence is strictly greater "
        "than the threshold. Multiple FOVs count once per patient. This is descriptive full-cohort discovery, "
        "with no feature cap or classifier fitting. Counts of zero are recorded in the CSV and omitted "
        "from the logarithmic plot.\n\n"
        "The additional `_smooth` figure connects counts at class prevalence boundaries with straight "
        "lines (linear in log-count display space). This is visual interpolation; the discrete counts "
        "in the source tables remain unchanged. Generate it with `--curve line`.\n\n"
        "Reproduce from the repository root: `python analysis/crc_characterization/plot_crc_motifs_vs_stringency.py`. "
        "The source artifacts are existing 1000-iteration FANMOD results, not a new graph or FANMOD run. "
        "Paths, hashes, and selection settings are in `provenance.json`.\n")
    print(counts.loc[counts.stringency.isin([0, 0.5, 1])].to_string(index=False))
    print(f"Saved figure and source tables to {args.output_dir}")


if __name__ == "__main__":
    main()
