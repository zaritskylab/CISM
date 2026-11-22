#!/usr/bin/env python3
"""Measure cell-type JS distances for fixed-0.46 shuffles with many motifs."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cism_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis.permutation_controls.make_cism_label_permutation_report import (
    build_motif_contribution_cache,
    build_motif_occurrence_table,
    motif_aware_cell_type_distribution,
    motif_mean_weights,
    parse_feature_list,
    read_patient_classes_frame,
    safe_jensen_shannon,
    assign_motifs_to_classes,
)
from analysis.permutation_controls.run_cism_label_permutation_test import DEFAULT_ARTIFACT, DEFAULT_OUTPUT_ROOT


def selected_ids(run_dir: Path) -> list[int]:
    features = pd.read_csv(run_dir / "fixed_motif_space_features.csv")
    if len(features) != 38 or features.test_patient_id.nunique() != 38:
        raise ValueError(f"Expected 38 LOOCV fold-feature rows in {run_dir}")
    return sorted({motif_id for value in features.features for motif_id in parse_feature_list(value)})


def motif_distribution(
    run_dir: Path,
    motif_ids: list[int],
    occurrences: pd.DataFrame,
    cell_names: dict[int, str],
    contributions: dict,
) -> tuple[pd.DataFrame, dict[str, int]]:
    classes = read_patient_classes_frame(run_dir)
    motifs_by_class = assign_motifs_to_classes(motif_ids, occurrences, classes, ["NN", "NP"])
    weights = motif_mean_weights(motif_ids, occurrences, classes, ["NN", "NP"])
    distribution = motif_aware_cell_type_distribution(motifs_by_class, weights, cell_names, contributions)
    return distribution, {label: len(motif_ids) for label, motif_ids in motifs_by_class.items()}


def vector(frame: pd.DataFrame, patient_class: str, cell_types: list[str]) -> np.ndarray:
    class_frame = frame.loc[frame.patient_class.eq(patient_class)].set_index("cell_type")
    if class_frame.index.duplicated().any() or not set(cell_types).issubset(class_frame.index):
        raise ValueError(f"Missing or duplicated {patient_class} cell types")
    return class_frame.reindex(cell_types).proportion.to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    parser.add_argument("--min-motifs", type=int, default=20)
    args = parser.parse_args()
    if args.min_motifs < 1:
        parser.error("--min-motifs must be positive")

    counts = pd.read_csv(args.output_dir / "fixed_fold_union_motif_counts.csv")
    counts = counts.loc[counts.fold_union_discriminative_motifs.ge(args.min_motifs)].copy()
    if counts.empty or counts.seed.duplicated().any():
        raise ValueError("No qualifying unique seeds")
    original_dir = args.run_root / "original/from_artifact"
    motif_ids_by_seed = {}
    all_ids = set(selected_ids(original_dir))
    for row in counts.itertuples():
        seed = int(row.seed)
        run_dir = args.run_root / "shuffled" / f"seed_{seed}"
        motif_ids = selected_ids(run_dir)
        if len(motif_ids) != row.fold_union_discriminative_motifs:
            raise ValueError(f"Motif-count CSV differs from fold files for seed {seed}")
        motif_ids_by_seed[seed] = motif_ids
        all_ids.update(motif_ids)

    with args.artifact.open("rb") as handle:
        artifact = pickle.load(handle)
    motif_table = artifact["discriminator"].cism.motifs_dataset
    cell_names = {int(k): v for k, v in artifact["discriminator"].common_cells.items()}
    cell_types = sorted(cell for cell in cell_names.values() if cell.lower() != "tumor")
    selected_table = motif_table.loc[motif_table.ID.isin(all_ids)]
    contributions = build_motif_contribution_cache(selected_table, cell_names, all_ids)
    if set(all_ids) != set(contributions):
        raise ValueError("Some selected motif graphs are missing from the artifact")
    occurrences = build_motif_occurrence_table(selected_table)
    original_general = pd.read_csv(original_dir / "general_cell_type_distribution.csv")
    original_motif, original_class_counts = motif_distribution(
        original_dir, selected_ids(original_dir), occurrences, cell_names, contributions)
    if any(count == 0 for count in original_class_counts.values()):
        raise ValueError("Original run has no motifs assigned to one class")

    records = []
    for seed, motif_ids in motif_ids_by_seed.items():
        run_dir = args.run_root / "shuffled" / f"seed_{seed}"
        shuffled_motif, class_counts = motif_distribution(
            run_dir, motif_ids, occurrences, cell_names, contributions)
        for patient_class in ("NN", "NP"):
            shuffled = vector(shuffled_motif, patient_class, cell_types)
            general = vector(original_general, patient_class, cell_types)
            original = vector(original_motif, patient_class, cell_types)
            has_distribution = class_counts[patient_class] > 0 and shuffled.sum() > 0
            records.append({
                "seed": seed,
                "fold_union_discriminative_motifs": len(motif_ids),
                "patient_class": patient_class,
                "n_class_assigned_motifs": class_counts[patient_class],
                "js_to_original_general": (
                    safe_jensen_shannon(shuffled, general) if has_distribution else np.nan),
                "js_to_original_motif_induced": (
                    safe_jensen_shannon(shuffled, original) if has_distribution else np.nan),
                "original_motif_to_general_js": safe_jensen_shannon(original, general),
            })
    distances = pd.DataFrame(records)
    stem = f"fixed_046_min_{args.min_motifs}_motifs_cell_type_js"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distances.to_csv(args.output_dir / f"{stem}_distances.csv", index=False)
    summary = distances.groupby("patient_class", as_index=False).agg(
        n_shuffles=("seed", "nunique"),
        n_defined_distances=("js_to_original_general", "count"),
        mean_js_to_original_general=("js_to_original_general", "mean"),
        median_js_to_original_general=("js_to_original_general", "median"),
        min_js_to_original_general=("js_to_original_general", "min"),
        max_js_to_original_general=("js_to_original_general", "max"),
        original_motif_to_general_js=("original_motif_to_general_js", "first"),
    )
    summary.to_csv(args.output_dir / f"{stem}_summary.csv", index=False)

    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, patient_class, color in zip(axes, ("NN", "NP"), ("#24558c", "#d97824")):
            class_data = distances.loc[distances.patient_class.eq(patient_class)]
            defined = class_data.js_to_original_general.dropna()
            ax.hist(defined, bins=np.linspace(0, 1, 21),
                    color=color, edgecolor="white", linewidth=0.6)
            reference = float(class_data.original_motif_to_general_js.iloc[0])
            ax.axvline(reference, color="#b34034", linewidth=1.5,
                       label=f"Original motif-induced = {reference:.3f}")
            ax.set(xlim=(0, 1), xlabel="JS distance to original general cell-type distribution",
                   title=f"{patient_class} (n = {len(defined)} defined)")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("Number of shuffles")
        fig.suptitle(f"Fixed 0.46: shuffles with at least {args.min_motifs} fold-union motifs")
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"{stem}_histogram.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
