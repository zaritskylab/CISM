#!/usr/bin/env python3
"""Recompute split-FOV stability using LOOCV fold-union motifs at stringency 0.46."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import gc
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.cohort_imbalance.imbalanced_bootstrap_stringency import (  # noqa: E402
    discover_hard_discriminative_motifs,
    load_patient_classes,
    select_motifs_by_label,
)


RUNTIME = REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
ORIGINAL_ARTIFACT = RUNTIME / "serialized" / "Melanoma_cism_ready.pkl"
SPLIT_DIR = RUNTIME / "split_fov_scale_experiment"
OUTPUT_DIR = SPLIT_DIR / "fold_union_stringency_046_stability"

LABELS = ["NN", "NP"]
PARTS = [2, 4, 8, 16]
STRINGENCY = 0.46
MAX_FEATURES = 30
P_VALUE_THRESHOLD = 0.05
RANDOM_REPEATS = 20
RANDOM_STATE = 0


def load_artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_motifs(path: Path) -> pd.DataFrame:
    artifact = load_artifact(path)
    return artifact["cism"].motifs_dataset.copy()


def fold_union_selected_motifs(
    motifs: pd.DataFrame,
    patient_classes: pd.DataFrame,
    cohort_patients: list[str],
) -> tuple[dict[str, set[int]], pd.DataFrame]:
    union_by_label = {label: set() for label in LABELS}
    rows = []
    for held_out in cohort_patients:
        train_patients = [patient for patient in cohort_patients if patient != held_out]
        summary = discover_hard_discriminative_motifs(
            motifs_df=motifs,
            patient_classes=patient_classes,
            cohort_patients=train_patients,
            p_value_threshold=P_VALUE_THRESHOLD,
        )
        selected = select_motifs_by_label(
            discriminative_summary=summary,
            labels=LABELS,
            stringency=STRINGENCY,
            max_features=MAX_FEATURES,
        )
        for label, ids in selected.items():
            ids = set(map(int, ids))
            union_by_label[label].update(ids)
            for motif_id in sorted(ids):
                rows.append({"held_out_patient": held_out, "label": label, "ID": motif_id})
    return union_by_label, pd.DataFrame(rows)


def as_set(union_by_label: dict[str, set[int]]) -> set[int]:
    return set().union(*union_by_label.values())


def stability_row(
    scenario: str,
    parts: int | str,
    repeat: int | None,
    reference: set[int],
    union_by_label: dict[str, set[int]],
) -> dict:
    selected = as_set(union_by_label)
    shared = selected & reference
    union = selected | reference
    return {
        "scenario": scenario,
        "grid_parts": parts,
        "repeat": repeat,
        "reference_motif_count": len(reference),
        "selected_motif_count": len(selected),
        "selected_NN": len(union_by_label["NN"]),
        "selected_NP": len(union_by_label["NP"]),
        "shared_with_reference": len(shared),
        "reference_recovered_fraction": len(shared) / len(reference) if reference else np.nan,
        "selected_from_reference_fraction": len(shared) / len(selected) if selected else np.nan,
        "jaccard": len(shared) / len(union) if union else np.nan,
    }


def plot_stability(stability: pd.DataFrame) -> None:
    agg = stability[stability["scenario"].eq("all_split_fovs_aggregated")]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(agg["grid_parts"], agg["jaccard"], marker="o", label="Jaccard")
    ax.plot(
        agg["grid_parts"],
        agg["reference_recovered_fraction"],
        marker="o",
        label="Reference recovered",
    )
    ax.set_xlabel("Grid parts per FOV")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Split fold-union stability vs native 19 motifs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_split_fovs_fold_union_stability.png", dpi=200)
    plt.close(fig)

    random = stability[stability["scenario"].eq("random_one_of_split2")]
    if not random.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.boxplot(
            [random["jaccard"], random["reference_recovered_fraction"]],
            labels=["Jaccard", "Reference recovered"],
            showfliers=False,
        )
        ax.set_ylim(0, 1)
        ax.set_title("Random one-of-two split-2 fold-union stability")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "random_split2_fold_union_stability.png", dpi=200)
        plt.close(fig)


def write_report(reference_by_label: dict[str, set[int]], stability: pd.DataFrame) -> Path:
    agg = stability[stability["scenario"].eq("all_split_fovs_aggregated")].copy()
    rand = stability[stability["scenario"].eq("random_one_of_split2")].copy()
    if rand.empty:
        rand_summary = pd.DataFrame()
    else:
        rand_summary = pd.DataFrame(
            [
                {
                    "repeats": int(rand["repeat"].count()),
                    "mean_selected_motif_count": float(rand["selected_motif_count"].mean()),
                    "mean_shared_with_reference": float(rand["shared_with_reference"].mean()),
                    "mean_reference_recovered_fraction": float(
                        rand["reference_recovered_fraction"].mean()
                    ),
                    "mean_jaccard": float(rand["jaccard"].mean()),
                    "sd_jaccard": float(rand["jaccard"].std()),
                }
            ]
        )

    def md_table(df: pd.DataFrame) -> str:
        view = df.fillna("")
        header = "| " + " | ".join(map(str, view.columns)) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
        return "\n".join([header, sep, *rows])

    path = OUTPUT_DIR / "fold_union_stringency_046_stability_report.md"
    lines = [
        "# Corrected Split-FOV Stability: Fold-Union Motifs At Stringency 0.46",
        "",
        "This recomputes stability using the motif set the original downstream analysis used: the union of motifs selected across LOOCV folds at stringency 0.46.",
        "",
        f"Native reference motifs: {len(as_set(reference_by_label))} total ({len(reference_by_label['NN'])} NN, {len(reference_by_label['NP'])} NP).",
        "",
        "## All Split FOVs Aggregated",
        "",
        md_table(agg),
        "",
        "## Random One-Of-Two Split-2 Tiles",
        "",
        f"Random repeats: {RANDOM_REPEATS}",
        "",
        md_table(rand_summary),
        "",
        f"![all split stability]({(OUTPUT_DIR / 'all_split_fovs_fold_union_stability.png').resolve()})",
        "",
        f"![random split stability]({(OUTPUT_DIR / 'random_split2_fold_union_stability.png').resolve()})",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_artifact = load_artifact(ORIGINAL_ARTIFACT)
    patient_classes = load_patient_classes(original_artifact["discriminator"], LABELS)
    cohort_patients = patient_classes.index.tolist()

    original_motifs = original_artifact["cism"].motifs_dataset.copy()
    reference_by_label, reference_rows = fold_union_selected_motifs(
        original_motifs,
        patient_classes,
        cohort_patients,
    )
    reference_rows.to_csv(OUTPUT_DIR / "native_fold_union_selected_motifs_by_fold.csv", index=False)
    pd.DataFrame(
        [
            {"label": label, "ID": motif_id}
            for label, motif_ids in reference_by_label.items()
            for motif_id in sorted(motif_ids)
        ]
    ).to_csv(OUTPUT_DIR / "native_fold_union_selected_motifs.csv", index=False)

    rows = []
    fold_rows = []
    reference = as_set(reference_by_label)
    print(
        f"Native reference: {len(reference)} motifs ({len(reference_by_label['NN'])} NN, {len(reference_by_label['NP'])} NP)",
        flush=True,
    )

    for parts in PARTS:
        print(f"Loading split {parts} artifact...", flush=True)
        motifs = load_motifs(SPLIT_DIR / f"split_{parts:02d}_cism_ready.pkl")
        selected_by_label, selected_rows = fold_union_selected_motifs(motifs, patient_classes, cohort_patients)
        selected_rows.insert(0, "scenario", "all_split_fovs_aggregated")
        selected_rows.insert(1, "grid_parts", parts)
        fold_rows.append(selected_rows)
        rows.append(stability_row("all_split_fovs_aggregated", parts, None, reference, selected_by_label))
        del motifs, selected_rows, selected_by_label
        gc.collect()

    print("Loading split 2 artifact for random one-of-two scenario...", flush=True)
    split2 = load_motifs(SPLIT_DIR / "split_02_cism_ready.pkl")
    metadata = pd.read_csv(SPLIT_DIR / "split_02_metadata.csv")
    tile_options = {
        key: group["split_fov"].astype(str).to_numpy()
        for key, group in metadata.groupby(["patient_uid", "source_fov"], observed=True)
    }
    rng = np.random.default_rng(RANDOM_STATE)
    for repeat in range(RANDOM_REPEATS):
        if repeat % 5 == 0:
            print(f"Random repeat {repeat}/{RANDOM_REPEATS}", flush=True)
        chosen_fovs = {str(rng.choice(options)) for options in tile_options.values()}
        motifs = split2[split2["FOV"].astype(str).isin(chosen_fovs)].copy()
        selected_by_label, selected_rows = fold_union_selected_motifs(motifs, patient_classes, cohort_patients)
        selected_rows.insert(0, "scenario", "random_one_of_split2")
        selected_rows.insert(1, "grid_parts", 2)
        selected_rows.insert(2, "repeat", repeat)
        fold_rows.append(selected_rows)
        rows.append(stability_row("random_one_of_split2", 2, repeat, reference, selected_by_label))

    stability = pd.DataFrame(rows)
    stability.to_csv(OUTPUT_DIR / "split_fold_union_stability.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "split_fold_union_selected_motifs_by_fold.csv",
        index=False,
    )
    plot_stability(stability)
    report = write_report(reference_by_label, stability)
    summary = {
        "stringency": STRINGENCY,
        "max_features": MAX_FEATURES,
        "native_reference_total": len(reference),
        "native_reference_NN": len(reference_by_label["NN"]),
        "native_reference_NP": len(reference_by_label["NP"]),
        "random_repeats": RANDOM_REPEATS,
        "output_dir": str(OUTPUT_DIR),
        "report": str(report),
    }
    (OUTPUT_DIR / "fold_union_stringency_046_stability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
