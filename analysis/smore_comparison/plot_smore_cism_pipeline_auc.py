#!/usr/bin/env python3
"""Compare CISM benchmark AUC with SMORE motifs evaluated by CISM's pipeline."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.smore_comparison.compact_smore_results import (
    CELL_TYPE_LABELS,
    DEFAULT_METADATA_DIR,
    DEFAULT_SMORE_RESULTS,
    load_labeled_smore,
)
from analysis.smore_comparison.evaluate_smore_matched import (
    OUTPUT as NATIVE_OUTPUT,
    RUN_TAG,
    check_outputs,
    load_catalog,
    validate_manifest,
)
ROOT = _REPO_ROOT
RESEARCH_CISM = Path("/home/omriavit/research/CISM")
OUTPUT = ROOT / "analysis/outputs/stringency_clean"
THRESHOLDS = (0.20, 0.30, 0.40, 0.42, 0.44, 0.46)
CISM_AUC = (0.3847222222222222, 0.5333333333333333,
            0.7000000000000001, 0.7819444444444444,
            0.7805555555555556, 0.8916666666666667)


def load_smore(source: str) -> pd.DataFrame:
    if source == "provisional":
        labeled = load_labeled_smore(DEFAULT_SMORE_RESULTS, DEFAULT_METADATA_DIR)
        labeled = labeled.rename(columns={"patient number": "patient"})
        labeled["p_value"] = 0.01
        return labeled

    manifest = pd.read_csv(NATIVE_OUTPUT / "cohort_fovs.csv")
    validate_manifest(manifest)
    run_dir = NATIVE_OUTPUT / "runs" / RUN_TAG
    status = check_outputs(manifest, run_dir)
    if not (status.mat & status.catalog).all():
        raise RuntimeError("Native SMORE extraction is incomplete; no native AUC plot will be made")
    catalog, _ = load_catalog(manifest, run_dir)
    return catalog.rename(columns={"consensus_cell_types": "seed",
                                   "sampled_frequency": "frequency"})


def run_cism_pipeline(
    labeled: pd.DataFrame, output_dir: Path,
    thresholds: tuple[float, ...] = THRESHOLDS,
    identity: str = "seed", feature_mode: str = "motif_only",
    expected_class_counts: dict[str, int] | None = None,
    save_fold_predictions: bool = False,
) -> pd.DataFrame:
    # The notebook's CISM fork has the requested failed-fold policies.
    sys.path.insert(0, str(RESEARCH_CISM))
    from auxiliary.smore_adapter import smore_to_cism_motifs
    from cism.cism import (
        CISM,
        DiscriminativeFeatureKey,
        HardDiscriminativeFC,
        TissueStateDiscriminativeMotifs,
    )

    cohort = labeled[["patient", "class"]].drop_duplicates().sort_values("patient")
    expected_class_counts = expected_class_counts or {"NN": 20, "NP": 18}
    expected_patients = sum(expected_class_counts.values())
    if cohort.patient.nunique() != expected_patients or cohort.groupby("class").patient.nunique().to_dict() != expected_class_counts:
        raise ValueError(f"SMORE data must contain {expected_class_counts}")
    if labeled.groupby("patient")["class"].nunique().max() != 1:
        raise ValueError("Conflicting class labels")
    labels_file = output_dir / "smore_cism_pipeline_patient_classes.csv"
    class_codes = cohort["class"].map({"NN": 0, "NP": 1})
    pd.DataFrame({"patient": "SMORE" + cohort.patient.astype(str),
                  "class_code": class_codes}).to_csv(labels_file, index=False, header=False)

    if identity not in {"seed", "motifNumber"}:
        raise ValueError(f"Unsupported identity: {identity}")
    if feature_mode not in {"motif_only", "motif_and_composition"}:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    cism_data = smore_to_cism_motifs(
        labeled, fov_col="fov", patient_col="patient", seed_col="seed",
        score_col="frequency", id_col="seed" if identity == "seed" else None,
        canonicalize_id=False,
        p_value_default=0.01, dataset_name="SMORE",
    )
    # Override the adapter's placeholder when native SMORE p-values exist.
    cism_data["p_value"] = labeled.p_value.to_numpy(dtype=float)
    cism = CISM(fanmod_path="", fanmod_exe="", network_dataset_root_path="",
                fanmod_output_root_path="", fanmod_cache_root_path="",
                motif_size=4, iterations=100)
    cism.motifs_dataset = cism_data
    discriminator = TissueStateDiscriminativeMotifs(
        cism=cism, tissue_state_csv_path=str(labels_file),
        tissue_state_to_string={0: "NN", 1: "NP"},
        common_cells_type={int(k): v for k, v in CELL_TYPE_LABELS.items()},
    )

    rows = []
    for threshold in thresholds:
        config = HardDiscriminativeFC(
            labels=["NN", "NP"],
            extract_by=(DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES
                        if feature_mode == "motif_only"
                        else DiscriminativeFeatureKey.CELL_IDENTITIES),
            use_cells_type_composition=feature_mode == "motif_and_composition",
            use_motifs=True,
            shared_percentage=threshold,
            max_class_features=30,
        )
        try:
            result = discriminator.analyze_motifs(
                feature_conf=config, exclude_patients=[], n_jobs=1,
                random_state=0, max_failed_folds=5,
            )
        except RuntimeError as exc:
            print(f"Stopping at stringency {threshold}: {exc}", flush=True)
            break
        if len(result.results) != expected_patients:
            raise ValueError(f"Stringency {threshold}: expected {expected_patients} folds")
        if save_fold_predictions:
            fold_table = pd.DataFrame({
                "patient": result.results.index.astype(str),
                "true_class": result.results["class"].to_numpy(),
                "pred_class": result.results.pred_class.to_numpy(),
                "probability_NN": result._positive_probability_vector(),
                "selected_features": result.results.cFeatures.to_numpy(),
                "failed": result.results.fold_failed.to_numpy(),
            })
            fold_table.to_csv(output_dir / f"smore_cism_pipeline_folds_{threshold:.2f}.csv", index=False)
        rows.append({
            "stringency": threshold,
            "auc_failed_equal": result.get_roc_auc_score_failed_equal(),
            "auc_failed_false": result.get_roc_auc_score_failed_false(),
            "failed_folds": result.get_failed_fold_count(),
            "mean_selected_features": float(result.results.cFeatures.mean()),
            "n_patients": expected_patients,
        })
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def plot_results(smore: pd.DataFrame, source: str, output_dir: Path) -> None:
    cism = pd.DataFrame({"stringency": THRESHOLDS, "auc": CISM_AUC})
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(cism.stringency, cism.auc, color="#733ca5", marker="o", lw=2.4,
            label="CISM 4-node benchmark (38 patients)")
    label = ("SMORE seeds, CISM classifier (p=0.01 placeholder)" if source == "provisional"
             else "Native SMORE consensus, CISM classifier")
    ax.plot(smore.stringency, smore.auc_failed_equal, color="#53679b", marker="s",
            lw=2.2, label=label)
    ax.set(xlim=(0.195, 0.505), ylim=(0, 1),
           xlabel="Discrimination stringency", ylabel="NN-NP AUC")
    ax.set_xticks([round(0.20 + 0.05 * i, 2) for i in range(7)])
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    stem = f"cism_benchmark_smore_{source}_cism_pipeline_auc"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("provisional", "native"), default="provisional")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labeled = load_smore(args.source)
    result = run_cism_pipeline(labeled, args.output_dir)
    if result.empty:
        raise RuntimeError("CISM did not finish any stringency; no plot was produced")
    result["source"] = args.source
    result["p_value_policy"] = ("forced_0.01" if args.source == "provisional"
                                 else "native_smore_p_value")
    result.to_csv(args.output_dir / f"smore_{args.source}_cism_pipeline_auc.csv", index=False)
    plot_results(result, args.source, args.output_dir)


if __name__ == "__main__":
    main()
