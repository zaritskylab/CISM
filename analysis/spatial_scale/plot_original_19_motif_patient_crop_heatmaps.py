#!/usr/bin/env python3
"""Plot patient-level crop enrichment heatmaps for the original 19 motifs."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import gc
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = _REPO_ROOT
CROP_ROOT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "crop_fov_scale_experiment"
)
REPORT_ROOT = CROP_ROOT / "melanoma_cropped_fov_report"
ORIGINAL_ARTIFACT = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
REFERENCE_CSV = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "split_fov_scale_experiment"
    / "fold_union_stringency_046_stability"
    / "native_fold_union_selected_motifs.csv"
)
CROP_FRACTIONS = [0.9, 0.8, 0.7, 0.6, 0.5]
TISSUE_COLUMNS = [1.0, *CROP_FRACTIONS]


def token(fraction: float) -> str:
    return f"{int(round(fraction * 100)):03d}"


def crop_artifact_path(fraction: float) -> Path:
    return CROP_ROOT / f"crop_{token(fraction)}_cism_ready.pkl"


def metadata_path(fraction: float) -> Path:
    return CROP_ROOT / f"crop_{token(fraction)}_metadata.csv"


def load_artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_source_fovs() -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path(0.9), usecols=["patient_uid", "source_fov", "patient_class"])
    return (
        metadata.rename(columns={"patient_uid": "Patient_uId"})
        .drop_duplicates(["Patient_uId", "source_fov"])
        .reset_index(drop=True)
    )


def significant_rows(motifs: pd.DataFrame) -> pd.DataFrame:
    rows = motifs[motifs["p_value"] < 0.05]
    if "p_value_direction" in rows.columns:
        rows = rows[rows["p_value_direction"].eq("proliferation")]
    return rows


def source_fov_from_crop(fov: str) -> str:
    return str(fov).split("_crop", 1)[0]


def original_percentages(reference: pd.DataFrame, source_fovs: pd.DataFrame) -> pd.DataFrame:
    print("Loading original 100% artifact", flush=True)
    artifact = load_artifact(ORIGINAL_ARTIFACT)
    motifs = significant_rows(artifact["cism"].motifs_dataset)
    motifs = motifs[motifs["ID"].isin(reference["ID"])][["Patient_uId", "FOV", "ID"]].drop_duplicates()
    motifs = motifs.rename(columns={"FOV": "source_fov"})
    motifs = motifs.merge(source_fovs[["Patient_uId", "source_fov"]], on=["Patient_uId", "source_fov"], how="inner")

    denom = source_fovs.groupby("Patient_uId", observed=True)["source_fov"].nunique()
    counts = (
        motifs.groupby(["ID", "Patient_uId"], observed=True)["source_fov"]
        .nunique()
        .reset_index(name="present_count")
    )
    rows = []
    for motif_id in reference["ID"]:
        for patient_uid, n_source in denom.items():
            present = counts.loc[
                counts["ID"].eq(motif_id) & counts["Patient_uId"].eq(patient_uid),
                "present_count",
            ]
            rows.append(
                {
                    "motif_id": int(motif_id),
                    "Patient_uId": patient_uid,
                    "tissue_retention": 1.0,
                    "percent_enriched": float(present.iloc[0] / n_source * 100) if len(present) else 0.0,
                    "present_crops": int(present.iloc[0]) if len(present) else 0,
                    "total_crops": int(n_source),
                }
            )
    del artifact, motifs
    gc.collect()
    return pd.DataFrame(rows)


def crop_percentages(reference: pd.DataFrame, source_fovs: pd.DataFrame, fraction: float) -> pd.DataFrame:
    print(f"Loading {fraction:.0%} crop artifact", flush=True)
    artifact = load_artifact(crop_artifact_path(fraction))
    motifs = significant_rows(artifact["cism"].motifs_dataset)
    motifs = motifs[motifs["ID"].isin(reference["ID"])][["Patient_uId", "FOV", "ID"]].drop_duplicates()

    metadata = pd.read_csv(metadata_path(fraction), usecols=["patient_uid", "crop_fov"])
    metadata = metadata.rename(columns={"patient_uid": "Patient_uId", "crop_fov": "FOV"})
    denom = metadata.groupby("Patient_uId", observed=True)["FOV"].nunique()
    counts = (
        motifs.merge(metadata, on=["Patient_uId", "FOV"], how="inner")
        .groupby(["ID", "Patient_uId"], observed=True)["FOV"]
        .nunique()
        .reset_index(name="present_count")
    )

    rows = []
    for motif_id in reference["ID"]:
        for patient_uid, n_crops in denom.items():
            present = counts.loc[
                counts["ID"].eq(motif_id) & counts["Patient_uId"].eq(patient_uid),
                "present_count",
            ]
            rows.append(
                {
                    "motif_id": int(motif_id),
                    "Patient_uId": patient_uid,
                    "tissue_retention": fraction,
                    "percent_enriched": float(present.iloc[0] / n_crops * 100) if len(present) else 0.0,
                    "present_crops": int(present.iloc[0]) if len(present) else 0,
                    "total_crops": int(n_crops),
                }
            )
    del artifact, motifs, metadata
    gc.collect()
    return pd.DataFrame(rows)


def patient_order(source_fovs: pd.DataFrame) -> list[str]:
    patients = (
        source_fovs[["Patient_uId", "patient_class"]]
        .drop_duplicates()
        .assign(patient_number=lambda df: df["Patient_uId"].str.extract(r"(\d+)").astype(int))
        .sort_values(["patient_class", "patient_number"])
    )
    return patients["Patient_uId"].tolist()


def plot_heatmaps(data: pd.DataFrame, reference: pd.DataFrame, source_fovs: pd.DataFrame) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ordered_patients = patient_order(source_fovs)
    patient_labels = [p.replace("Melanoma", "M") for p in ordered_patients]
    class_lookup = source_fovs.drop_duplicates("Patient_uId").set_index("Patient_uId")["patient_class"].to_dict()
    patient_labels = [f"{label} ({class_lookup[p]})" for label, p in zip(patient_labels, ordered_patients)]
    col_labels = [f"{int(round(c * 100))}%" for c in TISSUE_COLUMNS]

    reference = reference.copy()
    reference["label_order"] = reference["label"].map({"NN": 0, "NP": 1})
    reference = reference.sort_values(["label_order", "ID"])

    pdf_path = REPORT_ROOT / "original_19_motif_patient_crop_heatmaps.pdf"
    page_paths = []
    with PdfPages(pdf_path) as pdf:
        n_pages = math.ceil(len(reference) / 4)
        for page in range(n_pages):
            page_ref = reference.iloc[page * 4 : (page + 1) * 4]
            fig, axes = plt.subplots(1, len(page_ref), figsize=(4.2 * len(page_ref), 9.5), sharey=True)
            if len(page_ref) == 1:
                axes = [axes]
            for ax, motif in zip(axes, page_ref.itertuples(index=False)):
                matrix = (
                    data[data["motif_id"].eq(int(motif.ID))]
                    .pivot(index="Patient_uId", columns="tissue_retention", values="percent_enriched")
                    .reindex(index=ordered_patients, columns=TISSUE_COLUMNS)
                    .fillna(0)
                )
                image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="viridis")
                ax.set_title(f"{motif.label} motif\n{int(motif.ID)}", fontsize=10)
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, rotation=45, ha="right")
                ax.set_yticks(range(len(patient_labels)))
                ax.set_yticklabels(patient_labels, fontsize=6)
                ax.tick_params(axis="x", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            fig.subplots_adjust(right=0.92, wspace=0.18)
            cax = fig.add_axes([0.94, 0.18, 0.012, 0.64])
            cb = fig.colorbar(image, cax=cax)
            cb.set_label("% enriched FOV/crops", fontsize=9)
            fig.suptitle(
                f"Original Discriminative Motifs: Patient-Level Crop Enrichment ({page + 1}/{n_pages})",
                fontsize=13,
                fontweight="bold",
            )
            page_path = REPORT_ROOT / f"original_19_motif_patient_crop_heatmaps_page_{page + 1}.png"
            fig.savefig(page_path, dpi=220, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            page_paths.append(page_path)
    print(pdf_path.resolve())
    for path in page_paths:
        print(path.resolve())


def main() -> None:
    reference = pd.read_csv(REFERENCE_CSV)
    reference["ID"] = reference["ID"].astype(int)
    source_fovs = load_source_fovs()
    frames = [original_percentages(reference, source_fovs)]
    for fraction in CROP_FRACTIONS:
        frames.append(crop_percentages(reference, source_fovs, fraction))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(REPORT_ROOT / "original_19_motif_patient_crop_enrichment_percentages.csv", index=False)
    plot_heatmaps(data, reference, source_fovs)


if __name__ == "__main__":
    main()
