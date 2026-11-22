#!/usr/bin/env python3
"""Final FOV-scale tests for melanoma NN/NP enriched motif analysis."""

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

RUNTIME = REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime"
FOV_COUNTS_CSV = RUNTIME / "fov_scale_motif_counts_nn_np" / "fov_motif_counts_per_fov.csv"
ORIGINAL_ARTIFACT = RUNTIME / "serialized" / "Melanoma_cism_ready.pkl"
ASYM_DIR = RUNTIME / "asymmetric_stringency"
SPLIT_DIR = RUNTIME / "split_fov_scale_experiment"
OUTPUT_DIR = SPLIT_DIR / "final_fov_scale_tests"

PARTS = [2, 4, 8, 16]
LABELS = ["NN", "NP"]
P_VALUE_THRESHOLD = 0.05
RANDOM_REPEATS = 100
RANDOM_STATE = 0


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_enriched_motifs(path: Path) -> pd.DataFrame:
    artifact = load_pickle(path)
    df = artifact["cism"].motifs_dataset.copy()
    if "p_value_direction" in df.columns:
        df = df[df["p_value_direction"].eq("proliferation")].copy()
    df = df[df["p_value"].lt(P_VALUE_THRESHOLD)].copy()
    return df


def load_patient_classes() -> pd.Series:
    artifact = load_pickle(ORIGINAL_ARTIFACT)
    classes = artifact["discriminator"].get_patients_class(LABELS)
    class_col = [col for col in classes.columns if col.lower() in {"class", "patient_class"}][0]
    classes = classes[class_col]
    return classes[classes.isin(LABELS)]


def motif_sets_by_patient(motifs: pd.DataFrame) -> pd.Series:
    return (
        motifs.drop_duplicates(["Patient_uId", "ID"])
        .groupby("Patient_uId", observed=True)["ID"]
        .agg(lambda values: set(map(int, values)))
    )


def summarize_fov_scale_heatmaps() -> tuple[pd.DataFrame, pd.DataFrame]:
    fovs = pd.read_csv(FOV_COUNTS_CSV)
    patient = (
        fovs.groupby(["patient_uid", "patient_class"], observed=True)
        .agg(
            small_fovs=("scale_label", lambda s: int((s == "small").sum())),
            large_fovs=("scale_label", lambda s: int((s == "full").sum())),
            patient_enriched_motifs=("significant_motif_count", "sum"),
        )
        .reset_index()
    )

    original_motifs = load_enriched_motifs(ORIGINAL_ARTIFACT)
    patient_sets = motif_sets_by_patient(original_motifs)
    patient["patient_enriched_motif_union"] = patient["patient_uid"].map(
        lambda patient_uid: len(patient_sets.get(patient_uid, set()))
    )
    patient.to_csv(OUTPUT_DIR / "patient_fov_scale_motif_summary.csv", index=False)

    counts = patient.pivot_table(
        index="large_fovs",
        columns="small_fovs",
        values="patient_uid",
        aggfunc="count",
        fill_value=0,
    ).sort_index(ascending=True)
    mean_motifs = patient.pivot_table(
        index="large_fovs",
        columns="small_fovs",
        values="patient_enriched_motif_union",
        aggfunc="mean",
    ).sort_index(ascending=True)

    counts.to_csv(OUTPUT_DIR / "heatmap_patient_count_by_small_large_fovs.csv")
    mean_motifs.to_csv(OUTPUT_DIR / "heatmap_mean_patient_motifs_by_small_large_fovs.csv")
    plot_heatmap(counts, "Patients by FOV scale counts", "Number of patients", OUTPUT_DIR / "heatmap_patient_count_by_small_large_fovs.png", integer=True)
    plot_heatmap(mean_motifs, "Mean patient enriched motifs by FOV scale counts", "Mean motif IDs", OUTPUT_DIR / "heatmap_mean_patient_motifs_by_small_large_fovs.png", integer=False)
    return patient, original_motifs


def plot_heatmap(table: pd.DataFrame, title: str, colorbar_label: str, path: Path, integer: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    masked = np.ma.masked_invalid(table.to_numpy(dtype=float))
    im = ax.imshow(masked, origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(table.shape[1]), labels=table.columns.astype(int))
    ax.set_yticks(np.arange(table.shape[0]), labels=table.index.astype(int))
    ax.set_xlabel("Number of small FOVs")
    ax.set_ylabel("Number of large FOVs")
    ax.set_title(title)
    for y in range(table.shape[0]):
        for x in range(table.shape[1]):
            value = table.iloc[y, x]
            if pd.isna(value):
                continue
            text = f"{int(value)}" if integer else f"{value:,.0f}"
            ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10, weight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize_failed_patients(patient_summary: pd.DataFrame) -> pd.DataFrame:
    loocv = pd.read_csv(ASYM_DIR / "joint_asymmetric_stringency_loocv_results.csv")
    failed = loocv[(loocv["valid_fold"]) & (loocv["class"] != loocv["pred_class"])].copy()
    failed = failed.merge(
        patient_summary[
            [
                "patient_uid",
                "patient_class",
                "small_fovs",
                "large_fovs",
                "patient_enriched_motif_union",
            ]
        ],
        left_on="patient_id",
        right_on="patient_uid",
        how="left",
    )
    failed = failed[
        [
            "patient_id",
            "class",
            "pred_class",
            "positive_probability",
            "cFeatures",
            "small_fovs",
            "large_fovs",
            "patient_enriched_motif_union",
        ]
    ]
    failed.to_csv(OUTPUT_DIR / "best_model_failed_patients_fov_motifs.csv", index=False)
    return failed


def class_prevalence_summary(motifs: pd.DataFrame, patient_classes: pd.Series) -> pd.DataFrame:
    patient_ids = patient_classes.index.tolist()
    local = motifs[motifs["Patient_uId"].isin(patient_ids)].copy()
    present = local.drop_duplicates(["Patient_uId", "ID"])[["Patient_uId", "ID"]]
    present["patient_class"] = present["Patient_uId"].map(patient_classes)

    class_counts = patient_classes.value_counts().to_dict()
    prevalence = (
        present.groupby(["ID", "patient_class"], observed=True)["Patient_uId"]
        .nunique()
        .unstack(fill_value=0)
    )
    for label in LABELS:
        if label not in prevalence.columns:
            prevalence[label] = 0
    prevalence = prevalence[LABELS].reset_index()
    for label in LABELS:
        prevalence[f"{label}_prevalence"] = prevalence[label] / class_counts[label]

    freq = local.copy()
    freq["patient_class"] = freq["Patient_uId"].map(patient_classes)
    freq_summary = (
        freq.groupby(["ID", "patient_class"], observed=True)["Freq"].mean().unstack(fill_value=0)
    )
    for label in LABELS:
        if label not in freq_summary.columns:
            freq_summary[label] = 0.0
    freq_summary = freq_summary[LABELS].reset_index()
    freq_summary = freq_summary.rename(columns={label: f"{label}_mean_frequency" for label in LABELS})

    out = prevalence.merge(freq_summary, on="ID", how="left")
    out["dominant_group"] = np.where(out["NN_prevalence"] >= out["NP_prevalence"], "NN", "NP")
    out["minority_group"] = np.where(out["dominant_group"].eq("NN"), "NP", "NN")
    out["dominant_prevalence"] = out.apply(lambda r: r[f"{r['dominant_group']}_prevalence"], axis=1)
    out["minority_prevalence"] = out.apply(lambda r: r[f"{r['minority_group']}_prevalence"], axis=1)
    out["prevalence_gap"] = out["dominant_prevalence"] - out["minority_prevalence"]
    return out.sort_values(["prevalence_gap", "dominant_prevalence"], ascending=False)


def prevalence_summary_from_patient_sets(patient_sets: dict[str, set[int]], patient_classes: pd.Series) -> pd.DataFrame:
    class_counts = patient_classes.value_counts().to_dict()
    motif_to_patients: dict[int, list[str]] = {}
    for patient_uid, motif_ids in patient_sets.items():
        if patient_uid not in patient_classes.index:
            continue
        for motif_id in motif_ids:
            motif_to_patients.setdefault(int(motif_id), []).append(patient_uid)

    rows = []
    for motif_id, patients in motif_to_patients.items():
        classes = patient_classes.loc[patients]
        nn_count = int((classes == "NN").sum())
        np_count = int((classes == "NP").sum())
        nn_prev = nn_count / class_counts["NN"]
        np_prev = np_count / class_counts["NP"]
        dominant = "NN" if nn_prev >= np_prev else "NP"
        minority = "NP" if dominant == "NN" else "NN"
        dominant_prev = nn_prev if dominant == "NN" else np_prev
        minority_prev = np_prev if dominant == "NN" else nn_prev
        rows.append(
            {
                "ID": motif_id,
                "NN": nn_count,
                "NP": np_count,
                "NN_prevalence": nn_prev,
                "NP_prevalence": np_prev,
                "NN_mean_frequency": 0.0,
                "NP_mean_frequency": 0.0,
                "dominant_group": dominant,
                "minority_group": minority,
                "dominant_prevalence": dominant_prev,
                "minority_prevalence": minority_prev,
                "prevalence_gap": dominant_prev - minority_prev,
            }
        )
    return pd.DataFrame(rows).sort_values(["prevalence_gap", "dominant_prevalence"], ascending=False)


def prevalence_summary_from_presence(presence: pd.DataFrame, patient_classes: pd.Series) -> pd.DataFrame:
    if presence.empty:
        return pd.DataFrame(columns=["ID", "NN_prevalence", "NP_prevalence", "dominant_group", "minority_group"])
    local = presence[presence["Patient_uId"].isin(patient_classes.index)].copy()
    local["patient_class"] = local["Patient_uId"].map(patient_classes)
    class_counts = patient_classes.value_counts()
    counts = pd.crosstab(local["ID"].astype(int), local["patient_class"])
    for label in LABELS:
        if label not in counts.columns:
            counts[label] = 0
    counts = counts[LABELS].reset_index()
    for label in LABELS:
        counts[f"{label}_prevalence"] = counts[label] / class_counts[label]
        counts[f"{label}_mean_frequency"] = 0.0
    counts["dominant_group"] = np.where(counts["NN_prevalence"] >= counts["NP_prevalence"], "NN", "NP")
    counts["minority_group"] = np.where(counts["dominant_group"].eq("NN"), "NP", "NN")
    counts["dominant_prevalence"] = counts.apply(lambda r: r[f"{r['dominant_group']}_prevalence"], axis=1)
    counts["minority_prevalence"] = counts.apply(lambda r: r[f"{r['minority_group']}_prevalence"], axis=1)
    counts["prevalence_gap"] = counts["dominant_prevalence"] - counts["minority_prevalence"]
    return counts.sort_values(["prevalence_gap", "dominant_prevalence"], ascending=False)


def patient_sets_from_motifs(motifs: pd.DataFrame) -> dict[str, set[int]]:
    grouped = motif_sets_by_patient(motifs)
    return {str(patient_uid): set(map(int, motif_ids)) for patient_uid, motif_ids in grouped.items()}


def patient_sets_from_tile_sets(chosen_tile_sets: list[tuple[str, set[int]]]) -> dict[str, set[int]]:
    patient_sets: dict[str, set[int]] = {}
    for patient_uid, motif_ids in chosen_tile_sets:
        patient_sets.setdefault(str(patient_uid), set()).update(motif_ids)
    return patient_sets


def selected_by_joint_threshold(summary: pd.DataFrame, majority_min: float, minority_max: float) -> pd.DataFrame:
    return summary[
        (summary["dominant_prevalence"] >= majority_min)
        & (summary["minority_prevalence"] <= minority_max)
    ].copy()


def stability_row(scenario: str, parts: int | str, reference_name: str, reference_ids: set[int], selected: pd.DataFrame) -> dict:
    selected_ids = set(map(int, selected["ID"]))
    shared = selected_ids & reference_ids
    union = selected_ids | reference_ids
    by_class = selected["dominant_group"].value_counts().to_dict() if not selected.empty else {}
    return {
        "scenario": scenario,
        "grid_parts": parts,
        "reference": reference_name,
        "reference_motif_count": len(reference_ids),
        "selected_motif_count": len(selected_ids),
        "shared_with_reference": len(shared),
        "reference_recovered_fraction": len(shared) / len(reference_ids) if reference_ids else np.nan,
        "selected_from_reference_fraction": len(shared) / len(selected_ids) if selected_ids else np.nan,
        "jaccard": len(shared) / len(union) if union else np.nan,
        "selected_NN": int(by_class.get("NN", 0)),
        "selected_NP": int(by_class.get("NP", 0)),
    }


def split_source_fov(fov: str) -> str:
    return str(fov).split("_p", 1)[0]


def run_stability(patient_classes: pd.Series) -> pd.DataFrame:
    metrics = pd.read_csv(ASYM_DIR / "joint_asymmetric_stringency_metrics.csv").set_index("Unnamed: 0")["value"]
    majority_min = float(metrics["majority_min"])
    minority_max = float(metrics["minority_max"])

    original_joint = pd.read_csv(ASYM_DIR / "joint_asymmetric_selected_motifs.csv")
    original_direct = pd.read_csv(ASYM_DIR / "direct_asymmetric_selected_motifs.csv")
    references = {
        "joint_best_2": set(original_joint["ID"].astype(int)),
        "direct_top30": set(original_direct["ID"].astype(int)),
    }

    rows = []
    selected_frames = []
    for parts in PARTS:
        print(f"Loading split artifact {parts}...", flush=True)
        split = load_enriched_motifs(SPLIT_DIR / f"split_{parts:02d}_cism_ready.pkl")
        print(f"Summarizing all split FOVs for parts={parts} ({len(split):,} motif rows)...", flush=True)
        presence = split.drop_duplicates(["Patient_uId", "ID"])[["Patient_uId", "ID"]]
        split_summary = prevalence_summary_from_presence(presence, patient_classes)
        selected = selected_by_joint_threshold(split_summary, majority_min, minority_max)
        selected.insert(0, "scenario", "all_split_fovs_aggregated")
        selected.insert(1, "grid_parts", parts)
        selected_frames.append(selected)
        for reference_name, reference_ids in references.items():
            rows.append(stability_row("all_split_fovs_aggregated", parts, reference_name, reference_ids, selected))
        del split, split_summary, selected
        gc.collect()

    print("Loading split artifact 2 for random one-of-two scenario...", flush=True)
    split2 = load_enriched_motifs(SPLIT_DIR / "split_02_cism_ready.pkl")
    metadata = pd.read_csv(SPLIT_DIR / "split_02_metadata.csv")
    print("Precomputing split-2 tile motif presence table...", flush=True)
    tile_presence = split2.drop_duplicates(["Patient_uId", "FOV", "ID"])[["Patient_uId", "FOV", "ID"]].copy()
    tile_options = {
        key: group["split_fov"].to_numpy()
        for key, group in metadata.groupby(["patient_uid", "source_fov"], observed=True)
    }
    del split2
    gc.collect()
    rng = np.random.default_rng(RANDOM_STATE)
    for repeat in range(RANDOM_REPEATS):
        if repeat % 20 == 0:
            print(f"Random split-2 repeat {repeat}/{RANDOM_REPEATS}", flush=True)
        chosen_fovs = {str(rng.choice(options)) for options in tile_options.values()}
        random_presence = tile_presence[tile_presence["FOV"].isin(chosen_fovs)][["Patient_uId", "ID"]]
        random_presence = random_presence.drop_duplicates(["Patient_uId", "ID"])
        random_summary = prevalence_summary_from_presence(random_presence, patient_classes)
        selected = selected_by_joint_threshold(random_summary, majority_min, minority_max)
        selected.insert(0, "scenario", "random_one_of_split2")
        selected.insert(1, "grid_parts", 2)
        selected.insert(2, "repeat", repeat)
        selected_frames.append(selected)
        for reference_name, reference_ids in references.items():
            row = stability_row("random_one_of_split2", 2, reference_name, reference_ids, selected)
            row["repeat"] = repeat
            rows.append(row)

    stability = pd.DataFrame(rows)
    stability.to_csv(OUTPUT_DIR / "split_discriminative_motif_stability.csv", index=False)
    if selected_frames:
        pd.concat(selected_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "split_selected_discriminative_motifs_by_scenario.csv",
            index=False,
        )
    plot_stability(stability)
    return stability


def plot_stability(stability: pd.DataFrame) -> None:
    aggregate = stability[stability["scenario"].eq("all_split_fovs_aggregated")]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for reference, group in aggregate.groupby("reference", observed=True):
        ax.plot(group["grid_parts"], group["jaccard"], marker="o", label=reference)
    ax.set_xlabel("Grid parts per FOV")
    ax.set_ylabel("Jaccard vs original reference")
    ax.set_ylim(0, 1)
    ax.set_title("Discriminative motif stability: all split FOVs aggregated")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "split_all_aggregated_discriminative_stability.png", dpi=200)
    plt.close(fig)

    random = stability[stability["scenario"].eq("random_one_of_split2")]
    if random.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    data = [random.loc[random["reference"].eq(ref), "jaccard"] for ref in sorted(random["reference"].unique())]
    ax.boxplot(data, labels=sorted(random["reference"].unique()), showfliers=False)
    ax.set_ylabel("Jaccard vs original reference")
    ax.set_ylim(0, 1)
    ax.set_title("Discriminative motif stability: random one tile from split-2")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "split2_random_discriminative_stability.png", dpi=200)
    plt.close(fig)


def write_markdown(patient_summary: pd.DataFrame, failed: pd.DataFrame, stability: pd.DataFrame) -> Path:
    heatmap_counts = pd.read_csv(OUTPUT_DIR / "heatmap_patient_count_by_small_large_fovs.csv", index_col=0)
    heatmap_motifs = pd.read_csv(OUTPUT_DIR / "heatmap_mean_patient_motifs_by_small_large_fovs.csv", index_col=0)
    agg = stability[stability["scenario"].eq("all_split_fovs_aggregated")].copy()
    rand = (
        stability[stability["scenario"].eq("random_one_of_split2")]
        .groupby("reference", observed=True)
        .agg(
            mean_selected_motif_count=("selected_motif_count", "mean"),
            mean_shared_with_reference=("shared_with_reference", "mean"),
            mean_reference_recovered_fraction=("reference_recovered_fraction", "mean"),
            mean_jaccard=("jaccard", "mean"),
            sd_jaccard=("jaccard", "std"),
        )
        .reset_index()
    )

    def simple_markdown_table(df: pd.DataFrame, include_index: bool = False) -> str:
        view = df.reset_index() if include_index else df.copy()
        view = view.fillna("")
        columns = [str(col) for col in view.columns]
        rows = []
        for _, row in view.iterrows():
            rows.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        return "\n".join([header, sep, *rows])

    path = OUTPUT_DIR / "final_fov_scale_tests_report.md"
    lines = [
        "# Final FOV-Scale Tests",
        "",
        "Scope: enriched motifs only.",
        "",
        "## Heatmaps",
        "",
        "Patient count heatmap table:",
        "",
        simple_markdown_table(heatmap_counts, include_index=True),
        "",
        "Mean patient motif-union heatmap table:",
        "",
        simple_markdown_table(heatmap_motifs.round(1), include_index=True),
        "",
        f"![patient count heatmap]({(OUTPUT_DIR / 'heatmap_patient_count_by_small_large_fovs.png').resolve()})",
        "",
        f"![mean motif heatmap]({(OUTPUT_DIR / 'heatmap_mean_patient_motifs_by_small_large_fovs.png').resolve()})",
        "",
        "## Best-Model Failed Patients",
        "",
        "Best saved model used here: `joint_asymmetric_stringency`, accuracy 0.921, balanced accuracy 0.919, ROC AUC 0.889.",
        "",
        simple_markdown_table(failed),
        "",
        "## Split-FOV Discriminative Motif Stability",
        "",
        "The exact best joint model selected only 2 motifs, so `joint_best_2` is a strict small reference. `direct_top30` is included as a broader asymmetric discriminative panel.",
        "",
        "All split FOVs aggregated:",
        "",
        simple_markdown_table(agg),
        "",
        "Random one-of-two split tile per source FOV, summarized over 100 repeats:",
        "",
        simple_markdown_table(rand),
        "",
        f"![all split stability]({(OUTPUT_DIR / 'split_all_aggregated_discriminative_stability.png').resolve()})",
        "",
        f"![random split stability]({(OUTPUT_DIR / 'split2_random_discriminative_stability.png').resolve()})",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patient_summary, _original_motifs = summarize_fov_scale_heatmaps()
    failed = summarize_failed_patients(patient_summary)
    patient_classes = load_patient_classes()
    stability = run_stability(patient_classes)
    report = write_markdown(patient_summary, failed, stability)
    summary = {
        "output_dir": str(OUTPUT_DIR),
        "failed_patient_count": int(failed.shape[0]),
        "random_repeats": RANDOM_REPEATS,
        "report": str(report),
    }
    (OUTPUT_DIR / "final_fov_scale_tests_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
