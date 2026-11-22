from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from cism import CISM, TissueStateDiscriminativeMotifs, prepare_from_centroids, validate_network_dataset_directory
from cism.data_preparation.common import parse_network_filename

from .config import (
    DATASET_NAME,
    DATASET_TYPE,
    DEFAULT_RAW_CENTROIDS_PATH,
    EXISTING_ARTIFACT,
    EXTERNAL_WITHOUT_PP_DATASET_FOLDER,
    EXTERNAL_WITHOUT_PP_ROOT,
    FANMOD_BINARY_DIR,
    FANMOD_ITERATIONS,
    MAX_DISTANCE,
    MOTIF_SIZE,
    ORIGINAL_DATASET_DIR,
    PATIENT_CLASS_CSV,
    P_VALUE_THRESHOLD,
    REMOVED_CLUSTER_ALPHA,
    REMOVED_CLUSTER_BUFFER,
    RUNTIME_DIR,
    TUMOR_CELL_TYPE,
    VariantConfig,
    VARIANTS,
    fanmod_executable_name,
)


CENTROID_COLUMN_MAPPING = {
    "patient_id": "patient number",
    "fov": "fov",
    "cell_type": "pred_mapped",
    "centroid-0": "centroid-0",
    "centroid-1": "centroid-1",
}

TISSUE_STATE_TO_STRING = {
    "NP": "NP",
    "NN": "NN",
    "PN": "PN",
    "PP": "PP",
}

COMMON_CELLS_TYPE = {
    0: "DC",
    1: "b cell",
    2: "blood vessels",
    3: "cd4 apc",
    4: "cd4 t cell",
    5: "cd8 t cell",
    6: "gc",
    7: "hevs",
    8: "mac",
    9: "memory%cd4 t cell",
    10: "neutrophil",
    11: "nk cell",
    12: "stroma",
    13: "treg",
    14: "tumor",
    15: "unidentified",
}


def variant_runtime_dir(variant: VariantConfig) -> Path:
    return RUNTIME_DIR / variant.name


def artifact_path_for_variant(variant: VariantConfig) -> Path:
    return variant_runtime_dir(variant) / "artifact" / f"{variant.name}_cism_ready.pkl"


def load_patient_classes(labels: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
    patient_classes = pd.read_csv(PATIENT_CLASS_CSV, index_col=0, names=["patient_class"])
    if labels is not None:
        patient_classes = patient_classes[patient_classes["patient_class"].isin(labels)].copy()
    return patient_classes


def load_or_build_artifact(
    variant: VariantConfig,
    *,
    raw_centroids_path: Path | None = None,
    force_artifact: bool = False,
    force_preprocessing: bool = False,
    n_jobs: int = 1,
) -> dict[str, Any]:
    if variant.reuse_existing_artifact and not force_artifact:
        if not EXISTING_ARTIFACT.exists():
            raise FileNotFoundError(f"Existing melanoma artifact not found: {EXISTING_ARTIFACT}")
        with EXISTING_ARTIFACT.open("rb") as handle:
            artifact = pickle.load(handle)
        artifact["variant_name"] = variant.name
        artifact["labels"] = list(variant.labels)
        artifact["tumor_removed_classes"] = list(variant.tumor_removed_classes)
        return artifact

    direct_artifact_path = artifact_path_for_variant(variant)
    if direct_artifact_path.exists() and not force_artifact:
        with direct_artifact_path.open("rb") as handle:
            artifact = pickle.load(handle)
        artifact["variant_name"] = variant.name
        artifact["labels"] = list(variant.labels)
        artifact["tumor_removed_classes"] = list(variant.tumor_removed_classes)
        return artifact

    external_source = external_precomputed_source_for_variant(variant)
    if external_source is not None:
        artifact = build_cism_artifact(
            variant,
            dataset_root=external_source["dataset_root"],
            dataset_folder=external_source["dataset_folder"],
            force_artifact=False,
            n_jobs=n_jobs,
            fanmod_output_root=external_source["fanmod_output_root"],
            fanmod_cache_root=external_source["fanmod_cache_root"],
            tissue_state_csv_path=external_source["tissue_state_csv_path"],
            require_patient_class=False,
            reuse_precomputed=True,
        )
        direct_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with direct_artifact_path.open("wb") as handle:
            pickle.dump(artifact, handle)
        return artifact

    artifact_variant = artifact_source_variant(variant)
    artifact_path = artifact_path_for_variant(artifact_variant)
    if artifact_path.exists() and not force_artifact:
        with artifact_path.open("rb") as handle:
            artifact = pickle.load(handle)
        artifact["variant_name"] = variant.name
        artifact["labels"] = list(variant.labels)
        artifact["tumor_removed_classes"] = list(variant.tumor_removed_classes)
        if artifact_variant.name != variant.name:
            print(
                f"Reusing {artifact_variant.name} CISM artifact for {variant.name}; "
                "downstream evaluation filters labels."
            )
        return artifact

    if artifact_variant.requires_tumor_removal:
        dataset_root, dataset_folder = build_tumor_removal_dataset(
            artifact_variant,
            raw_centroids_path=raw_centroids_path or DEFAULT_RAW_CENTROIDS_PATH,
            force_preprocessing=force_preprocessing,
        )
    else:
        dataset_root = ORIGINAL_DATASET_DIR.parent
        dataset_folder = ORIGINAL_DATASET_DIR.name

    artifact = build_cism_artifact(
        artifact_variant,
        dataset_root=dataset_root,
        dataset_folder=dataset_folder,
        force_artifact=force_artifact,
        n_jobs=n_jobs,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    artifact["variant_name"] = variant.name
    artifact["labels"] = list(variant.labels)
    artifact["tumor_removed_classes"] = list(variant.tumor_removed_classes)
    return artifact


def artifact_source_variant(variant: VariantConfig) -> VariantConfig:
    """Return the CISM artifact variant that can serve this evaluation variant.

    The all-classes tumor-removed artifact is a superset for the without-PP
    evaluation: PN tumors are removed, NN/NP are unchanged, and PP patients are
    filtered out downstream. Reusing it avoids a duplicate FANMOD enumeration.
    """
    if variant.name == "without_pp_remove_tumor":
        return VARIANTS["all_classes_remove_tumor"]
    return variant


def external_precomputed_source_for_variant(variant: VariantConfig) -> dict[str, Path | str] | None:
    if variant.external_precomputed_source != "processed_data_23_10_2025":
        return None

    root = EXTERNAL_WITHOUT_PP_ROOT
    dataset_folder = EXTERNAL_WITHOUT_PP_DATASET_FOLDER
    paths = {
        "dataset_root": root / "data",
        "dataset_folder": dataset_folder,
        "fanmod_output_root": root / "fanmod_output",
        "fanmod_cache_root": root / "parse_cache",
        "tissue_state_csv_path": root / "patient_class.csv",
    }
    missing = [
        str(path)
        for key, path in paths.items()
        if key != "dataset_folder" and isinstance(path, Path) and not path.exists()
    ]
    if missing:
        print(
            "External precomputed source is unavailable; falling back to local artifact construction. "
            f"Missing: {missing}"
        )
        return None

    dataset_dir = paths["dataset_root"] / dataset_folder  # type: ignore[operator]
    fanmod_dir = paths["fanmod_output_root"] / dataset_folder / str(MOTIF_SIZE)  # type: ignore[operator]
    cache_dir = paths["fanmod_cache_root"] / dataset_folder / str(MOTIF_SIZE)  # type: ignore[operator]
    if not dataset_dir.exists() or not fanmod_dir.exists() or not cache_dir.exists():
        print(
            "External precomputed source is incomplete; falling back to local artifact construction. "
            f"dataset_dir={dataset_dir.exists()} fanmod_dir={fanmod_dir.exists()} cache_dir={cache_dir.exists()}"
        )
        return None

    return paths


def build_tumor_removal_dataset(
    variant: VariantConfig,
    *,
    raw_centroids_path: Path,
    force_preprocessing: bool = False,
) -> tuple[Path, str]:
    variant_dir = variant_runtime_dir(variant)
    dataset_root = variant_dir / "prepared_datasets"
    combined_dir = dataset_root / variant.name
    cleaned_dir = variant_dir / "prepared_cleaned_all_patients"
    combined_dir.mkdir(parents=True, exist_ok=True)

    if not force_preprocessing and _prepared_dataset_is_complete(combined_dir):
        print(f"Reusing prepared tumor-removal dataset: {combined_dir}")
        return dataset_root, variant.name

    raw_centroids_path = Path(raw_centroids_path)
    if not raw_centroids_path.exists():
        raise FileNotFoundError(
            "Tumor-removal variants require the raw melanoma centroid table so CISM can use "
            f"the built-in alpha-shape remover. Provide --raw-centroids-path. Missing: {raw_centroids_path}"
        )

    patient_classes = load_patient_classes()
    remove_patient_ids = set(
        patient_classes[patient_classes["patient_class"].isin(variant.tumor_removed_classes)].index.astype(str)
    )

    raw_centroids = pd.read_csv(raw_centroids_path, index_col=0)
    prepare_from_centroids(
        dataframe=raw_centroids,
        path_to_output_dir=str(cleaned_dir),
        column_mapping=CENTROID_COLUMN_MAPPING,
        max_distance=MAX_DISTANCE,
        exclude_cell_type=TUMOR_CELL_TYPE,
        removed_cluster_alpha=REMOVED_CLUSTER_ALPHA,
        removed_cluster_buffer=REMOVED_CLUSTER_BUFFER,
    )

    original_graphs = sorted(ORIGINAL_DATASET_DIR.glob("Patient_*_FOV*.txt"))
    for original_graph in tqdm(original_graphs, desc=f"{variant.name}: mix cleaned/original graphs", unit="graph"):
        patient_id = _patient_uid_from_graph_name(original_graph.name)
        source = cleaned_dir / original_graph.name if patient_id in remove_patient_ids else original_graph
        if not source.exists():
            raise FileNotFoundError(f"Expected prepared graph missing for {patient_id}: {source}")
        shutil.copy2(source, combined_dir / original_graph.name)

    shutil.copy2(PATIENT_CLASS_CSV, combined_dir / "patient_class.csv")
    metadata = {
        "variant": variant.name,
        "tumor_removed_classes": list(variant.tumor_removed_classes),
        "tumor_removed_patient_count": len(remove_patient_ids),
        "raw_centroids_path": str(raw_centroids_path),
        "alpha": REMOVED_CLUSTER_ALPHA,
        "buffer": REMOVED_CLUSTER_BUFFER,
        "max_distance": MAX_DISTANCE,
    }
    (variant_dir / "artifact").mkdir(parents=True, exist_ok=True)
    (variant_dir / "artifact" / "preprocessing_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_network_dataset_directory(combined_dir, require_patient_class=True)
    return dataset_root, variant.name


def build_cism_artifact(
    variant: VariantConfig,
    *,
    dataset_root: Path,
    dataset_folder: str,
    force_artifact: bool,
    n_jobs: int,
    fanmod_output_root: Path | None = None,
    fanmod_cache_root: Path | None = None,
    tissue_state_csv_path: Path | None = None,
    require_patient_class: bool = True,
    reuse_precomputed: bool = False,
) -> dict[str, Any]:
    if reuse_precomputed:
        return build_cism_artifact_from_parse_cache(
            variant,
            dataset_root=dataset_root,
            dataset_folder=dataset_folder,
            fanmod_output_root=fanmod_output_root,
            fanmod_cache_root=fanmod_cache_root,
            tissue_state_csv_path=tissue_state_csv_path,
        )

    variant_dir = variant_runtime_dir(variant)
    fanmod_output_root = fanmod_output_root or (variant_dir / "fanmod_output")
    fanmod_cache_root = fanmod_cache_root or (variant_dir / "fanmod_cache")
    fanmod_output_root.mkdir(parents=True, exist_ok=True)
    fanmod_cache_root.mkdir(parents=True, exist_ok=True)

    cism = CISM(
        fanmod_exe=fanmod_executable_name(),
        fanmod_path=str(FANMOD_BINARY_DIR),
        network_dataset_root_path=str(dataset_root),
        fanmod_output_root_path=str(fanmod_output_root),
        fanmod_cache_root_path=str(fanmod_cache_root),
        motif_size=MOTIF_SIZE,
        iterations=FANMOD_ITERATIONS,
    )
    cism.add_dataset(
        dataset_folder=dataset_folder,
        dataset_type=DATASET_TYPE,
        dataset_name=DATASET_NAME,
        force_run_fanmod=False if reuse_precomputed else force_artifact,
        force_parse=False if reuse_precomputed else force_artifact,
        n_jobs=n_jobs,
        require_patient_class=require_patient_class,
    )
    if cism.motifs_dataset is None or cism.motifs_dataset.empty:
        raise RuntimeError(f"CISM produced no motif rows for variant {variant.name}.")

    tissue_state_csv_path = str(tissue_state_csv_path or (Path(dataset_root) / dataset_folder / "patient_class.csv"))
    discriminator = TissueStateDiscriminativeMotifs(
        cism=cism,
        tissue_state_csv_path=tissue_state_csv_path,
        tissue_state_to_string=TISSUE_STATE_TO_STRING,
        common_cells_type=COMMON_CELLS_TYPE,
    )
    return {
        "cism": cism,
        "discriminator": discriminator,
        "network_dataset_root_path": str(dataset_root),
        "dataset_folder": dataset_folder,
        "dataset_name": DATASET_NAME,
        "fanmod_path": str(FANMOD_BINARY_DIR),
        "fanmod_exe": fanmod_executable_name(),
        "fanmod_output_root_path": str(fanmod_output_root),
        "fanmod_cache_root_path": str(fanmod_cache_root),
        "tissue_state_csv_path": tissue_state_csv_path,
        "reuse_precomputed": reuse_precomputed,
        "tissue_state_to_string": TISSUE_STATE_TO_STRING,
        "common_cells_type": COMMON_CELLS_TYPE,
        "labels": list(variant.labels),
        "variant_name": variant.name,
        "tumor_removed_classes": list(variant.tumor_removed_classes),
    }


def build_cism_artifact_from_parse_cache(
    variant: VariantConfig,
    *,
    dataset_root: Path,
    dataset_folder: str,
    fanmod_output_root: Path | None,
    fanmod_cache_root: Path | None,
    tissue_state_csv_path: Path | None,
) -> dict[str, Any]:
    if fanmod_output_root is None or fanmod_cache_root is None or tissue_state_csv_path is None:
        raise ValueError("Precomputed CISM artifact construction requires output, cache, and tissue-state paths.")

    dataset_dir = Path(dataset_root) / dataset_folder
    files = validate_network_dataset_directory(dataset_dir, require_patient_class=False)
    patient_classes = pd.read_csv(tissue_state_csv_path, index_col=0, names=["patient_class"])
    patient_classes = patient_classes[patient_classes["patient_class"].isin(variant.labels)].copy()
    included_patient_numbers = {
        str(patient_uid).replace(DATASET_NAME, "", 1) for patient_uid in patient_classes.index.astype(str)
    }

    cache_dir = Path(fanmod_cache_root) / dataset_folder / str(MOTIF_SIZE)
    selected_cache_files: list[Path] = []
    for file in files:
        patient_num, fov = parse_network_filename(file)
        if patient_num not in included_patient_numbers:
            continue
        cache_file = cache_dir / f"{patient_num}_FOV{fov}.df_cache"
        if not cache_file.exists():
            raise FileNotFoundError(f"Expected precomputed parse cache is missing: {cache_file}")
        selected_cache_files.append(cache_file)

    if not selected_cache_files:
        raise RuntimeError(f"No parse caches matched labels {variant.labels} in {cache_dir}.")

    parse_results = []
    for cache_file in tqdm(
        selected_cache_files,
        desc=f"{variant.name}: loading parse caches",
        unit="cache",
    ):
        parse_results.append(_load_cism_parse_cache(cache_file))

    non_empty_results = [result for result in parse_results if result is not None and not result.empty]
    if not non_empty_results:
        raise RuntimeError(f"Precomputed parse caches produced no motif rows for variant {variant.name}.")

    motifs_dataset = pd.concat(non_empty_results, ignore_index=True)
    motifs_dataset["FOV"] = motifs_dataset["FOV"].astype("category")
    motifs_dataset[DATASET_TYPE] = DATASET_NAME
    motifs_dataset["Patient_uId"] = motifs_dataset[DATASET_TYPE] + motifs_dataset["Patient"].astype("str")
    motifs_dataset[DATASET_TYPE] = motifs_dataset[DATASET_TYPE].astype("category")
    motifs_dataset["nunique_colors"] = motifs_dataset["nunique_colors"].astype("int8")
    motifs_dataset["Patient_uId"] = motifs_dataset["Patient_uId"].astype("category")
    motifs_dataset["FOV"] = motifs_dataset["FOV"].astype("category")
    motifs_dataset.reset_index(drop=True, inplace=True)

    cism = CISM(
        fanmod_exe=fanmod_executable_name(),
        fanmod_path=str(FANMOD_BINARY_DIR),
        network_dataset_root_path=str(dataset_root),
        fanmod_output_root_path=str(fanmod_output_root),
        fanmod_cache_root_path=str(fanmod_cache_root),
        motif_size=MOTIF_SIZE,
        iterations=FANMOD_ITERATIONS,
    )
    cism.motifs_dataset = motifs_dataset
    discriminator = TissueStateDiscriminativeMotifs(
        cism=cism,
        tissue_state_csv_path=str(tissue_state_csv_path),
        tissue_state_to_string=TISSUE_STATE_TO_STRING,
        common_cells_type=COMMON_CELLS_TYPE,
    )
    return {
        "cism": cism,
        "discriminator": discriminator,
        "network_dataset_root_path": str(dataset_root),
        "dataset_folder": dataset_folder,
        "dataset_name": DATASET_NAME,
        "fanmod_path": str(FANMOD_BINARY_DIR),
        "fanmod_exe": fanmod_executable_name(),
        "fanmod_output_root_path": str(fanmod_output_root),
        "fanmod_cache_root_path": str(fanmod_cache_root),
        "tissue_state_csv_path": str(tissue_state_csv_path),
        "reuse_precomputed": True,
        "tissue_state_to_string": TISSUE_STATE_TO_STRING,
        "common_cells_type": COMMON_CELLS_TYPE,
        "labels": list(variant.labels),
        "variant_name": variant.name,
        "tumor_removed_classes": list(variant.tumor_removed_classes),
    }


def _load_cism_parse_cache(cache_file: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(cache_file, index_col=[0], engine="pyarrow")
    except Exception:
        df = pd.read_csv(cache_file, index_col=[0])
    if df.empty:
        return df

    df = df[~df["Mean_Freq"].isna()].copy()
    df["ID"] = df["ID"].astype("int64")
    df["Patient"] = df["Patient"].astype("int16")
    df["Freq"] = pd.to_numeric(df["Freq"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["z_score"] = pd.to_numeric(df["z_score"], errors="coerce")
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
    df["FOV"] = df["FOV"].astype("category")
    df["Mean_Freq"] = df["Mean_Freq"].astype("float32")
    df["STD"] = df["STD"].astype("float32")
    df["z_score"] = df["z_score"].astype("float32")
    if "hash" in df.columns:
        df = df.drop("hash", axis=1)
    df = df[df["p_value"] < P_VALUE_THRESHOLD].copy()
    df["p_value"] = df["p_value"].astype("float32")
    quantile_slice = df["Freq"].quantile(q=1)
    df = df[df["Freq"] <= quantile_slice].copy()
    return df


def _patient_uid_from_graph_name(filename: str) -> str:
    patient_part = filename.split("_FOV", 1)[0].replace("Patient_", "")
    return f"{DATASET_NAME}{patient_part}"


def _prepared_dataset_is_complete(dataset_dir: Path) -> bool:
    if not dataset_dir.exists():
        return False
    if not (dataset_dir / "patient_class.csv").exists():
        return False
    try:
        prepared_graphs = validate_network_dataset_directory(dataset_dir, require_patient_class=True)
    except Exception:
        return False
    original_graph_count = len(list(ORIGINAL_DATASET_DIR.glob("Patient_*_FOV*.txt")))
    return len(prepared_graphs) == original_graph_count
