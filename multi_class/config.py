from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELANOMA_TUTORIAL_DIR = PROJECT_ROOT / "melanoma_tutorials"
NESTED_TUTORIAL_RUNTIME_DIR = MELANOMA_TUTORIAL_DIR / "melanoma_tutorials" / "runtime"
FLAT_TUTORIAL_RUNTIME_DIR = MELANOMA_TUTORIAL_DIR / "runtime"
TUTORIAL_RUNTIME_DIR = (
    NESTED_TUTORIAL_RUNTIME_DIR
    if (NESTED_TUTORIAL_RUNTIME_DIR / "serialized" / "Melanoma_cism_ready.pkl").exists()
    else FLAT_TUTORIAL_RUNTIME_DIR
)
ORIGINAL_DATASET_DIR = MELANOMA_TUTORIAL_DIR / "pet_data" / "Melanoma"
PATIENT_CLASS_CSV = ORIGINAL_DATASET_DIR / "patient_class.csv"
EXISTING_ARTIFACT = TUTORIAL_RUNTIME_DIR / "serialized" / "Melanoma_cism_ready.pkl"
DEFAULT_RAW_CENTROIDS_PATH = Path("/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv")
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
FANMOD_BINARY_DIR = PROJECT_ROOT / "FANMOD_binaries"
EXTERNAL_WITHOUT_PP_ROOT = Path("/groups/assafza_group/assafza/Omri/processed_data_23_10_2025")
EXTERNAL_WITHOUT_PP_DATASET_FOLDER = "melanoma_distance_100_XX"

DATASET_TYPE = "Disease"
DATASET_NAME = "Melanoma"
MOTIF_SIZE = 4
FANMOD_ITERATIONS = 1000
P_VALUE_THRESHOLD = 0.05
MAX_DISTANCE = 100
TUMOR_CELL_TYPE = "tumor"
REMOVED_CLUSTER_ALPHA = 0.01
REMOVED_CLUSTER_BUFFER = 0

ALL_LABELS = ("NN", "NP", "PN", "PP")
WITHOUT_PP_LABELS = ("NN", "NP", "PN")
OTHER_LABEL = "OTHER"
RANDOM_STATE = 0
MAX_CLASS_FEATURES = 30
STRINGENCY_GRID = tuple(round(value, 4) for value in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95))


@dataclass(frozen=True)
class VariantConfig:
    name: str
    labels: tuple[str, ...]
    tumor_removed_classes: tuple[str, ...]
    reuse_existing_artifact: bool = False
    external_precomputed_source: str | None = None

    @property
    def requires_tumor_removal(self) -> bool:
        return bool(self.tumor_removed_classes)


VARIANTS = {
    "all_classes_keep_tumor": VariantConfig(
        name="all_classes_keep_tumor",
        labels=ALL_LABELS,
        tumor_removed_classes=(),
        reuse_existing_artifact=True,
    ),
    "all_classes_remove_tumor": VariantConfig(
        name="all_classes_remove_tumor",
        labels=ALL_LABELS,
        tumor_removed_classes=("PN", "PP"),
    ),
    "without_pp_remove_tumor": VariantConfig(
        name="without_pp_remove_tumor",
        labels=WITHOUT_PP_LABELS,
        tumor_removed_classes=("PN",),
        external_precomputed_source="processed_data_23_10_2025",
    ),
}

MOTIF_PANEL_METHODS = ("one_vs_all_union", "one_vs_one_union", "one_vs_all_positive_union")
CONFIDENCE_STACK_METHODS = (
    "one_vs_all_confidence_stack",
    "one_vs_one_third_class_confidence_stack",
)
METHODS = (*MOTIF_PANEL_METHODS, *CONFIDENCE_STACK_METHODS)


def fanmod_executable_name() -> str:
    import platform

    return "LocalFANMOD - Windows" if platform.system().lower().startswith("win") else "LocalFANMOD"
