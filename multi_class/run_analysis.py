from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_class.config import (  # noqa: E402
    CONFIDENCE_STACK_METHODS,
    DEFAULT_RAW_CENTROIDS_PATH,
    MAX_CLASS_FEATURES,
    METHODS,
    RANDOM_STATE,
    RUNTIME_DIR,
    VARIANTS,
)
from multi_class.confidence_stacking import run_confidence_stack  # noqa: E402
from multi_class.data_artifacts import load_or_build_artifact, load_patient_classes  # noqa: E402
from multi_class.evaluate import evaluate_multiclass_panel  # noqa: E402
from multi_class.motif_panels import tune_and_select_panel  # noqa: E402


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-class melanoma discriminative motif analysis.")
    parser.add_argument("--variant", choices=[*VARIANTS.keys(), "all"], default=os.environ.get("VARIANT", "all"))
    parser.add_argument("--method", choices=[*METHODS, "all"], default=os.environ.get("METHOD", "all"))
    parser.add_argument("--force-artifact", action="store_true", default=env_bool("FORCE_ARTIFACT"))
    parser.add_argument("--force-preprocessing", action="store_true", default=env_bool("FORCE_PREPROCESSING"))
    parser.add_argument("--force-tuning", action="store_true", default=env_bool("FORCE_TUNING"))
    parser.add_argument("--force-evaluation", action="store_true", default=env_bool("FORCE_EVALUATION"))
    parser.add_argument(
        "--raw-centroids-path",
        type=Path,
        default=Path(os.environ.get("RAW_CENTROIDS_PATH", str(DEFAULT_RAW_CENTROIDS_PATH))),
    )
    parser.add_argument("--n-jobs", type=int, default=env_int("N_JOBS", 1), help="CISM/FANMOD build jobs.")
    parser.add_argument("--max-class-features", type=int, default=env_int("MAX_CLASS_FEATURES", MAX_CLASS_FEATURES))
    parser.add_argument("--random-state", type=int, default=env_int("RANDOM_STATE", RANDOM_STATE))
    parser.add_argument(
        "--max-patients-per-class",
        type=int,
        default=env_int("MAX_PATIENTS_PER_CLASS", None),
        help="Debug/smoke-test option that limits patients per class before tuning/evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = list(VARIANTS.values()) if args.variant == "all" else [VARIANTS[args.variant]]
    methods = list(METHODS) if args.method == "all" else [args.method]

    for variant in tqdm(variants, desc="Variants", unit="variant"):
        print(f"\n=== Variant: {variant.name} ===")
        artifact = load_or_build_artifact(
            variant,
            raw_centroids_path=args.raw_centroids_path,
            force_artifact=args.force_artifact,
            force_preprocessing=args.force_preprocessing,
            n_jobs=args.n_jobs,
        )
        motifs_df = artifact["discriminator"].cism.motifs_dataset.copy()
        patient_classes = load_patient_classes(variant.labels)
        for method in tqdm(methods, desc=f"{variant.name} methods", unit="method", leave=False):
            print(f"--- Method: {method} ---")
            output_dir = RUNTIME_DIR / variant.name / method
            if method in CONFIDENCE_STACK_METHODS:
                metrics = run_confidence_stack(
                    motifs_df,
                    patient_classes,
                    variant.labels,
                    method,
                    output_dir,
                    force_tuning=args.force_tuning,
                    force_evaluation=args.force_evaluation,
                    max_class_features=args.max_class_features,
                    random_state=args.random_state,
                    max_patients_per_class=args.max_patients_per_class,
                )
                print(
                    "Saved outputs to "
                    f"{output_dir} | n_confidence_features={metrics['n_confidence_features']} "
                    f"accuracy={metrics['accuracy']:.3f} "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
                    f"macro_ovr_auc={metrics['roc_auc_ovr_macro']:.3f}"
                )
                continue
            selected_motifs = tune_and_select_panel(
                motifs_df,
                patient_classes,
                variant.labels,
                method,
                output_dir,
                force_tuning=args.force_tuning,
                max_class_features=args.max_class_features,
                random_state=args.random_state,
                max_patients_per_class=args.max_patients_per_class,
            )
            metrics = evaluate_multiclass_panel(
                motifs_df,
                patient_classes,
                selected_motifs,
                variant.labels,
                output_dir,
                force_evaluation=args.force_evaluation,
                random_state=args.random_state,
                max_patients_per_class=args.max_patients_per_class,
            )
            print(
                "Saved outputs to "
                f"{output_dir} | n_motifs={metrics['n_motifs']} "
                f"accuracy={metrics['accuracy']:.3f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
                f"macro_ovr_auc={metrics['roc_auc_ovr_macro']:.3f}"
            )


if __name__ == "__main__":
    main()
