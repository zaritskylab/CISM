# -------------------- All Imports --------------------
# Packages
import os
import time
import torch

# Preprocessing Imports
from graphs_preprocess.one_hot.data_process import CellGraphDataset

# Utils Imports
from utils.utils import (
    print_job_execution_details,
    set_seed,
    preprocess_dataset_v2,
    verify_dataset_label_mapping,
    calculate_fov_and_patient_auc,
)
from utils.plotting_utils import (
    plot_patient_confusion_matrix,
    plot_cv_losses,
    plot_nested_cv_losses,
    plot_patient_roc_curve,
    plot_prediction_probabilities,
)
from utils.cv_options import CVOptions, InnerCVOptions

# Training methods
from training_methods.fold_cross_validation_training import train_with_group_cv
from training_methods.lopo_training import train_with_lopo_updated


# -------------------- Configurations & Consts --------------------
# Which dataset (a config/<name>_config.py module) and which seed:
#   python model_training.py --dataset seaad_adnc --seed 0
# Export PYTHONHASHSEED=<seed> before starting Python; the run exits on a mismatch.
import argparse
import importlib

DATASET_CONFIGS = {
    "melanoma": "config.melanoma_config",
    "tnbc": "config.tnbc_config",
    "crc": "config.crc_config",
    "human_glioma": "config.human_glioma_config",
    "seaad_adnc": "config.seaad_adnc_config",
}
parser = argparse.ArgumentParser(description="Train the GraphSAGE baseline.")
parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="human_glioma")
parser.add_argument("--seed", type=int, default=4)
args = parser.parse_args()

_config = importlib.import_module(DATASET_CONFIGS[args.dataset])
DATA_SETTINGS = _config.DATA_SETTINGS
MODEL_SETTINGS = _config.MODEL_SETTINGS
PREPROCESS_SETTINGS = _config.PREPROCESS_SETTINGS
TRAINING_SETTINGS = _config.TRAINING_SETTINGS
PLOTS_SETTINGS = _config.PLOTS_SETTINGS
# --------------------------------------------------------------------

# Script derived consts
TRAINING_SETTINGS = TRAINING_SETTINGS.model_copy(update={'random_seed': args.seed})
PLOTS_SETTINGS = PLOTS_SETTINGS.model_copy(
  update={'random_seed': TRAINING_SETTINGS.random_seed}
)
job_id = print_job_execution_details(TRAINING_SETTINGS.random_seed)
EXPERIMENT_FOLDER_NAME = PLOTS_SETTINGS.full_plots_prefix + "-" + job_id
OUTPUT_DIR = (
    PLOTS_SETTINGS.results_folder_path_prefix
    + f"/{MODEL_SETTINGS.model_class.__name__}"
    + "/"
    + EXPERIMENT_FOLDER_NAME
)


# -------------------- Script Initialization --------------------
# Define where to save runs results
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Job Started. Plots and models will be saved to: {OUTPUT_DIR}")
# Setting up device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Measuring the time taken for the entire process
start_time = time.time()
# Set the random seed for reproducibility
set_seed(TRAINING_SETTINGS.random_seed)


# -------------------- Data Preprocessing --------------------
print("\n---------- Data Preprocessing ----------", end="")
dataset = CellGraphDataset(
    DATA_SETTINGS.bottom_up_data_full_dir,
    PREPROCESS_SETTINGS.groups,
    max_distance=PREPROCESS_SETTINGS.max_distance,
    plots_output_dir=OUTPUT_DIR,
    column_map=DATA_SETTINGS.column_mapping.model_dump(),
)
preprocessed_data, patient_ids = preprocess_dataset_v2(dataset)
verify_dataset_label_mapping(dataset)
dataset.print_node_feature_example()
print("Finished preprocessing the dataset")


# -------------------- Model Training --------------------
if TRAINING_SETTINGS.cross_validation.cv_type == CVOptions.FOLDS:
    print(
        f"\n---------- Starting {TRAINING_SETTINGS.cv_full_type} Training ----------",
        end="",
    )
    cv_result = train_with_group_cv(
        preprocessed_data,
        patient_ids,
        model_settings=MODEL_SETTINGS,
        training_settings=TRAINING_SETTINGS,
        augmentation_settings=PREPROCESS_SETTINGS.augmentation,
        device=device,
    )

    y_true, y_pred, histories, bundle = cv_result
    # StratifiedGroupKFold(shuffle=True) does not preserve patient_ids'
    # original order, so use the fold-aligned array returned in bundle
    # instead of the raw patient_ids for any positional alignment.
    test_patient_ids = bundle["all_test_patient_ids"]
    assert len(test_patient_ids) == len(y_pred) == len(y_true)
    print("\n---------- Final Results ----------")
    # Calculate and print AUC score on the test set (FOV & Patient Level)
    fov_auc, patient_auc = calculate_fov_and_patient_auc(
        y_true,
        y_pred,
        test_patient_ids,
        agg_method=TRAINING_SETTINGS.evaluation.patient_aggregation,
    )
    print(
        f"FOV-level AUC Score on Test Set with {TRAINING_SETTINGS.cv_full_type} CV (Seed {TRAINING_SETTINGS.random_seed}): {fov_auc:.4f}"
    )
    print(
        f"Patient-level AUC Score on Test Set with {TRAINING_SETTINGS.cv_full_type} CV (Seed {TRAINING_SETTINGS.random_seed}): {patient_auc:.4f}"
    )
    # Plotting the Train Loss curve, ROC Curve & Confusion Matrix
    if TRAINING_SETTINGS.cross_validation.inner.option == InnerCVOptions.NESTEDCV:
        plot_nested_cv_losses(histories, PLOTS_SETTINGS.full_plots_prefix, OUTPUT_DIR)
    else:
        plot_cv_losses(histories, PLOTS_SETTINGS.full_plots_prefix, OUTPUT_DIR)
    plot_prediction_probabilities(y_pred, OUTPUT_DIR)
    plot_patient_roc_curve(
        y_true,
        y_pred,
        test_patient_ids,
        TRAINING_SETTINGS.cv_full_type,
        PLOTS_SETTINGS.full_plots_prefix,
        OUTPUT_DIR,
        agg_method=TRAINING_SETTINGS.evaluation.patient_aggregation,
    )
    plot_patient_confusion_matrix(
        y_true,
        y_pred,
        test_patient_ids,
        PREPROCESS_SETTINGS.groups,
        PLOTS_SETTINGS.full_plots_prefix,
        OUTPUT_DIR,
    )
    print("Finished plotting the Train Loss curve, ROC curve & Confusion Matrix")

elif TRAINING_SETTINGS.cross_validation.cv_type == CVOptions.LOPO:
    print(
        f"\n---------- Starting {TRAINING_SETTINGS.cv_full_type} Training ----------",
        end="",
    )
    y_true, y_pred, test_patient_ids, histories = train_with_lopo_updated(
        preprocessed_data,
        patient_ids,
        model_settings=MODEL_SETTINGS,
        training_settings=TRAINING_SETTINGS,
        augmentation_settings=PREPROCESS_SETTINGS.augmentation,
        device=device,
    )
    print("\n---------- Final Results ----------")
    # Calculate and print AUC score on the test set (FOV & Patient Level)
    fov_auc, patient_auc = calculate_fov_and_patient_auc(
        y_true,
        y_pred,
        test_patient_ids,
        agg_method=TRAINING_SETTINGS.evaluation.patient_aggregation,
    )
    print(
        f"FOV-level AUC Score on Test Set with {TRAINING_SETTINGS.cv_full_type} CV (Seed {TRAINING_SETTINGS.random_seed}): {fov_auc:.4f}"
    )
    print(
        f"Patient-level AUC Score on Test Set with {TRAINING_SETTINGS.cv_full_type} CV (Seed {TRAINING_SETTINGS.random_seed}): {patient_auc:.4f}"
    )
    # Plotting the ROC Curve & Confusion Matrix
    plot_patient_roc_curve(
        y_true,
        y_pred,
        test_patient_ids,
        TRAINING_SETTINGS.cv_full_type,
        PLOTS_SETTINGS.full_plots_prefix,
        OUTPUT_DIR,
        agg_method=TRAINING_SETTINGS.evaluation.patient_aggregation,
    )
    plot_patient_confusion_matrix(
        y_true,
        y_pred,
        test_patient_ids,
        PREPROCESS_SETTINGS.groups,
        PLOTS_SETTINGS.full_plots_prefix,
        OUTPUT_DIR,
    )
    print("Finished plotting the ROC curve & Confusion Matrix")

else:
    print(
        "No cross-validation method selected. Please choose either 'Fold', 'LOPO' or 'RandomForest'."
    )


# Measure the total time taken for the entire process
end_time = time.time()
total_time = end_time - start_time
days, remainder = divmod(total_time, 86400)
formatted_time = (
    f"{int(days)}d " f"{time.strftime('%Hh %Mm %Ss', time.gmtime(remainder))}"
)
print(f"Total time taken for the entire process: {formatted_time}")
