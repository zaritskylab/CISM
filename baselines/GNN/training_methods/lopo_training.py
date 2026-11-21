import torch
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary

from config.settings.cross_validation_settings import LopoCVSettings
from config.settings.model_settings import ModelSettings
from config.settings.patient_sampling_settings import AllPatientsSettings
from config.settings.preprocess_settings import AugmentationSettings
from config.settings.training_settings import TrainingSettings
from training_methods.class_weight_utils import (
  build_pos_weighted_criterion,
)
from training_methods.patient_split_utils import (
  patient_level_labels,
  stratified_patient_train_val_indices,
)
from utils.data_augmentation_utils import presample
from utils.batch_training_utils import (
  evaluate_batched,
  train_and_validate_batch_early_stopping,
  train_batch,
)
from utils.utils import audit_output_distribution, set_seed


def _build_train_loader(preprocessed_data: dict,
                         indices: np.ndarray,
                         augmentation_config: dict | None,
                         seed: int) -> DataLoader:
  """
  Build the (optionally augmented) training DataLoader for one fold.

  Args:
    preprocessed_data (dict): Mapping from sample index to graph Data.
    indices (np.ndarray): Sample indices to include in the loader.
    augmentation_config (dict | None): sub_ratio/n_subs_per_fov
      config, or None to disable augmentation.
    seed (int): Seed for shuffling and augmentation sampling.

  Returns:
    (DataLoader): Loader over the (augmented) training samples.

  """
  g = torch.Generator()
  g.manual_seed(seed)
  if augmentation_config:
    subset = {
      new: preprocessed_data[old] for new, old in enumerate(indices)
    }
    augmented = presample(subset, augmentation_config, seed=seed)
    train_list = [sub for subs in augmented.values() for sub in subs]
    return DataLoader(train_list, batch_size=8, shuffle=True, generator=g)
  train_list = [preprocessed_data[i] for i in indices]
  return DataLoader(train_list, batch_size=1, shuffle=True, generator=g)


def train_with_lopo_updated(
  preprocessed_data: dict,
  patient_ids: list,
  *,
  device: torch.device,
  model_settings: ModelSettings,
  training_settings: TrainingSettings,
  augmentation_settings: AugmentationSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
  """
  Train and evaluate a GNN with leave-one-patient-out CV.

  Args:
    preprocessed_data (dict): Mapping from sample index to graph Data.
    patient_ids (list): Patient id per sample, aligned with sorted
      preprocessed_data keys.
    device (torch.device): Device to train on.
    model_settings (ModelSettings): Model and loss construction.
    training_settings (TrainingSettings): LOPO, fitting, evaluation,
      and reproducibility settings.
    augmentation_settings (AugmentationSettings | None): Optional
      pre-sampling augmentation settings.

  Returns:
    (tuple): (y_true, y_pred, test_patient_ids, fold_histories) over
      all held-out patients.

  """
  if not isinstance(training_settings, TrainingSettings):
    raise TypeError(
      'training_settings must be a TrainingSettings instance.'
    )

  cv_settings = training_settings.cross_validation
  if not isinstance(cv_settings, LopoCVSettings):
    raise ValueError(
      'train_with_lopo_updated requires LopoCVSettings; received '
      f'{type(cv_settings).__name__}.'
    )
  if not isinstance(
    training_settings.patient_sampling,
    AllPatientsSettings,
  ):
    raise ValueError(
      'train_with_lopo_updated requires AllPatientsSettings. Use '
      'train_with_lopo_balanced for majority undersampling.'
    )

  fit_settings = training_settings.fit
  optimizer_settings = fit_settings.optimizer
  regularization_settings = fit_settings.regularization
  criterion = model_settings.loss_fn
  epochs = fit_settings.epochs
  lr = optimizer_settings.learning_rate
  weight_decay = optimizer_settings.weight_decay
  random_seed = training_settings.random_seed
  smoothing = regularization_settings.label_smoothing
  grad_clip_norm = regularization_settings.gradient_clip_norm
  use_early_stopping = fit_settings.early_stopping.enabled
  val_patient_ratio = cv_settings.validation_patient_ratio
  early_stopping_patience = fit_settings.early_stopping.patience
  augmentation_config = (
    augmentation_settings.model_dump()
    if augmentation_settings is not None
    else None
  )

  data_indices = np.array(sorted(preprocessed_data.keys()))
  ys = np.array([preprocessed_data[i].y.item() for i in data_indices])
  patient_ids = np.array(patient_ids)

  unique, counts = np.unique(ys, return_counts=True)
  dist = {int(k): int(v) for k, v in zip(unique, counts)}
  print(f"\nClass distribution in full dataset: {dist}")

  logo = LeaveOneGroupOut()
  n_splits = logo.get_n_splits(groups=patient_ids)
  patient_level_labels(ys, patient_ids, context="LOPO dataset")
  logo_splits = list(logo.split(data_indices, ys, groups=patient_ids))

  # Check every validation split before fitting the first expensive fold.
  if use_early_stopping:
    for fold, (train_val_idx, _) in enumerate(logo_splits):
      stratified_patient_train_val_indices(
        ys[train_val_idx],
        patient_ids[train_val_idx],
        val_patient_ratio,
        random_state=random_seed + fold,
        context=f"LOPO fold {fold + 1} validation split",
      )

  all_true_labels = []
  all_pred_probs = []
  all_test_patient_ids = []
  fold_histories = []

  pbar = tqdm(
    enumerate(logo_splits),
    total=n_splits, desc="LOPO Progress",
  )

  for fold, (train_val_idx, test_idx) in pbar:
    current_test_patient = patient_ids[test_idx][0]
    pbar.set_description(f"LOPO - Testing Patient: {current_test_patient}")
    print(f"\n--- Fold {fold + 1} | Patient {current_test_patient} "
          f"held out ---")

    train_indices = data_indices[train_val_idx]  # all non-test patients
    test_indices = data_indices[test_idx]

    # Model & optimizer — fresh for every patient, seeded per fold
    set_seed(random_seed + fold)
    input_size = preprocessed_data[0].x.shape[1]
    model = model_settings.model_class(
      input_dim=input_size,
      hidden_dim=model_settings.hidden_layers,
      dropout=model_settings.dropout_rate,
    ).to(device)
    optimizer = torch.optim.AdamW(
      model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    if use_early_stopping:
      # Carve a patient-grouped validation split out of the training
      # patients so we can early-stop instead of blindly training for
      # a fixed number of epochs. The fixed-epoch loop was letting
      # logits blow up to +-200 and left ~45% of held-out patients
      # saturated at a prediction of exactly 0 or 1.
      sub_train_pos, sub_val_pos = stratified_patient_train_val_indices(
        ys[train_val_idx],
        patient_ids[train_val_idx],
        val_patient_ratio,
        random_state=random_seed + fold,
        context=f"LOPO fold {fold + 1} validation split",
      )
      sub_train_indices = data_indices[train_val_idx[sub_train_pos]]
      val_indices = data_indices[train_val_idx[sub_val_pos]]
      lopo_criterion = build_pos_weighted_criterion(
        ys[train_val_idx[sub_train_pos]],
        criterion,
        device,
        f"LOPO fold {fold + 1} train split",
      )

      if augmentation_config:
        print(f"  Augmenting train data for patient "
              f"{current_test_patient}...")
      train_loader = _build_train_loader(
        preprocessed_data, sub_train_indices, augmentation_config,
        random_seed + fold,
      )
      # Validation is always the original, un-augmented FOVs
      val_list = [preprocessed_data[i] for i in val_indices]
      val_loader = DataLoader(val_list, batch_size=1, shuffle=False)

      best_state, history = train_and_validate_batch_early_stopping(
        model, train_loader, val_loader, optimizer, lopo_criterion,
        device, epochs=epochs, smoothing=smoothing,
        patience=early_stopping_patience, grad_clip_norm=grad_clip_norm,
      )
      model.load_state_dict(best_state)
    else:
      # No validation split: use the configured fixed epoch count.
      lopo_criterion = build_pos_weighted_criterion(
        ys[train_val_idx],
        criterion,
        device,
        f"LOPO fold {fold + 1} train split",
      )
      if augmentation_config:
        print(f"  Augmenting train data for patient "
              f"{current_test_patient}...")
      train_loader = _build_train_loader(
        preprocessed_data, train_indices, augmentation_config,
        random_seed + fold,
      )
      history = {'train_loss': []}
      for _ in range(1, epochs + 1):
        loss = train_batch(
          model, train_loader, optimizer, lopo_criterion, device,
          smoothing=smoothing, grad_clip_norm=grad_clip_norm,
        )
        history['train_loss'].append(loss)

    fold_histories.append(history)

    # Evaluate on the held-out patient
    test_list = [preprocessed_data[i] for i in test_indices]
    test_loader = DataLoader(test_list, batch_size=1)
    _, y_true_fold, y_pred_fold = evaluate_batched(
      model, test_loader, lopo_criterion, device,
    )

    pred_mean = np.mean(y_pred_fold)
    print(f"  Patient {current_test_patient} | "
          f"true={y_true_fold[0][0]:.0f} | pred_mean={pred_mean:.3f} | "
          f"n_fovs={len(test_indices)}")

    audit_output_distribution(
      model, preprocessed_data, train_indices, device, "Train",
    )
    audit_output_distribution(
      model, preprocessed_data, test_indices, device, "Test",
    )

    all_true_labels.extend(y_true_fold)
    all_pred_probs.extend(y_pred_fold)
    all_test_patient_ids.extend([current_test_patient] * len(test_idx))

  print("\n---------- Training Details ----------")
  if augmentation_config:
    print(f"Created extra {augmentation_config['n_subs_per_fov']} data "
          f"points per FOV, with each in size of "
          f"{augmentation_config['sub_ratio'] * 100}% of original FOV")
  print(f"Used model {type(model).__name__} for the training process")
  print("Model Architecture:")
  sample_data = Data(
    x=preprocessed_data[0].x.to(device),
    edge_index=preprocessed_data[0].edge_index.to(device),
  )
  print(summary(model, sample_data))
  print(f"Used optimizer {type(optimizer).__name__} with learning rate: "
        f"{lr} and weight decay: {weight_decay}")
  criterion_name = getattr(
    criterion,
    '__name__',
    type(criterion).__name__,
  )
  print(f"Using Loss Function: {criterion_name}")
  print(f"Using label smoothing: {smoothing}")
  if use_early_stopping:
    print(f"Using validation-based early stopping "
          f"(val_patient_ratio={val_patient_ratio}, "
          f"patience={early_stopping_patience})")
  if grad_clip_norm is not None:
    print(f"Using gradient clipping with max norm {grad_clip_norm}")
  print(f"Trained the model on up to {epochs} epochs with seed "
        f"{random_seed}")

  return (
    np.array(all_true_labels),
    np.array(all_pred_probs),
    np.array(all_test_patient_ids),
    fold_histories,
  )
