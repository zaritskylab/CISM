import torch
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from torch_geometric.data import Data

from config.settings.cross_validation_settings import LopoCVSettings
from config.settings.model_settings import ModelSettings
from config.settings.patient_sampling_settings import (
  MajorityUndersamplingSettings,
)
from config.settings.preprocess_settings import AugmentationSettings
from config.settings.training_settings import TrainingSettings
from utils.utils import audit_output_distribution, set_seed
from utils.batch_training_utils import train_batch, evaluate_batched
from utils.data_augmentation_utils import presample


def train_with_lopo_balanced(
    preprocessed_data: dict,
    patient_ids: list,
    *,
    device: torch.device,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    augmentation_settings: AugmentationSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
  """
  LOPO training with per-fold majority-class undersampling.

  For each fold the training patients are split by class label.
  If the majority class has more patients than the resolved cap,
  a random subset of that size is drawn.  The minority class is
  always kept in full.

  When `majority_n_patients` is None the cap is set dynamically to
  `len(minority_patients)` for that fold, guaranteeing a 1:1 ratio
  and pos_weight=1.0 every fold.

  When `n_majority_draws > 1`, the random majority draw is repeated
  that many times per fold.  A fresh model is trained for each draw
  and the resulting test predictions are averaged before recording.
  This reduces variance from a single unlucky majority subsample.

  Args:
    preprocessed_data (dict): Mapping from int index to PyG Data objects.
    patient_ids (np.ndarray): Patient ID per graph, aligned with keys of
      preprocessed_data.
    device (torch.device): Device to run training on.
    model_settings (ModelSettings): Model and loss construction.
    training_settings (TrainingSettings): LOPO, fitting, majority
      undersampling, and reproducibility settings.
    augmentation_settings (AugmentationSettings | None): Optional
      pre-sampling augmentation settings.

  Returns:
    tuple: (all_true_labels, all_pred_probs, all_test_patient_ids,
            fold_histories) as numpy arrays / list of dicts.

  """
  if not isinstance(training_settings, TrainingSettings):
    raise TypeError(
      'training_settings must be a TrainingSettings instance.'
    )

  cv_settings = training_settings.cross_validation
  if not isinstance(cv_settings, LopoCVSettings):
    raise ValueError(
      'train_with_lopo_balanced requires LopoCVSettings; received '
      f'{type(cv_settings).__name__}.'
    )

  sampling_settings = training_settings.patient_sampling
  if not isinstance(
    sampling_settings,
    MajorityUndersamplingSettings,
  ):
    raise ValueError(
      'train_with_lopo_balanced requires '
      'MajorityUndersamplingSettings.'
    )

  fit_settings = training_settings.fit
  if fit_settings.early_stopping.enabled:
    raise ValueError(
      'train_with_lopo_balanced does not support early stopping. '
      'Set fit.early_stopping.enabled to False.'
    )

  optimizer_settings = fit_settings.optimizer
  regularization_settings = fit_settings.regularization
  criterion = model_settings.loss_fn
  epochs = fit_settings.epochs
  lr = optimizer_settings.learning_rate
  weight_decay = optimizer_settings.weight_decay
  random_seed = training_settings.random_seed
  label_smoothing = regularization_settings.label_smoothing
  grad_clip_norm = regularization_settings.gradient_clip_norm
  majority_n_patients = sampling_settings.majority_n_patients
  n_majority_draws = sampling_settings.n_majority_draws
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

  all_true_labels = []
  all_pred_probs = []
  all_test_patient_ids = []
  fold_histories = []

  pbar = tqdm(
    enumerate(logo.split(data_indices, ys, groups=patient_ids)),
    total=n_splits,
    desc="LOPO Balanced Progress",
  )

  for fold, (train_val_idx, test_idx) in pbar:
    current_test_patient = patient_ids[test_idx][0]
    pbar.set_description(
      f"LOPO Balanced - Testing Patient: {current_test_patient}"
    )
    print(
      f"\n--- Fold {fold + 1} | Patient {current_test_patient} held out ---"
    )

    train_indices = data_indices[train_val_idx]
    test_indices = data_indices[test_idx]
    train_labels = ys[train_val_idx]

    # Per-fold majority undersampling at the patient level
    train_patient_ids = patient_ids[train_val_idx]
    unique_train_patients = np.unique(train_patient_ids)

    # Build per-patient label (label of that patient's FOVs, all equal)
    patient_label = {
      pid: int(train_labels[train_patient_ids == pid][0])
      for pid in unique_train_patients
    }

    class_0_patients = [p for p, l in patient_label.items() if l == 0]
    class_1_patients = [p for p, l in patient_label.items() if l == 1]

    # Identify majority and minority patient lists
    if len(class_0_patients) >= len(class_1_patients):
      majority_patients = class_0_patients
      minority_patients = class_1_patients
    else:
      majority_patients = class_1_patients
      minority_patients = class_0_patients

    # Dynamic cap: match minority size for exact 1:1 ratio each fold
    resolved_cap = (
      majority_n_patients
      if majority_n_patients is not None
      else len(minority_patients)
    )

    do_subsample = len(majority_patients) > resolved_cap
    print(
      f"  majority={len(majority_patients)}, "
      f"minority={len(minority_patients)}, "
      f"cap={resolved_cap}"
      + (" → subsampling" if do_subsample else " → no subsampling needed")
    )

    # Multiple independent majority draws — averaged at prediction time
    draw_pred_accumulator = None
    draw_true_fold = None
    draw_history = {'train_loss': []}

    for draw in range(n_majority_draws):
      draw_seed = random_seed + fold * 1000 + draw

      if do_subsample:
        rng = np.random.default_rng(draw_seed)
        sampled_majority = list(
          rng.choice(
            majority_patients, size=resolved_cap, replace=False
          )
        )
        kept_patients = set(minority_patients) | set(sampled_majority)
      else:
        kept_patients = set(unique_train_patients)

      # Filter to kept patients for this draw
      keep_mask = np.isin(train_patient_ids, list(kept_patients))
      draw_train_indices = data_indices[train_val_idx][keep_mask]
      draw_train_labels = train_labels[keep_mask]

      # pos_weight from the filtered training labels of this draw
      n_neg = (draw_train_labels == 0).sum()
      n_pos = (draw_train_labels == 1).sum()
      pos_weight = torch.tensor(
        [n_neg / (n_pos + 1e-8)], dtype=torch.float
      ).to(device)
      lopo_criterion = criterion(pos_weight=pos_weight)
      if n_majority_draws > 1:
        print(
          f"  [draw {draw + 1}/{n_majority_draws}] "
          f"pos_weight={pos_weight.item():.3f} "
          f"(n_neg={n_neg}, n_pos={n_pos})"
        )
      else:
        print(
          f"  pos_weight: {pos_weight.item():.3f}  "
          f"(n_neg={n_neg}, n_pos={n_pos})"
        )

      # Fresh model and optimizer per draw
      set_seed(draw_seed)
      input_size = preprocessed_data[0].x.shape[1]
      model = model_settings.model_class(
        input_dim=input_size,
        hidden_dim=model_settings.hidden_layers,
        dropout=model_settings.dropout_rate,
      ).to(device)
      optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
      )

      # Augmentation
      if augmentation_config:
        train_subset = {
          new: preprocessed_data[old]
          for new, old in enumerate(draw_train_indices)
        }
        augmented = presample(
          train_subset, augmentation_config, seed=draw_seed
        )
        train_list = [
          sub for subs in augmented.values() for sub in subs
        ]
        g = torch.Generator()
        g.manual_seed(draw_seed)
        train_loader = DataLoader(
          train_list, batch_size=8, shuffle=True, generator=g
        )
      else:
        train_list = [preprocessed_data[i] for i in draw_train_indices]
        g = torch.Generator()
        g.manual_seed(draw_seed)
        train_loader = DataLoader(
          train_list, batch_size=1, shuffle=True, generator=g
        )

      # Train for fixed epochs
      for epoch in range(1, epochs + 1):
        loss = train_batch(
          model, train_loader, optimizer, lopo_criterion,
          device, smoothing=label_smoothing,
          grad_clip_norm=grad_clip_norm,
        )
        draw_history['train_loss'].append(loss)

      # Evaluate on the held-out patient for this draw
      test_list = [preprocessed_data[i] for i in test_indices]
      test_loader = DataLoader(test_list, batch_size=1)
      _, y_true_draw, y_pred_draw = evaluate_batched(
        model, test_loader, lopo_criterion, device
      )

      # Accumulate predictions across draws for averaging
      if draw_pred_accumulator is None:
        draw_pred_accumulator = np.array(y_pred_draw, dtype=float)
        draw_true_fold = y_true_draw
      else:
        draw_pred_accumulator += np.array(y_pred_draw, dtype=float)

    fold_histories.append(draw_history)

    # Average predictions over all draws
    y_pred_fold = draw_pred_accumulator / n_majority_draws
    y_true_fold = draw_true_fold

    pred_mean = np.mean(y_pred_fold)
    print(
      f"  Patient {current_test_patient} | true={y_true_fold[0][0]:.0f} | "
      f"pred_mean={pred_mean:.3f} | n_fovs={len(test_indices)}"
      + (f" | draws={n_majority_draws}" if n_majority_draws > 1 else "")
    )

    audit_output_distribution(
      model, preprocessed_data, draw_train_indices, device, "Train"
    )
    audit_output_distribution(
      model, preprocessed_data, test_indices, device, "Test"
    )

    all_true_labels.extend(y_true_fold)
    all_pred_probs.extend(y_pred_fold)
    all_test_patient_ids.extend(
      [current_test_patient] * len(test_idx)
    )

  print("\n---------- Training Details ----------")
  if augmentation_config:
    print(
      f"Created extra {augmentation_config['n_subs_per_fov']} data "
      f"points per FOV, with each in size of "
      f"{augmentation_config['sub_ratio']*100}% of original FOV"
    )
  if majority_n_patients is not None:
    print(
      f"Majority class undersampled to at most "
      f"{majority_n_patients} patients per fold"
    )
  else:
    print(
      "Majority class dynamically capped to minority size per fold "
      "(1:1 balance)"
    )
  if n_majority_draws > 1:
    print(
      f"Predictions averaged over {n_majority_draws} independent "
      f"majority draws per fold"
    )
  print(f"Used model {type(model).__name__} for the training process")
  print("Model Architecture:")
  print("Model Config - ")
  for param_name, param_value in model_settings.model_dump().items():
    print(f"  {param_name}: {param_value}")
  sample_data = Data(
    x=preprocessed_data[0].x.to(device),
    edge_index=preprocessed_data[0].edge_index.to(device),
  )
  print(summary(model, sample_data))
  print(
    f"Used optimizer {type(optimizer).__name__} with learning rate: "
    f"{lr} and weight decay: {weight_decay}"
  )
  if label_smoothing > 0.0:
    print(f"Used label smoothing: {label_smoothing}")
  if grad_clip_norm is not None:
    print(f"Using gradient clipping with max norm {grad_clip_norm}")
  criterion_name = getattr(
    criterion,
    '__name__',
    type(criterion).__name__,
  )
  print(f"Using Loss Function: {criterion_name}")
  print(
    f"Trained the model on {epochs} epochs with seed {random_seed}"
  )

  return (
    np.array(all_true_labels),
    np.array(all_pred_probs),
    np.array(all_test_patient_ids),
    fold_histories,
  )
