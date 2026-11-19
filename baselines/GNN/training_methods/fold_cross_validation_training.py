import torch
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from torch_geometric.data import Data

from config.settings.cross_validation_settings import (
    FoldCVSettings,
    NestedInnerCVSettings,
    TrainTestSplitSettings,
)
from config.settings.fit_settings import FitSettings
from config.settings.model_settings import ModelSettings
from config.settings.patient_sampling_settings import AllPatientsSettings
from config.settings.preprocess_settings import AugmentationSettings
from config.settings.training_settings import TrainingSettings
from training_methods.class_weight_utils import (
    build_pos_weighted_criterion,
)
from training_methods.patient_split_utils import (
    stratified_patient_train_val_indices,
    validate_stratified_group_kfold,
)
from utils.batch_training_utils import (
    train_and_validate_batch,
    train_and_validate_batch_early_stopping,
    evaluate_batched,
    train_fixed_epochs,
)
from utils.utils import check_split_distribution, set_seed
from utils.data_augmentation_utils import presample
from utils.metric_utils import safe_roc_auc


def _train_and_validate(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    fit_settings: FitSettings,
):
    # Dispatches to the early-stopping trainer when opted in, else the
    # original fixed-epoch, best-val-loss-only trainer.
    regularization = fit_settings.regularization
    if fit_settings.early_stopping.enabled:
        return train_and_validate_batch_early_stopping(
            model, train_loader, val_loader, optimizer, criterion, device,
            epochs=fit_settings.epochs,
            smoothing=regularization.label_smoothing,
            patience=fit_settings.early_stopping.patience,
            grad_clip_norm=regularization.gradient_clip_norm,
        )
    return train_and_validate_batch(
        model, train_loader, val_loader, optimizer, criterion, device,
        fit_settings.epochs,
        regularization.label_smoothing,
        grad_clip_norm=regularization.gradient_clip_norm,
    )


def train_with_group_cv(
    preprocessed_data: dict,
    patient_ids: list,
    *,
    device: torch.device,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    augmentation_settings: AugmentationSettings | None = None,
):
    """Train with stratified group CV using validated settings objects."""
    if not isinstance(training_settings, TrainingSettings):
        raise TypeError(
            'training_settings must be a TrainingSettings instance.'
        )

    cv_settings = training_settings.cross_validation
    if not isinstance(cv_settings, FoldCVSettings):
        raise ValueError(
            'train_with_group_cv requires FoldCVSettings; received '
            f'{type(cv_settings).__name__}.'
        )
    if not isinstance(
        training_settings.patient_sampling,
        AllPatientsSettings,
    ):
        raise ValueError(
            'train_with_group_cv currently requires '
            'AllPatientsSettings.'
        )

    fit_settings = training_settings.fit
    optimizer_settings = fit_settings.optimizer
    regularization_settings = fit_settings.regularization
    inner_settings = cv_settings.inner
    criterion = model_settings.loss_fn
    epochs = fit_settings.epochs
    lr = optimizer_settings.learning_rate
    weight_decay = optimizer_settings.weight_decay
    random_seed = training_settings.random_seed
    smoothing = regularization_settings.label_smoothing
    grad_clip_norm = regularization_settings.gradient_clip_norm
    use_early_stopping = fit_settings.early_stopping.enabled
    early_stopping_patience = fit_settings.early_stopping.patience
    augmentation_config = (
        augmentation_settings.model_dump()
        if augmentation_settings is not None
        else None
    )

    data_indices = np.array(sorted(list(preprocessed_data.keys()))) # Force deterministic order    
    ys = np.array([preprocessed_data[i].y.item() for i in data_indices])
    patient_ids = np.array(patient_ids)

    validate_stratified_group_kfold(
        ys,
        patient_ids,
        cv_settings.num_folds,
        context="Outer grouped CV",
    )
    
    all_out_of_fold_true = []
    all_out_of_fold_pred = []
    all_out_of_fold_pids = []
    fold_histories = []
    fold_results_bundle = {}

    gkf = StratifiedGroupKFold(
        n_splits=cv_settings.num_folds,
        shuffle=True,
        random_state=random_seed,
    )
    fold_pbar = tqdm(
        enumerate(gkf.split(data_indices, ys, groups=patient_ids)),
        total=cv_settings.num_folds,
        desc=f"{cv_settings.num_folds}-Fold CV",
    )

    for fold, (train_val_idx, test_idx) in fold_pbar:
        fold_pbar.set_description(f"Processing Fold {fold + 1}")
        print(f"--- Processing Fold {fold + 1} ---")

        # Print out the train & test patients in the current fold
        test_patient_ids = np.unique(patient_ids[test_idx])
        train_patient_ids = np.unique(patient_ids[train_val_idx])
        print(f"Fold {fold+1} test patients: {sorted(test_patient_ids.tolist())}")
        print(f"Fold {fold+1} train patients: {sorted(train_patient_ids.tolist())}")

        # Indices for this fold
        test_indices = data_indices[test_idx]

        # Checking NN vs NP distribution in the splitting
        check_split_distribution(train_val_idx, patient_ids, preprocessed_data, f"Fold {fold+1} | Train+Val")
        check_split_distribution(test_idx, patient_ids, preprocessed_data, f"Fold {fold+1} | Test")
        
        input_size = preprocessed_data[0].x.shape[1]

        # Initialize Model + setting seed for each fold-model
        set_seed(random_seed + fold)
        model = model_settings.model_class(
            input_dim=input_size,
            hidden_dim=model_settings.hidden_layers,
            dropout=model_settings.dropout_rate,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if isinstance(inner_settings, TrainTestSplitSettings):
            sub_train_idx, sub_val_idx = (
                stratified_patient_train_val_indices(
                    ys[train_val_idx],
                    patient_ids[train_val_idx],
                    inner_settings.validation_patient_ratio,
                    random_state=random_seed + fold,
                    context=f"Fold {fold + 1} validation split",
                )
            )
            
            train_indices = data_indices[train_val_idx[sub_train_idx]]
            val_indices = data_indices[train_val_idx[sub_val_idx]]
            fold_criterion = build_pos_weighted_criterion(
                ys[train_val_idx[sub_train_idx]],
                criterion,
                device,
                f"Fold {fold + 1} train split",
            )

            # Checking NN vs NP distribution in the splitting
            check_split_distribution(train_indices, patient_ids, preprocessed_data, f"Fold {fold+1} | 80-20 Train")
            check_split_distribution(val_indices, patient_ids, preprocessed_data, f"Fold {fold+1} | 80-20 Val")

            # AUGMENTATION STEP
            if augmentation_config:
                print(f" > Augmenting Fold {fold+1} Train Data...")
                # Augment only the training indices
                train_subset = {new: preprocessed_data[old] for new, old in enumerate(train_indices)}
                augmented_dict = presample(train_subset, augmentation_config, seed=random_seed+fold)
                # Convert to List of Data objects and wrap in DataLoader
                train_list = [sub for subs in augmented_dict.values() for sub in subs]
                g = torch.Generator()
                g.manual_seed(random_seed + fold)
                train_loader = DataLoader(train_list, batch_size=8, shuffle=True, generator=g) # Train using augmented dict, but validate on original val_indices (un-augmented)
            else:
                train_list = [preprocessed_data[i] for i in train_indices]
                g = torch.Generator()
                g.manual_seed(random_seed + fold)
                train_loader = DataLoader(train_list, batch_size=1, shuffle=True, generator=g)
            
            # Validation is always the original full FOVs
            val_list = [preprocessed_data[i] for i in val_indices]
            val_loader = DataLoader(val_list, batch_size=1, shuffle=False)

            best_state, history = _train_and_validate(
                model, train_loader, val_loader, optimizer, fold_criterion, device,
                fit_settings,
            )
            model.load_state_dict(best_state)
            fold_histories.append(history)

        elif isinstance(inner_settings, NestedInnerCVSettings):
            validate_stratified_group_kfold(
                ys[train_val_idx],
                patient_ids[train_val_idx],
                inner_settings.num_folds,
                context=f"Fold {fold + 1} inner grouped CV",
            )
            # We perform another StratifiedGroupKFold inside the outer training set
            inner_kf = StratifiedGroupKFold(
                n_splits=inner_settings.num_folds,
                shuffle=True,
                random_state=random_seed,
            )
            inner_best_epochs = []
            nested_history = {
                'train_loss': [],
                'val_loss': [],
                'best_epochs': [],
            }

            print(
                "  > Starting Nested CV "
                f"({inner_settings.num_folds}-Fold Inner)..."
            )
            
            # Split the train_val_idx using the patient groups
            for inner_f, (i_train_idx, i_val_idx) in enumerate(inner_kf.split(train_val_idx, ys[train_val_idx], groups=patient_ids[train_val_idx])):
                # Map back to global data indices
                inner_train_indices = data_indices[train_val_idx[i_train_idx]]
                inner_val_indices = data_indices[train_val_idx[i_val_idx]]
                inner_criterion = build_pos_weighted_criterion(
                    ys[train_val_idx[i_train_idx]],
                    criterion,
                    device,
                    (
                        f"Fold {fold + 1} inner fold "
                        f"{inner_f + 1} train split"
                    ),
                )
                
                # Fresh model for each inner fold
                inner_model = model_settings.model_class(
                    input_dim=input_size,
                    hidden_dim=model_settings.hidden_layers,
                    dropout=model_settings.dropout_rate,
                ).to(device)
                inner_opt = torch.optim.AdamW(inner_model.parameters(), lr=lr, weight_decay=weight_decay)
                
                print(f"    - Inner Fold {inner_f+1}")
                check_split_distribution(inner_train_indices, patient_ids, preprocessed_data, f"Inner Fold {inner_f+1} Train")
                
                # AUGMENTATION STEP
                if augmentation_config:
                    print(f" > Augmenting (Fold {fold+1} | Inner Fold {inner_f}) Train Data...")
                    inner_subset = {new: preprocessed_data[old] for new, old in enumerate(inner_train_indices)}
                    aug_inner = presample(inner_subset, augmentation_config, seed=random_seed+inner_f)
                    i_train_list = [sub for subs in aug_inner.values() for sub in subs]
                    g = torch.Generator()
                    g.manual_seed(random_seed + fold)
                    i_train_loader = DataLoader(i_train_list, batch_size=8, shuffle=True, generator=g)
                else:
                    i_train_list = [preprocessed_data[i] for i in inner_train_indices]
                    g = torch.Generator()
                    g.manual_seed(random_seed + fold)
                    i_train_loader = DataLoader(i_train_list, batch_size=1, shuffle=True, generator=g)
                
                i_val_list = [preprocessed_data[i] for i in inner_val_indices]
                i_val_loader = DataLoader(i_val_list, batch_size=1, shuffle=False)

                _, history = _train_and_validate(
                    inner_model, i_train_loader, i_val_loader, inner_opt,
                    inner_criterion, device,
                    fit_settings,
                )

                # Epoch numbers are one-based, while np.argmin returns a
                # zero-based index.
                best_epoch = int(np.argmin(history['val_loss'])) + 1
                inner_best_epochs.append(best_epoch)
                print(
                    f"    - Inner Fold {inner_f + 1} best epoch: "
                    f"{best_epoch}"
                )
                
                nested_history['train_loss'].append(history['train_loss'])
                nested_history['val_loss'].append(history['val_loss'])
                nested_history['best_epochs'].append(best_epoch)

            # Use the median best epoch so one unusually easy or difficult
            # inner validation fold does not determine the final duration.
            refit_epochs = max(1, int(np.median(inner_best_epochs)))
            nested_history['refit_epochs'] = refit_epochs

            print(
                f"  > Inner-fold best epochs: {inner_best_epochs}; "
                f"refitting for {refit_epochs} epochs."
            )

            # Every outer-training graph participates in the final refit.
            refit_train_indices = data_indices[train_val_idx]
            fold_criterion = build_pos_weighted_criterion(
                ys[train_val_idx],
                criterion,
                device,
                f"Fold {fold + 1} final refit",
            )

            if augmentation_config:
                print(f" > Augmenting Fold {fold + 1} Final Refit Data...")
                refit_subset = {
                    new: preprocessed_data[old]
                    for new, old in enumerate(refit_train_indices)
                }
                augmented_refit = presample(
                    refit_subset,
                    augmentation_config,
                    seed=random_seed + fold,
                )
                refit_train_list = [
                    subgraph
                    for subgraphs in augmented_refit.values()
                    for subgraph in subgraphs
                ]
                refit_batch_size = 8
            else:
                refit_train_list = [
                    preprocessed_data[i] for i in refit_train_indices
                ]
                refit_batch_size = 1

            refit_generator = torch.Generator()
            refit_generator.manual_seed(random_seed + fold)
            refit_train_loader = DataLoader(
                refit_train_list,
                batch_size=refit_batch_size,
                shuffle=True,
                generator=refit_generator,
            )

            # Reinitialize after inner CV so this model has never seen an
            # inner validation fold or the outer test fold.
            set_seed(random_seed + fold)
            model = model_settings.model_class(
                input_dim=input_size,
                hidden_dim=model_settings.hidden_layers,
                dropout=model_settings.dropout_rate,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            refit_history = train_fixed_epochs(
                model,
                refit_train_loader,
                optimizer,
                fold_criterion,
                device,
                epochs=refit_epochs,
                smoothing=smoothing,
                grad_clip_norm=grad_clip_norm,
            )
            nested_history['refit_train_loss'] = (
                refit_history['train_loss']
            )
            fold_histories.append(nested_history)

        else:
            raise TypeError(
                'Unsupported inner cross-validation settings: '
                f'{type(inner_settings).__name__}.'
            )

        # Evaluate on the hold-out test fold
        test_list = [preprocessed_data[i] for i in test_indices]
        test_loader = DataLoader(test_list, batch_size=1)
        val_loss, y_true_fold, y_pred_fold = evaluate_batched(model, test_loader, fold_criterion, device)  

        # --- SAVE FOLD DATA ---
        all_out_of_fold_true.extend(y_true_fold)
        all_out_of_fold_pred.extend(y_pred_fold)
        all_out_of_fold_pids.extend(patient_ids[test_idx])
        fold_results_bundle[fold + 1] = {
            'model_state': model.state_dict(),
            'test_indices': test_indices,
            'auc': safe_roc_auc(
                y_true_fold,
                y_pred_fold,
                context=f"Outer fold {fold + 1}",
            )
        }

        print(f"Fold {fold+1} | y_pred: {y_pred_fold.min():.3f}–{y_pred_fold.max():.3f} "
                f"| mean: {y_pred_fold.mean():.3f} | std: {y_pred_fold.std():.3f}")
        train_labels = ys[train_val_idx]
        test_labels = ys[test_idx]
        print(f"Fold {fold+1} | Train: {train_labels.sum():.0f} pos / {len(train_labels)} total "
          f"| Test: {test_labels.sum():.0f} pos / {len(test_labels)} total")
        print(f"--- Finished Fold {fold + 1} ---")
    
    print("---------- Training Details ----------")
    if augmentation_config:
        print(f"Created extra {augmentation_config['n_subs_per_fov']} data points per FOV, with each in size of {augmentation_config['sub_ratio']*100}% of original FOV")
    print(f"Used model {type(model).__name__} for the training process")
    print("Model Architecture:")
    sample_data = Data(x=preprocessed_data[0].x.to(device), edge_index=preprocessed_data[0].edge_index.to(device))
    print(summary(model, sample_data))
    print(f"Used optimizer {type(optimizer).__name__} with learning rate: {lr} and weight decay: {weight_decay}")
    print(f"Using Loss Function: {criterion}")
    print(f"Using Label Smoothing with value of {smoothing}")
    if use_early_stopping:
        print(f"Using validation-based early stopping (patience={early_stopping_patience})")
    if grad_clip_norm is not None:
        print(f"Using gradient clipping with max norm {grad_clip_norm}")
    print(f"Trained the model on up to {epochs} epochs with seed {random_seed}")

    # Per-row patient ID aligned with the concatenated out-of-fold
    # true/pred arrays above (StratifiedGroupKFold with shuffle=True
    # does not preserve the original patient_ids order, so callers must
    # use this instead of the raw patient_ids array for any positional
    # true/pred/patient alignment, e.g. calculate_fov_and_patient_auc).
    fold_results_bundle['all_test_patient_ids'] = np.array(all_out_of_fold_pids)

    return np.array(all_out_of_fold_true), np.array(all_out_of_fold_pred), fold_histories, fold_results_bundle
