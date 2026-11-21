import numpy as np
from sklearn.model_selection import train_test_split


def patient_level_labels(labels, patient_ids, *, context="dataset"):
    """Return unique patients and their labels, rejecting mixed labels."""
    labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids)
    if labels.shape[0] != patient_ids.shape[0]:
        raise ValueError(
            f"{context}: labels and patient_ids must have equal length."
        )

    unique_patients = np.unique(patient_ids)
    patient_labels = []
    for patient_id in unique_patients:
        unique_labels = np.unique(labels[patient_ids == patient_id])
        if unique_labels.size != 1:
            raise ValueError(
                f"{context}: patient {patient_id!r} has inconsistent "
                f"labels {unique_labels.tolist()}."
            )
        patient_labels.append(unique_labels[0])
    return unique_patients, np.asarray(patient_labels)


def validate_stratified_group_kfold(
    labels,
    patient_ids,
    n_splits,
    *,
    context,
):
    """Fail before training if every grouped fold cannot contain each class."""
    _, grouped_labels = patient_level_labels(
        labels, patient_ids, context=context
    )
    classes, counts = np.unique(grouped_labels, return_counts=True)
    if classes.size < 2 or np.any(counts < n_splits):
        distribution = {
            label.item() if hasattr(label, "item") else label: int(count)
            for label, count in zip(classes, counts)
        }
        raise ValueError(
            f"{context}: stratified {n_splits}-fold CV requires at least "
            f"{n_splits} patients in every class; patient counts are "
            f"{distribution}."
        )


def stratified_patient_train_val_indices(
    labels,
    patient_ids,
    validation_patient_ratio,
    *,
    random_state,
    context,
):
    """Split row positions through a stratified patient-level split."""
    labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids)
    unique_patients, grouped_labels = patient_level_labels(
        labels, patient_ids, context=context
    )
    classes, counts = np.unique(grouped_labels, return_counts=True)
    distribution = {
        label.item() if hasattr(label, "item") else label: int(count)
        for label, count in zip(classes, counts)
    }
    if classes.size < 2 or np.any(counts < 2):
        raise ValueError(
            f"{context}: a stratified train/validation split requires at "
            f"least 2 patients in every class; patient counts are "
            f"{distribution}."
        )

    try:
        train_patients, val_patients = train_test_split(
            unique_patients,
            test_size=validation_patient_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=grouped_labels,
        )
    except ValueError as error:
        raise ValueError(
            f"{context}: cannot create a stratified patient validation "
            f"split with validation_patient_ratio="
            f"{validation_patient_ratio}; patient counts are "
            f"{distribution}."
        ) from error

    train_positions = np.flatnonzero(np.isin(patient_ids, train_patients))
    val_positions = np.flatnonzero(np.isin(patient_ids, val_patients))
    return train_positions, val_positions
