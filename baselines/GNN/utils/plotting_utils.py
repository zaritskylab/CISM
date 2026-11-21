import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
  roc_auc_score,
  roc_curve,
  confusion_matrix,
  ConfusionMatrixDisplay,
)

from utils.aggregation_options import PatientAggregation


def save_plot_to_job_dir(fig, filename, output_dir):
    """
    Saves a figure to the job-specific folder with a standardized name.
    """
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Memory management for long runs
    print(f"💾 Saved: {save_path}")


def _contrasting_text_color(image, value):
    """Return black or white based on the heatmap cell's rendered color."""
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.5 else "white"


def plot_patient_roc_curve(
    y_true,
    y_pred,
    patient_ids,
    cv_type,
    plot_prefix,
    output_dir,
    agg_method: PatientAggregation = PatientAggregation.MEAN,
):
    """
    Groups FOV predictions by Patient ID, aggregates them, and plots the
    ROC curve.
    """
    # 1. Aggregate results by patient
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'y_true': y_true.flatten(),
        'y_pred': y_pred.flatten()
    })

    # Keep the ground truth and aggregate the model's confidence across
    # all FOVs of a patient, matching calculate_fov_and_patient_auc.
    pred_agg = 'max' if agg_method == PatientAggregation.MAX else 'mean'
    patient_results = results_df.groupby('patient_id').agg({
        'y_true': 'first',  # Ground truth is the same for all FOVs of a patient
        'y_pred': pred_agg,
    })

    y_true_patient = patient_results['y_true'].values
    y_pred_patient = patient_results['y_pred'].values

    # 2. Calculate ROC Metrics
    fpr, tpr, _ = roc_curve(y_true_patient, y_pred_patient)
    auc = roc_auc_score(y_true_patient, y_pred_patient)

    # 3. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='forestgreen', lw=2, label=f'Patient ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Patient-Level ROC Curve - {cv_type} CV')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # 4. Save using your helper
    save_plot_to_job_dir(plt.gcf(), f'{plot_prefix}_patient_auc', output_dir)
    
    return auc # Useful for logging

def plot_patient_confusion_matrix(y_true, y_pred, patient_ids, groups, plot_prefix, output_dir, threshold=0.5):
    """
    Aggregates FOV predictions by patient using the mean probability,
    then plots a patient-level confusion matrix.
    """
    patient_ids = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids)
    patient_y_true = []
    patient_y_pred_probs = []

    # 1. Aggregate FOVs per Patient
    for pid in unique_patients:
        indices = np.where(patient_ids == pid)[0]
        # Taking the mean of probabilities for all FOVs of this patient
        avg_prob = np.mean(y_pred[indices])
        # Ground truth should be the same for all FOVs of a patient
        actual_label = y_true[indices][0] 
        patient_y_pred_probs.append(avg_prob)
        patient_y_true.append(actual_label)

    patient_y_true = np.array(patient_y_true)
    patient_y_pred_probs = np.array(patient_y_pred_probs)
    
    # 2. Apply threshold to aggregated probabilities
    y_pred_binary = (patient_y_pred_probs >= threshold).astype(int)
    
    # 3. Generate the matrix
    cm = confusion_matrix(patient_y_true, y_pred_binary)
    
    # 4. Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Confusion Matrix Metrics (Patient) - Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

    # 5. Plotting (reusing your custom styling)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=groups)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

    for text in ax.texts:
        text.remove()

    for i in range(cm.shape[0]):
        row_total = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = (count / row_total) * 100 if row_total > 0 else 0
            color = _contrasting_text_color(disp.im_, count)
            
            ax.text(j, i - 0.08, f"{percentage:.1f}%", 
                    ha="center", va="center", 
                    fontsize=16, fontweight='bold', color=color)
            
            ax.text(j, i + 0.15, f"n={count}", 
                    ha="center", va="center", 
                    fontsize=11, color=color)
    
    title_str = (f"Patient-Level Confusion Matrix (N={len(unique_patients)})\n"
                 f"Acc: {accuracy:.2f} | Sens: {sensitivity:.2f} | Spec: {specificity:.2f}")
    
    plt.title(title_str, fontsize=14, pad=20)
    plt.tight_layout()
    #plt.savefig(f'{plot_prefix}_patient_confusion_matrix.png')
    save_plot_to_job_dir(fig, f'{plot_prefix}_patient_confusion_matrix.png', output_dir)

def _stack_ragged_histories(sequences):
    """
    Stack loss-history lists of possibly different lengths into one
    2D array by right-padding shorter ones with their own last value.

    Needed because early stopping makes folds stop at different
    epochs, so their history lists are no longer all the same length.
    """
    max_len = max(len(seq) for seq in sequences)
    return np.array([
        list(seq) + [seq[-1]] * (max_len - len(seq))
        for seq in sequences
    ])

def plot_cv_losses(fold_histories, plot_prefix, output_dir):
    num_folds = len(fold_histories)
    # Create a grid: 2 rows, 3 columns (for up to 5 folds + 1 mean plot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    all_train_losses = []
    all_val_losses = []

    for i, hist in enumerate(fold_histories):
        ax = axes[i]
        epochs = range(1, len(hist['train_loss']) + 1)
        
        # Plot Training Loss
        ax.plot(epochs, hist['train_loss'], label='Train Loss', color='royalblue', lw=2)
        all_train_losses.append(hist['train_loss'])
        
        # Plot Validation Loss if available
        if 'val_loss' in hist:
            ax.plot(epochs, hist['val_loss'], label='Val Loss', color='darkorange', linestyle='--', lw=2)
            all_val_losses.append(hist['val_loss'])
            
            # Highlight the best epoch (lowest val loss)
            best_epoch = np.argmin(hist['val_loss'])
            ax.scatter(best_epoch + 1, hist['val_loss'][best_epoch], color='red', zorder=5, label='Best State')

        ax.set_title(f'Fold {i+1}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('BCE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- Create the Mean/Average Plot in the last slot ---
    ax_mean = axes[5] if num_folds <= 5 else axes[-1]
    
    avg_train = np.mean(_stack_ragged_histories(all_train_losses), axis=0)
    ax_mean.plot(range(1, len(avg_train) + 1), avg_train, color='royalblue', lw=2, label='Mean Train')

    if all_val_losses:
        avg_val = np.mean(_stack_ragged_histories(all_val_losses), axis=0)
        ax_mean.plot(range(1, len(avg_val) + 1), avg_val, color='darkorange', linestyle='--', lw=2, label='Mean Val')

    ax_mean.set_title('Average Performance Across All Folds')
    ax_mean.set_xlabel('Epochs')
    ax_mean.set_ylabel('BCE Loss')
    ax_mean.legend()
    ax_mean.grid(True, alpha=0.3)

    # Remove any extra empty subplots if they exist
    for j in range(i + 1, len(axes)):
        if j != 5: # Keep the mean plot if it's in the 6th slot
            fig.delaxes(axes[j])

    plt.tight_layout()
    # plt.savefig(f'{plot_prefix}_losses_curve.png')
    save_plot_to_job_dir(fig, f'{plot_prefix}_losses_curve.png', output_dir)

def plot_nested_cv_losses(fold_histories, plot_prefix, output_dir):
    """
    Plots the average training and validation loss for each outer fold
    across its nested inner folds.
    """
    num_outer_folds = len(fold_histories)
    fig, axes = plt.subplots(num_outer_folds, 1, figsize=(10, 5 * num_outer_folds), sharex=True)
    
    # Handle the case of only 1 fold (axes won't be a list)
    if num_outer_folds == 1:
        axes = [axes]

    for i, history in enumerate(fold_histories):
        # history['train_loss'] is a list of lists (one list per inner
        # fold, possibly of different lengths under early stopping).
        train_folds = _stack_ragged_histories(history['train_loss'])
        val_folds = _stack_ragged_histories(history['val_loss'])
        
        avg_train = np.mean(train_folds, axis=0)
        avg_val = np.mean(val_folds, axis=0)
        epochs = range(1, len(avg_train) + 1)

        axes[i].plot(epochs, avg_train, label='Avg Inner Train Loss', color='blue', lw=2)
        axes[i].plot(epochs, avg_val, label='Avg Inner Val Loss', color='red', linestyle='--', lw=2)
        
        # Optional: Plot a shaded area for the variance between inner folds
        std_val = np.std(val_folds, axis=0)
        axes[i].fill_between(epochs, avg_val - std_val, avg_val + std_val, color='red', alpha=0.1)

        axes[i].set_title(f'Outer Fold {i+1}: Inner CV Consensus')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout()
    # plt.savefig(f'{plot_prefix}_losses_curve.png')    
    save_plot_to_job_dir(fig, f'{plot_prefix}_losses_curve.png', output_dir)

def plot_prediction_probabilities(y_pred, output_dir):
    print(f"y_pred range: {y_pred.min():.3f} – {y_pred.max():.3f}")
    print(f"y_pred mean:  {y_pred.mean():.3f}")
    plt.hist(y_pred, bins=20)
    plt.title("Distribution of predicted probabilities")
    save_plot_to_job_dir(plt.gcf(), "pred_distribution", output_dir)
