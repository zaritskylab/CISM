import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def train_batch(model, loader, optimizer, criterion, device, smoothing=0, grad_clip_norm=None):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        # Ensure targets match output shape (batch_size, 1)
        # Note: batch.y is often [batch_size], so we reshape
        target = batch.y.float().view(-1, 1)

        # --- APPLY LABEL SMOOTHING ---
        if smoothing > 0:
            # Formula: new_target = target * (1 - smoothing) + 0.5 * smoothing
            # For 0: 0 * 0.9 + 0.05 = 0.05
            # For 1: 1 * 0.9 + 0.05 = 0.95
            target = target * (1.0 - smoothing) + 0.5 * smoothing

        loss = criterion(out, target)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        # Scale loss by batch size to get accurate epoch average
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def train_fixed_epochs(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs,
    smoothing=0,
    grad_clip_norm=None,
):
    """Train on all supplied data for an explicit number of epochs.

    This is used after inner cross-validation has selected a training
    duration. It intentionally has no validation or checkpoint
    selection because the final outer-fold model must use every
    outer-training patient.

    Returns:
      dict: Training history containing one loss value per epoch.

    Raises:
      ValueError: If epochs is less than one.

    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    history = {"train_loss": []}

    for _ in range(epochs):
        train_loss = train_batch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            smoothing=smoothing,
            grad_clip_norm=grad_clip_norm,
        )
        history["train_loss"].append(train_loss)

    return history

def evaluate_batched(model, loader, criterion, device):
    model.eval()
    y_true, y_pred_probs = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            target = batch.y.float().view(-1, 1)
            
            loss = criterion(out, target)
            total_loss += loss.item() * batch.num_graphs
            
            y_true.append(target.cpu())
            
            y_pred_probs.append(torch.sigmoid(out).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred_probs = torch.cat(y_pred_probs, dim=0).numpy()
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, y_true, y_pred_probs

def train_and_validate_batch(model, train_loader, val_loader, optimizer, criterion, device, epochs=50, smoothing=0.1, loss_debug=False, grad_clip_norm=None):
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(1, epochs + 1):
        train_loss = train_batch(model, train_loader, optimizer, criterion, device, smoothing, grad_clip_norm)
        val_loss, y_true_val, y_pred_val = evaluate_batched(model, val_loader, criterion, device)
        
        unique_classes = np.unique(y_true_val)
        val_auc = roc_auc_score(y_true_val, y_pred_val) if len(unique_classes) >= 2 else 0.5

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        if best_model_state is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if loss_debug:
            print(f"Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC {val_auc:.4f}")
    
    return best_model_state, history

def train_and_validate_batch_early_stopping(model, train_loader, val_loader, optimizer, criterion, device, epochs=50, smoothing=0.1, patience=15, grad_clip_norm=None):
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}

    # Early Stopping Counters
    trigger_times = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_batch(model, train_loader, optimizer, criterion, device, smoothing, grad_clip_norm)
        val_loss, _, _ = evaluate_batched(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Clone to CPU so later epochs don't mutate the saved best state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            trigger_times = 0 # Reset counter
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"🛑 Early Stopping triggered at epoch {epoch}!")
                break

        print(f"Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return best_model_state, history
