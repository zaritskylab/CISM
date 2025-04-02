import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
import numpy as np
from model import GCN_Model


def train(model, data_dict, train_indices, optimizer, device):
    model.train()
    total_loss = 0
    for idx in train_indices:
        data = data_dict[idx].to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_indices)


def evaluate(model, data_dict, val_indices, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx in val_indices:
            data = data_dict[idx].to(device)
            out = model(data)
            y_true.append(data.y.cpu().float().view(-1, 1))
            y_pred.append(out.cpu().float().view(-1, 1))
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    val_loss = F.binary_cross_entropy(y_pred, y_true)
    return val_loss.item(), y_true.numpy(), y_pred.numpy()


def train_and_validate(data_dict, model, optimizer, train_idx, val_idx, device, epochs=50):
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train(model, data_dict, train_idx, optimizer, device)
        val_loss, y_true, y_pred = evaluate(model, data_dict, val_idx, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model on val examples
    
    return best_model_state


def cross_validation(preprocessed_data, group1, group2, cv_type="3-fold", epochs=500, lr=4e-3, test_size=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ys = [preprocessed_data[i].y for i in preprocessed_data]
    filtered_data = preprocessed_data
    data_indices = list(filtered_data.keys())

    all_true, all_pred = [], []
    train_val_indices, test_indices = train_test_split(data_indices, test_size=test_size, stratify=ys)
    if cv_type == "3-fold":
        kf = StratifiedKFold(n_splits=3)
        splits = kf.split(train_val_indices, [ys[i] for i in train_val_indices])
    else:
        loo = LeaveOneOut()
        splits = loo.split(train_val_indices)

    best_model_state = None
    for train_idx, val_idx in splits:
        train_indices = [train_val_indices[i] for i in train_idx]
        val_indices = [train_val_indices[i] for i in val_idx]

        model = GCN_Model(input_size=preprocessed_data[0].x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        best_model_state = train_and_validate(preprocessed_data, model, optimizer, train_indices, val_indices, device, epochs)

    model.load_state_dict(best_model_state)
    test_loss, y_true, y_pred = evaluate(model, preprocessed_data, test_indices, device)

    return np.array(y_true), np.array(y_pred), model


'''
### Example usage, run after creating a pickle dataset :

with open('mel_alpha_0.01_buffer_0.pickle', 'rb') as p:
    preprocessed_data = pickle.load(p)
group1, group2 = 'NP', 'PN'

# Perform either LOO or 3-fold CV
y_true, y_pred, model = cross_validation(preprocessed_data, group1, group2, cv_type="LOO")

# Calculate and print AUC score on the test set
auc = roc_auc_score(y_true.flatten(), y_pred.flatten())
print(f"AUC Score on Test Set: {auc:.4f}")
'''
