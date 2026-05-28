import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc
from sklearn.utils import resample
import numpy as np
from tqdm import tqdm
import pickle
import random
import os
import json
import argparse
from datetime import datetime
from gnn import Enhanced_GNN


# Set seeds for reproducibility across all libraries
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(model, data_dict, train_indices, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    
    # Random shuffle of training indices for better training stability
    train_indices = random.sample(train_indices, len(train_indices))
    
    for idx in train_indices:
        data = data_dict[idx].to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Use weighted loss if class weights are provided
        if class_weights is not None:
            weight = class_weights[int(data.y.item())]
            loss = F.binary_cross_entropy(out, data.y.float().view(-1, 1), 
                                        weight=torch.tensor([weight]).to(device))
        else:
            loss = F.binary_cross_entropy(out, data.y.float().view(-1, 1))
        
        # Add L1 regularization to encourage sparsity
        l1_lambda = 5e-6  # Lower L1 regularization to avoid underfitting
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
            
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_indices)

def evaluate(model, data_dict, val_indices, device, class_weights=None):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    
    with torch.no_grad():
        for idx in val_indices:
            data = data_dict[idx].to(device)
            out = model(data)
            y_true.append(data.y.cpu().float().view(-1, 1))
            y_pred.append(out.cpu().float().view(-1, 1))
            if class_weights is not None:
                weight = class_weights[int(data.y.item())]
                loss = F.binary_cross_entropy(out, data.y.float().view(-1, 1), 
                                             weight=torch.tensor([weight]).to(device))
            else:
                loss = F.binary_cross_entropy(out, data.y.float().view(-1, 1))
            total_loss += loss.item()
            
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    val_loss = total_loss / len(val_indices)
    
    return val_loss, y_true, y_pred

def train_and_validate(data_dict, model, optimizer, train_idx, val_idx, device, scheduler=None, 
                       class_weights=None, patience=20, epochs=300, verbose=1):
    best_val_loss = float('inf')
    best_val_auc = 0
    best_model_state = None
    counter = 0  
    last_improvement = 0
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(1, epochs + 1):
        # Train the model
        train_loss = train(model, data_dict, train_idx, optimizer, device, class_weights)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, y_true, y_pred = evaluate(model, data_dict, val_idx, device, class_weights)
        val_losses.append(val_loss)
        
        # Calculate AUC score for validation
        if len(np.unique(y_true)) > 1:  # Make sure we have both classes
            val_auc = roc_auc_score(y_true.flatten(), y_pred.flatten())
        else:
            val_auc = 0.5
        
        val_aucs.append(val_auc)
        #if verbose > 0 and (epoch % 20 == 0 or epoch == 1):
        #    print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val AUC = {val_auc:.4f}')
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None
            }
            counter = 0  # Reset early stopping counter
            last_improvement = epoch
        else:
            counter += 1
            
        # Restart from best model if no improvement for a while
        if epoch - last_improvement > patience // 2 and epoch < epochs - patience:
            if verbose > 0:
                print(f"No improvement for {epoch - last_improvement} epochs, loading best model and reducing LR")
            model.load_state_dict(best_model_state['model'])
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            counter = 0  # Reset counter after loading best model
        
        if counter >= patience and epoch > epochs // 2:
            if verbose > 0:
                print(f'Early stopping at epoch {epoch}')
            break
    
    # Return training history along with best model
    return best_model_state, best_val_auc, best_val_loss, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs
    }

def perform_cross_validation(preprocessed_data, cv_type='kfold', n_folds=3, test_size=0.2, 
                             val_ratio=0.15, epochs=300, lr=0.002, weight_decay=5e-5, 
                             hidden_channels=[64, 32, 16], dropout_rate=0.25, conv_type='gcn', 
                             residual=True, verbose=1, group1='NP', group2='NN'):
    """
    Perform cross-validation using either k-fold CV or Leave-One-Out CV
    
    Args:
        preprocessed_data: Dictionary of graph data
        cv_type: 'kfold' or 'loo' for Leave-One-Out
        n_folds: Number of folds for k-fold CV (ignored if cv_type='loo')
        test_size: Test set size for k-fold CV (ignored if cv_type='loo')
        val_ratio: Validation set size relative to training data
        epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization weight
        hidden_channels: List of hidden layer sizes
        dropout_rate: Dropout rate
        conv_type: GNN convolution type ('gcn', 'sage', 'graph', 'gat')
        residual: Whether to use residual connections
        verbose: Verbosity level (0, 1, 2)
        group1, group2: Group labels for data
        
    Returns:
        Dictionary with results
    """
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{cv_type}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save hyperparameters
    hyperparams = {
        'cv_type': cv_type,
        'n_folds': n_folds if cv_type == 'kfold' else 'N/A',
        'test_size': test_size if cv_type == 'kfold' else 'N/A',
        'val_ratio': val_ratio,
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'hidden_channels': hidden_channels,
        'dropout_rate': dropout_rate,
        'conv_type': conv_type,
        'residual': residual
    }
    
    with open(f"{results_dir}/hyperparams.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Set device consistently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose > 0:
        print(f"Using device: {device}")
    
    # Extract all data indices and labels
    data_indices = list(preprocessed_data.keys())
    ys = [preprocessed_data[i].y.item() for i in data_indices]
    
    # Check class balance
    class_counts = {0: ys.count(0), 1: ys.count(1)}
    if verbose > 0:
        print(f"Class distribution: {class_counts}")
        print(f"Total samples: {len(ys)}")
    
    # Calculate class weights for imbalanced data
    total_samples = len(ys)
    minority_class = 0 if class_counts[0] < class_counts[1] else 1
    majority_class = 1 if minority_class == 0 else 0
    
    class_weights = {
        minority_class: (total_samples / (2 * class_counts[minority_class])) * 1.1,  # Boost minority class
        majority_class: (total_samples / (2 * class_counts[majority_class])) * 0.9   # Reduce majority class
    }
    if verbose > 0:
        print(f"Class weights: {class_weights}")
    
    # Storage for predictions and metrics
    all_y_true = []
    all_y_pred = []
    all_indices = []
    all_fold_aucs = []
    
    # Get input dimension
    input_dim = preprocessed_data[list(preprocessed_data.keys())[0]].x.shape[1]
    
    # Set up cross-validation
    if cv_type == 'loo':
        # Leave-One-Out CV
        cv = LeaveOneOut()
        splits = list(cv.split(data_indices))
        if verbose > 0:
            print(f"Starting Leave-One-Out Cross-Validation with {len(splits)} splits")
    else:
        # Stratified K-Fold CV
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(cv.split(data_indices, ys))
        if verbose > 0:
            print(f"Starting {n_folds}-fold Cross-Validation")
    
    # Track best model for final evaluation
    best_model_state = None
    best_val_auc = 0
    
    # Use tqdm for progress bar if verbose
    fold_iterator = tqdm(enumerate(splits), total=len(splits), desc=f"{cv_type} Folds") if verbose > 0 else enumerate(splits)
    
    # Iterate through folds
    for fold_idx, (train_val_idx, test_idx) in fold_iterator:
        # Reset seed for each fold to ensure determinism
        set_seed(42 + fold_idx)
        
        # Get training/validation and test indices
        if cv_type == 'loo':
            # For LOO, test is a single sample
            test_indices = [data_indices[i] for i in test_idx]
        else:
            # For k-fold, test is a proportion of data
            train_val_idx, test_indices_temp = train_test_split(
                [data_indices[i] for i in train_val_idx],
                test_size=test_size,
                stratify=[ys[data_indices.index(i)] for i in [data_indices[j] for j in train_val_idx]],
                random_state=42 + fold_idx
            )
            test_indices = test_indices_temp
        
        # Split remaining data into train and validation
        train_val_ys = [ys[data_indices.index(i)] for i in train_val_idx]
        
        # Make sure both classes are represented in validation
        train_indices, val_indices = train_test_split(
            [data_indices[i] for i in train_val_idx], 
            test_size=val_ratio, 
            stratify=train_val_ys, 
            random_state=42 + fold_idx
        )
        
        # Create model
        model = Enhanced_GNN(
            input_size=input_dim,
            hidden_channels=hidden_channels,
            dropout_rate=dropout_rate,
            conv_type=conv_type,
            residual=residual
        ).to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Train and validate
        fold_best_state, fold_val_auc, _, _ = train_and_validate(
            preprocessed_data, model, optimizer, train_indices, val_indices, 
            device, scheduler, class_weights, patience=20, epochs=epochs,
            verbose=(verbose > 1)  # Only show detailed training progress if verbose > 1
        )
        
        # Keep track of the best model across all folds
        if fold_val_auc > best_val_auc:
            best_val_auc = fold_val_auc
            best_model_state = fold_best_state['model']
        
        # Load best model for this fold
        model.load_state_dict(fold_best_state['model'])
        
        # Evaluate on test set
        _, test_y_true, test_y_pred = evaluate(model, preprocessed_data, test_indices, device)
        
        # Calculate fold test AUC
        if len(np.unique(test_y_true)) > 1:
            fold_test_auc = roc_auc_score(test_y_true.flatten(), test_y_pred.flatten())
            all_fold_aucs.append(fold_test_auc)
        else:
            fold_test_auc = 0.5  # Default if only one class
            if verbose > 0:
                print(f"Warning: Fold {fold_idx+1} test set has only one class, AUC set to 0.5")
        
        # Save results for this fold
        all_y_true.append(test_y_true)
        all_y_pred.append(test_y_pred)
        all_indices.extend(test_indices)
        
        # Save individual fold predictions
        fold_result = {
            'fold': fold_idx,
            'test_indices': test_indices,
            'y_true': test_y_true.flatten().tolist(),
            'y_pred': test_y_pred.flatten().tolist(),
            'test_auc': float(fold_test_auc),
            'val_auc': float(fold_val_auc)
        }
        
        with open(f"{results_dir}/fold_{fold_idx}.json", 'w') as f:
            json.dump(fold_result, f, indent=2)
        
        # Report fold progress if verbose
        if verbose > 0:
            print(f"Fold {fold_idx+1} AUC: {fold_test_auc:.4f}")
    
    # Combine all test predictions and ground truths
    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)
    
    # Calculate overall metrics
    final_auc = roc_auc_score(all_y_true.flatten(), all_y_pred.flatten())
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(all_y_true.flatten(), all_y_pred.flatten())
    pr_auc_score = pr_auc(recall, precision)
    
    if verbose > 0:
        print(f"\nFinal Cross-Validation Results:")
        print(f"AUC Score: {final_auc:.4f}")
        print(f"PR-AUC Score: {pr_auc_score:.4f}")
        if cv_type == 'kfold':
            print(f"Fold AUCs: {[f'{auc:.4f}' for auc in all_fold_aucs]}")
            print(f"Mean Fold AUC: {np.mean(all_fold_aucs):.4f} ± {np.std(all_fold_aucs):.4f}")
    
    # Create a final model with the best parameters
    final_model = Enhanced_GNN(
        input_size=input_dim,
        hidden_channels=hidden_channels,
        dropout_rate=dropout_rate,
        conv_type=conv_type,
        residual=residual
    ).to(device)
    
    # Load the best model state
    final_model.load_state_dict(best_model_state)
    
    # Class-specific performance
    class_0_indices = [i for i, y in enumerate(all_y_true.flatten()) if y == 0]
    class_1_indices = [i for i, y in enumerate(all_y_true.flatten()) if y == 1]
    
    class_0_preds = all_y_pred.flatten()[class_0_indices]
    class_1_preds = all_y_pred.flatten()[class_1_indices]
    
    if verbose > 0:
        print(f"Class 0 average prediction: {np.mean(class_0_preds):.4f} ± {np.std(class_0_preds):.4f}")
        print(f"Class 1 average prediction: {np.mean(class_1_preds):.4f} ± {np.std(class_1_preds):.4f}")
    
    # Calculate confidence intervals using bootstrapping
    n_bootstraps = 1000
    bootstrap_aucs = []
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = resample(range(len(all_y_true)), replace=True, n_samples=len(all_y_true))
        bootstrap_y_true = all_y_true[indices]
        bootstrap_y_pred = all_y_pred[indices]
        
        if len(np.unique(bootstrap_y_true)) < 2:
            # Skip bootstraps that don't have both classes
            continue
            
        bootstrap_auc = roc_auc_score(bootstrap_y_true, bootstrap_y_pred)
        bootstrap_aucs.append(bootstrap_auc)
    
    # Calculate 95% confidence interval
    bootstrap_aucs.sort()
    lower_ci = bootstrap_aucs[int(0.025 * len(bootstrap_aucs))]
    upper_ci = bootstrap_aucs[int(0.975 * len(bootstrap_aucs))]
    
    if verbose > 0:
        print(f"Bootstrap 95% CI for AUC: [{lower_ci:.4f}, {upper_ci:.4f}]")
    
    # Save final results
    final_results = {
        'cv_type': cv_type,
        'test_indices': all_indices,
        'auc': float(final_auc),
        'pr_auc': float(pr_auc_score),
        'bootstrap_ci': [float(lower_ci), float(upper_ci)],
        'class_0_avg': float(np.mean(class_0_preds)),
        'class_0_std': float(np.std(class_0_preds)),
        'class_1_avg': float(np.mean(class_1_preds)),
        'class_1_std': float(np.std(class_1_preds)),
        'fold_aucs': [float(auc) for auc in all_fold_aucs] if cv_type == 'kfold' else None,
        'fold_mean_auc': float(np.mean(all_fold_aucs)) if cv_type == 'kfold' else None,
        'fold_std_auc': float(np.std(all_fold_aucs)) if cv_type == 'kfold' else None,
        'hyperparams': hyperparams
    }
    
    with open(f"{results_dir}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save final model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_dim': input_dim,
        'hyperparams': hyperparams
    }, f"{results_dir}/best_{cv_type}_model.pt")
    
    # Return the results dictionary
    return {
        'auc': final_auc,
        'pr_auc': pr_auc_score,
        'model': final_model,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'fold_aucs': all_fold_aucs if cv_type == 'kfold' else None,
        'results_dir': results_dir
    }

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Graph Neural Network for Classification')
    parser.add_argument('--data_file', type=str, default='mel_alpha_0.01_buffer_0_NPNN.pickle',
                        help='Path to preprocessed data file')
    parser.add_argument('--cv_type', type=str, default='kfold', choices=['kfold', 'loo'],
                        help='Cross-validation type: kfold or loo (leave-one-out)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for k-fold cross-validation')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size for k-fold CV')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--hidden_channels', type=str, default='64,32,16',
                        help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.25,
                        help='Dropout rate')
    parser.add_argument('--conv_type', type=str, default='gcn', choices=['gcn', 'sage', 'graph', 'gat'],
                        help='GNN convolution type')
    parser.add_argument('--residual', action='store_true',
                        help='Use residual connections')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity level: 0=quiet, 1=normal, 2=detailed')
    parser.add_argument('--group1', type=str, default='NP',
                        help='First group label')
    parser.add_argument('--group2', type=str, default='NN',
                        help='Second group label')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    # Convert hidden_channels string to list of ints
    args.hidden_channels = [int(x) for x in args.hidden_channels.split(',')]
    return args

# Main execution function
def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set global seed for reproducibility
    set_seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    with open(args.data_file, 'rb') as p:
        preprocessed_data = pickle.load(p)
    
    print(f"Comparing groups: {args.group1} vs {args.group2}")
    
    # Get data stats
    data_indices = list(preprocessed_data.keys())
    ys = [preprocessed_data[i].y.item() for i in data_indices]
    class_counts = {0: ys.count(0), 1: ys.count(1)}
    print(f"Class distribution: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    print(f"Total samples: {len(ys)}")
    
    # Perform cross-validation
    results = perform_cross_validation(
        preprocessed_data=preprocessed_data,
        cv_type=args.cv_type,
        n_folds=args.n_folds,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        dropout_rate=args.dropout_rate,
        conv_type=args.conv_type,
        residual=args.residual,
        verbose=args.verbose,
        group1=args.group1,
        group2=args.group2
    )
    
    print(f"\nFinal {args.cv_type.upper()} CV Results:")
    print(f"AUC Score: {results['auc']:.4f}")
    print(f"PR-AUC Score: {results['pr_auc']:.4f}")
    if args.cv_type == 'kfold':
        print(f"Fold AUCs: {[f'{auc:.4f}' for auc in results['fold_aucs']]}")
        print(f"Mean Fold AUC: {np.mean(results['fold_aucs']):.4f} ± {np.std(results['fold_aucs']):.4f}")
    print(f"Results saved to {results['results_dir']}")
    
    # Return results for further use or visualization
    return results

# If run as a script
if __name__ == "__main__":
    main()
