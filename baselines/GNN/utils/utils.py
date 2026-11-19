import copy
import os
import random
import sys
import torch
import torch_geometric
import numpy as np

from tqdm import tqdm
from utils.aggregation_options import PatientAggregation
from utils.metric_utils import safe_roc_auc


def print_job_execution_details(
    random_seed: int | None = None,
    permutation_seed: int | None = None,
) -> str:
    print(f"\n---------- Execution Details ----------")

    # Get the SLURM Job ID
    job_id = os.environ.get('SLURM_JOB_ID', 'Not a SLURM job')
    print(f"SLURM_JOBID: {job_id}")
    print(f"Node: {os.environ.get('SLURM_JOB_NODELIST', 'local')}")
    # Checking directory before local imports
    print(f"working from directory: {os.getcwd()}")

    if random_seed is not None:
        # Receving the random seed set in the sbatch file, and checking it is consistent with the one used in the code
        print(f"Random seed from environment: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
        print(f"Random seed from code: {random_seed}")

        if os.environ.get('PYTHONHASHSEED') and int(os.environ.get('PYTHONHASHSEED')) != random_seed:
            sys.exit(f"❌ ERROR: Random seed mismatch! Environment: {os.environ.get('PYTHONHASHSEED')}, Code: {random_seed}")

    if permutation_seed is not None:
        print(f"Permutation seed from code: {permutation_seed}")

    return job_id


def set_seed(seed=42):
    """
    Sets the seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    
    # Ensure deterministic behavior in CuDNN
    # Note: This can slightly slow down training but ensures same results every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Force deterministic scatter operations specifically
    # This affects scatter_add used by SAGEConv aggregation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Seed PyTorch Geometric's internal RNG (covers scatter ops)
    torch_geometric.seed_everything(seed)

    # For newer PyTorch versions dealing with scatter/gather:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    torch.use_deterministic_algorithms(True)
    os.environ["PYTORCH_INTERPRETER_CONSTRAINTS"] = "1"
    
    print(f"✅ Random seed set to: {seed}")

def verify_dataset_label_mapping(dataset, num_samples=3):
    print("\n----- Label Mapping Audit -----")
    mapping = dataset.group_encoder
    print(f"Mapping found: {mapping}")

    for i in range(num_samples):
        random_idx = random.randint(0, len(dataset) - 1)        
        sample_data = dataset[random_idx]
        metadata = dataset.patient_fov_combos.iloc[random_idx]

        actual_group = metadata[dataset.label_col]
        assigned_label = sample_data.y.item()

        print(f"[Sample {i+1}] Index: {random_idx}")
        print(f"Patient: {metadata[dataset.patient_col]} | Group: '{actual_group}'")
        print(f"GNN Target Label: {assigned_label}")

        # Logic Check
        if mapping[actual_group] == assigned_label:
            print(f"✅ PASS: Mapping for '{actual_group}' is correct.")
        else:
            print(f"❌ FAIL: Mapping mismatch for '{actual_group}'!")

    print("----- Audit Complete -----")

def preprocess_dataset_v2(dataset):
    preprocessed_data = {}
    patient_ids = []

    for i, data in (enumerate(tqdm(dataset, total = len(dataset)))):
        preprocessed_data[i] = data
        # Extract the patient number for this specific index from the dataset's internal table
        patient_id = dataset.patient_fov_combos.iloc[i][dataset.patient_col]
        patient_ids.append(patient_id)

    return preprocessed_data, np.array(patient_ids)

def audit_output_distribution(model, data_dict, indices, device, split_name="Test"):
    model.eval()
    logits_nn = [] # Class 0
    logits_np = [] # Class 1
    
    with torch.no_grad():
        for idx in indices:
            data = copy.copy(data_dict[idx]).to(device)
            logit = model(data).item()
            label = data.y.item()
            
            if label == 0: # NN
                logits_nn.append(logit)
            else: # NP
                logits_np.append(logit)
                
    print(f"--- {split_name} Distribution Audit ---")
    if logits_nn:
        print(f"NN (0) Logits: Mean={np.mean(logits_nn):.4f}, Range=[{min(logits_nn):.2f}, {max(logits_nn):.2f}]")
    if logits_np:
        print(f"NP (1) Logits: Mean={np.mean(logits_np):.4f}, Range=[{min(logits_np):.2f}, {max(logits_np):.2f}]")
    
def calculate_fov_and_patient_auc(
    y_true,
    y_pred,
    patient_ids_test,
    agg_method: PatientAggregation = PatientAggregation.MEAN,
):
    # 1. Standard FOV-level AUC (Your current method)
    fov_auc = safe_roc_auc(y_true, y_pred, context="FOV-level evaluation")

    # 2. Patient-level AUC (Aggregating FOVs)
    patient_results = {}
    for i in range(len(patient_ids_test)):
        p_id = patient_ids_test[i]
        if p_id not in patient_results:
            patient_results[p_id] = {'preds': [], 'true': y_true[i]}
        patient_results[p_id]['preds'].append(y_pred[i])

    agg_fn = np.max if agg_method == PatientAggregation.MAX else np.mean
    print(f"Patient-level AUC using '{agg_method.value}' FOV aggregation")

    p_true = []
    p_pred_agg = []
    for p_id, val in patient_results.items():
        p_true.append(val['true'])
        p_pred_agg.append(agg_fn(val['preds'])) # Aggregate probability per patient

    patient_auc = safe_roc_auc(
        p_true,
        p_pred_agg,
        context="Patient-level evaluation",
    )

    return fov_auc, patient_auc

def check_split_distribution(indices, patient_ids, data_dict, split_name="Split"):
    """
    Prints the unique patient count and FOV count for NN and NP classes.
    """
    # 1. Get the group/label for each index
    # We assume NN=0 and NP=1 based on your previous audit
    labels = [data_dict[idx].y.item() for idx in indices]
    patients_in_split = patient_ids[indices]
    
    # 2. Separate by class
    nn_patients = set()
    np_patients = set()
    nn_fov_count = 0
    np_fov_count = 0
    
    for label, p_id in zip(labels, patients_in_split):
        if label == 0: # NN
            nn_patients.add(p_id)
            nn_fov_count += 1
        else: # NP
            np_patients.add(p_id)
            np_fov_count += 1
            
    total_pts = len(nn_patients) + len(np_patients)
    
    print(f"--- 📊 {split_name} Distribution ---")
    print(f"Total Unique Patients: {total_pts}")
    print(f"  [NN]: {len(nn_patients)} patients ({nn_fov_count} FOVs) | {len(nn_patients) / total_pts : %}")
    print(f"  [NP]: {len(np_patients)} patients ({np_fov_count} FOVs) | {len(np_patients) / total_pts : %}")
    
    # Alert if a class is missing (Critical for small folds)
    if len(nn_patients) == 0 or len(np_patients) == 0:
        print(f"⚠️ WARNING: {split_name} is missing a whole class!")
        
    return {"nn_pts": len(nn_patients), "np_pts": len(np_patients)}
