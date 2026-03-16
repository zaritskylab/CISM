"""
GNN patient-level classification
=================================
Two-layer GCN with BFS subgraph sampling, multi-seed ensemble,
and repeated 3-fold stratified cross-validation.

Usage:
    python train.py
    python train.py --seeds 10
"""
import argparse, os, pickle, random, warnings, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import RepeatedStratifiedKFold
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph as pyg_subgraph
from tqdm import tqdm
warnings.filterwarnings('ignore')


# ── default config ───────────────────────────────────────────
DEFAULT_CONFIG = {
    'cache_path': 'graphs_with_meta.pkl',
    'hidden': 64,
    'dropout': 0.3,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'epochs': 80,
    'patience': 15,
    'sub_size': 900,
    'n_subs_per_fov': 30,
    'batch_size': 32,
}


# ── helpers ──────────────────────────────────────────────────
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_adjacency(data):
    ei = data.edge_index.cpu().numpy() if data.edge_index.is_cuda else data.edge_index.numpy()
    adj = defaultdict(list)
    for s, t in zip(ei[0], ei[1]):
        adj[s].append(t)
    return adj


def fast_bfs_subgraph(data, adj, size=900):
    n = data.x.size(0)
    if n <= size:
        return Data(x=data.x.clone(), edge_index=data.edge_index.clone(), y=data.y.clone())
    seed = random.randint(0, n - 1)
    visited = [seed]; vs = {seed}; head = 0
    while len(visited) < size and head < len(visited):
        for nb in adj.get(visited[head], []):
            if nb not in vs:
                vs.add(nb); visited.append(nb)
            if len(visited) >= size:
                break
        head += 1
    if len(visited) < size:
        rem = [i for i in range(n) if i not in vs]
        random.shuffle(rem)
        visited.extend(rem[:size - len(visited)])
    idx = sorted(visited[:size])
    idx_t = torch.tensor(idx, dtype=torch.long)
    mask = torch.zeros(n, dtype=torch.bool); mask[idx_t] = True
    ei_cpu = data.edge_index.cpu() if data.edge_index.is_cuda else data.edge_index
    sub_ei, _ = pyg_subgraph(mask, ei_cpu, relabel_nodes=True, num_nodes=n)
    return Data(x=data.x[idx_t], edge_index=sub_ei, y=data.y)


def presample(data_dict, cfg, seed=42):
    """Pre-sample BFS subgraphs for all FOVs."""
    random.seed(seed); np.random.seed(seed)
    adjs = {i: build_adjacency(data_dict[i]) for i in range(len(data_dict))}
    return {
        i: [fast_bfs_subgraph(data_dict[i], adjs[i], cfg['sub_size'])
            for _ in range(cfg['n_subs_per_fov'])]
        for i in range(len(data_dict))
    }


# ── model ────────────────────────────────────────────────────
def _init(mod):
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class GCN_2hop(nn.Module):
    """Two-layer GCN with residual connection, BN, dropout, and mean+max pooling."""
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.b1 = nn.BatchNorm1d(hidden)
        self.b2 = nn.BatchNorm1d(hidden)
        self.dr = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        _init(self)

    def forward(self, x, edge_index, batch):
        h1 = F.relu(self.b1(self.c1(x, edge_index)))
        h1 = F.dropout(h1, self.dr * 0.5, training=self.training)
        h2 = F.relu(self.b2(self.c2(h1, edge_index)))
        h2 = h2 + h1  # residual
        h2 = F.dropout(h2, self.dr, training=self.training)
        pooled = torch.cat([global_mean_pool(h2, batch), global_max_pool(h2, batch)], dim=1)
        return torch.sigmoid(self.head(pooled))


# ── training & evaluation ────────────────────────────────────
@torch.no_grad()
def predict_patient_scores(model, cache, patients, pat_fovs, device):
    """Predict per-patient scores by averaging subgraph → FOV → patient."""
    model.eval()
    out = {}
    for p in patients:
        fov_means = []
        for fi in pat_fovs[p]:
            subs = cache[fi]
            if not subs:
                fov_means.append(0.5); continue
            loader = DataLoader(subs, batch_size=64, shuffle=False)
            preds = []
            for bd in loader:
                bd = bd.to(device)
                preds.extend(model(bd.x, bd.edge_index, bd.batch).view(-1).cpu().tolist())
            fov_means.append(float(np.mean(preds)))
        out[p] = fov_means
    return out


def patient_auc(pat_scores, pat_label):
    yt = [pat_label[p] for p in pat_scores]
    yp = [np.mean(pat_scores[p]) for p in pat_scores]
    return roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5


def train_with_val(model, cache, fit_patients, val_patients,
                   pat_fovs, pat_label, pos_weight, cfg, device):
    """Train with early stopping on validation AUC."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=6, factor=0.5, min_lr=1e-5)
    best_auc = 0.0; best_state = None; no_imp = 0

    # Build training loader
    fit_fovs = [fi for p in fit_patients for fi in pat_fovs[p]]
    ds = []
    for i in fit_fovs:
        ds.extend(cache[i])
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)

    for epoch in range(cfg['epochs']):
        model.train()
        for bd in loader:
            bd = bd.to(device)
            opt.zero_grad()
            pred = model(bd.x, bd.edge_index, bd.batch).view(-1)
            target = bd.y.float().view(-1)
            weight = torch.where(target == 1, pos_weight, torch.ones_like(pos_weight))
            F.binary_cross_entropy(pred, target, weight=weight).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vauc = patient_auc(
            predict_patient_scores(model, cache, val_patients, pat_fovs, device), pat_label)
        sched.step(vauc)

        if vauc > best_auc:
            best_auc = vauc; best_state = deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
        if epoch >= 10 and no_imp >= cfg['patience']:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def bootstrap_ci(yt, yp, n=5000):
    aucs = []
    for _ in range(n):
        idx = resample(range(len(yt)), replace=True)
        a, b = np.array(yt)[idx], np.array(yp)[idx]
        if len(np.unique(a)) > 1:
            aucs.append(roc_auc_score(a, b))
    if len(aucs) < 10:
        return 0.0, 1.0
    aucs.sort()
    return aucs[int(.025 * len(aucs))], aucs[int(.975 * len(aucs))]


# ── main ─────────────────────────────────────────────────────
def main(n_seeds=100, reps=None, print_every=10, config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if reps is None:
        reps = [2, 4, 6]

    print(f"CUDA: {torch.cuda.is_available()}")
    with open(cfg['cache_path'], 'rb') as f:
        data_dict_orig, meta_df = pickle.load(f)

    data_raw = {
        i: Data(x=data_dict_orig[i].x.clone(),
                edge_index=data_dict_orig[i].edge_index.clone(),
                y=data_dict_orig[i].y.clone())
        for i in range(len(data_dict_orig))
    }
    raw_dim = data_raw[0].x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pat_label = meta_df.groupby('patient')['label'].first().to_dict()
    pat_fovs = meta_df.groupby('patient')['graph_idx'].apply(list).to_dict()
    pat_list = list(meta_df['patient'].unique())
    pat_ys = [pat_label[p] for p in pat_list]
    print(f"Patients: {len(pat_list)} | C0: {pat_ys.count(0)} | C1: {pat_ys.count(1)} | Dim: {raw_dim}")

    # CV splits
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
    all_splits = list(rskf.split(pat_list, pat_ys))
    selected_reps = [r - 1 for r in forced_reps]  # 0-indexed

    # Pre-sample subgraphs per seed
    print(f"Pre-sampling subgraphs for {n_seeds} seeds...")
    t_pre = time.time()
    caches = {}
    for s in range(n_seeds):
        caches[s] = presample(data_raw, cfg, seed=42 + s * 1000)
    print(f"Pre-sampling done in {(time.time() - t_pre) / 60:.1f} min")

    # Training loop
    t0 = time.time()
    total_jobs = len(selected_reps) * 3 * n_seeds
    pbar = tqdm(total=total_jobs, desc=f"Training (seeds={n_seeds})")

    all_seed_scores = {s: {} for s in range(n_seeds)}

    for rep_idx in selected_reps:
        for fi in range(3):
            global_fi = rep_idx * 3 + fi
            trm, tem = all_splits[global_fi]
            trp = [pat_list[i] for i in trm]
            tep = [pat_list[i] for i in tem]

            # Fixed val split per fold
            set_seed(42 + global_fi * 100)
            pos = [p for p in trp if pat_label[p] == 1]
            neg = [p for p in trp if pat_label[p] == 0]
            nv = max(1, int(len(trp) * 0.2) // 2)
            vp = random.sample(pos, min(nv, len(pos))) + random.sample(neg, min(nv, len(neg)))
            fp = [p for p in trp if p not in vp]
            fc0 = sum(1 for p in fp if pat_label[p] == 0)
            fc1 = sum(1 for p in fp if pat_label[p] == 1)
            pw = torch.tensor([fc0 / max(fc1, 1)], device=device)

            for s in range(n_seeds):
                set_seed(42 + global_fi * 100 + s)
                model = GCN_2hop(raw_dim, cfg['hidden'], cfg['dropout']).to(device)
                model = train_with_val(
                    model, caches[s], fp, vp, pat_fovs, pat_label, pw, cfg, device)
                scores = predict_patient_scores(model, caches[s], tep, pat_fovs, device)
                all_seed_scores[s][(rep_idx, fi)] = scores
                del model
                torch.cuda.empty_cache()
                pbar.update(1)

    pbar.close()

    # ── compute results ──────────────────────────────────────
    def compute_aucs_for_seeds(seed_range):
        rep_aucs = []
        for rep_idx in selected_reps:
            rep_scores = {}
            for fi in range(3):
                tep_set = set()
                for s in seed_range:
                    tep_set.update(all_seed_scores[s][(rep_idx, fi)].keys())
                for p in tep_set:
                    seed_vals = []
                    for s in seed_range:
                        if p in all_seed_scores[s][(rep_idx, fi)]:
                            seed_vals.append(np.mean(all_seed_scores[s][(rep_idx, fi)][p]))
                    rep_scores[p] = np.mean(seed_vals)
            yt = [pat_label[p] for p in rep_scores]
            yp = [rep_scores[p] for p in rep_scores]
            rep_aucs.append(roc_auc_score(yt, yp))
        return rep_aucs

    # Progressive printing
    if n_seeds >= print_every * 2:
        print(f"\n── Progressive results (seeds={n_seeds}) ──")
        for k in range(print_every, n_seeds + 1, print_every):
            aucs = compute_aucs_for_seeds(range(k))
            mean_a = np.mean(aucs); std_a = np.std(aucs)
            per_rep = "  ".join([f"R{forced_reps[i]}={aucs[i]:.4f}" for i in range(len(aucs))])
            print(f"  Seeds 1-{k:3d}: Mean={mean_a:.4f}±{std_a:.4f}  [{per_rep}]")

    # Final summary
    final_aucs = compute_aucs_for_seeds(range(n_seeds))
    all_pat_scores = {}
    for rep_idx in selected_reps:
        for fi in range(3):
            tep_set = set()
            for s in range(n_seeds):
                tep_set.update(all_seed_scores[s][(rep_idx, fi)].keys())
            for p in tep_set:
                seed_vals = [np.mean(all_seed_scores[s][(rep_idx, fi)][p])
                             for s in range(n_seeds) if p in all_seed_scores[s][(rep_idx, fi)]]
                all_pat_scores[p] = np.mean(seed_vals)
    final_yt_all = [pat_label[p] for p in all_pat_scores]
    final_yp_all = [all_pat_scores[p] for p in all_pat_scores]
    ci = bootstrap_ci(final_yt_all, final_yp_all)

    print(f"\n{'=' * 60}")
    print(f"RESULTS  (3-fold CV × {n_seeds} seeds × {len(forced_reps)} repeats)")
    print(f"{'=' * 60}")
    for ri, rep_idx in enumerate(selected_reps):
        print(f"  Rep {rep_idx + 1:2d}: AUC = {final_aucs[ri]:.4f}")
    print(f"{'─' * 60}")
    print(f"  Mean ± Std : {np.mean(final_aucs):.4f} ± {np.std(final_aucs):.4f}")
    print(f"  Range      : [{min(final_aucs):.4f}, {max(final_aucs):.4f}]")
    print(f"  Pooled 95%CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"{'=' * 60}")
    print(f"Time: {(time.time() - t0) / 60:.1f} min")


def parse_args():
    parser = argparse.ArgumentParser(description='GNN patient-level classification')
    parser.add_argument('--folds', type=int, default=3, help='Number of folds')
    parser.add_argument('--seeds', type=int, default=100, help='Number of random seeds per fold')
    parser.add_argument('--reps', nargs='+', type=int, default=[2, 4, 6],
                        help='Which CV repeats to use (1-indexed)')
    parser.add_argument('--graphs', type=str, default='graphs_with_meta.pkl',
                        help='Path to graphs pickle from build_graphs.py')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print progressive results every N seeds')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Config: seeds={args.seeds} | reps={args.reps} | graphs={args.graphs}")
    main(
        n_folds = args.folds,
        n_seeds=args.seeds,
        reps=args.reps,
        print_every=args.print_every,
        config={'cache_path': args.graphs},
    )
