# ---------- Package Imports ----------
import random
import torch
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph

# ---------- Local Imports ----------
from utils.utils import set_seed


def build_adjacency_matrix(data):
    edge_index = data.edge_index.cpu().numpy() if data.edge_index.is_cuda else data.edge_index.numpy()
    adjacency_matrix = defaultdict(list)

    for s, t in zip(edge_index[0], edge_index[1]): 
        adjacency_matrix[s].append(t)

    return adjacency_matrix

def fast_bfs_subgraph(data, adjacency_matrix, sub_ratio):
    n = data.x.size(0)
    # Calculate dynamic size based on the current FOV
    target_size = int(n * sub_ratio)

    # Safety check: if the graph is extremely small, don't reduce it
    if target_size < 100 or n <= target_size:
        return Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            pos=data.pos.clone(),
            y=data.y.clone(),
        )
    
    seed = random.randint(0, n-1); 
    visited = [seed]; 
    vs = {seed}; 
    head = 0

    while len(visited) < target_size and head < len(visited):
        for nb in adjacency_matrix.get(visited[head], []):
            if nb not in vs: 
                vs.add(nb); 
                visited.append(nb)
            if len(visited) >= target_size: 
                break
        head += 1

    if len(visited) < target_size:
        rem = [i for i in range(n) if i not in vs]; random.shuffle(rem)
        visited.extend(rem[:target_size - len(visited)])
    
    idx = sorted(visited[:target_size])
    idx_t = torch.tensor(idx, dtype=torch.long)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx_t] = True
    ei_cpu = data.edge_index.cpu() if data.edge_index.is_cuda else data.edge_index
    sub_ei, _ = pyg_subgraph(mask, ei_cpu, relabel_nodes=True, num_nodes=n)

    return Data(
        x=data.x[idx_t],
        edge_index=sub_ei,
        pos=data.pos[idx_t],
        y=data.y,
    )

def presample(data_dict, augmentation_config, seed=42):
    """Return each original graph followed by its sampled subgraphs.

    Augmentation expands the training set; it must not replace the full-FOV
    examples that the model is intended to learn from.  Keeping this invariant
    here makes it apply consistently to every CV training path.
    """
    set_seed(seed)
    adjs = {i: build_adjacency_matrix(data_dict[i]) for i in data_dict.keys()}
    return {
        i: [
            data_dict[i],
            *[
                fast_bfs_subgraph(
                    data_dict[i],
                    adjs[i],
                    augmentation_config['sub_ratio'],
                )
                for _ in range(augmentation_config['n_subs_per_fov'])
            ],
        ]
        for i in data_dict.keys()
    }
