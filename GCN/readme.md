# GNN patient-level classification from spatial tissue graphs

A two-layer graph convolutional network (GCN) pipeline for binary patient-level classification using spatial multicellular networks derived from multiplexed imaging data.

## Overview

Each cell in the tissue becomes a graph node with a one-hot cell-type feature vector. Edges are constructed via Delaunay triangulation with a distance threshold. The GCN operates on BFS-sampled subgraphs and aggregates predictions hierarchically (subgraphs → FOV → patient) to produce a patient-level classification score.

## Pipeline

```
┌─────────────────────┐       ┌──────────────────────┐       ┌────────────────┐
│  Cell-level CSV      │──────▶│  build_graphs.py     │──────▶│  graphs.pkl    │
│  (coords, cell type, │       │  Delaunay + α-shape  │       │  (PyG Data +   │
│   patient, FOV, group)│       │  → PyG graphs        │       │   metadata)    │
└─────────────────────┘       └──────────────────────┘       └───────┬────────┘
                                                                     │
                                                                     ▼
                                                             ┌───────────────┐
                                                             │  train.py     │
                                                             │  GCN + CV     │
                                                             │  → AUC        │
                                                             └───────────────┘
```

## Usage

### 1. Build graphs

```bash
python build_graphs.py --csv cell_data.csv --groups NP NN --alpha 0.01 --output graphs_with_meta.pkl
```

**Required CSV columns**: `patient number`, `fov`, `Group`, `pred`, `centroid-0`, `centroid-1`

| Argument | Default | Description |
|---|---|---|
| `--csv` | `cell_type_18_7_2024.csv` | Input CSV path |
| `--groups` | `NP NN` | Group labels for binary classification |
| `--max_distance` | `100` | Max edge length in Delaunay graph |
| `--alpha` | `0.01` | Alpha shape parameter for boundary filtering |
| `--buffer_value` | `0` | Buffer around alpha shapes |
| `--cells_to_filter` | `['Tumor']` | Cell types to exclude |
| `--output` | `graphs_with_meta.pkl` | Output pickle path |

### 2. Train and evaluate

```bash
python train.py --graphs graphs_with_meta.pkl --seeds 100 --folds 3
```

| Argument | Default | Description |
|---|---|---|
| `--graphs` | `graphs_with_meta.pkl` | Path to pickle from step 1 |
| `--seeds` | `100` | Random seeds per fold (ensemble size) |
| `--folds` | `3` | On how many folds to evaluate |
| `--print_every` | `10` | Print progressive results every N seeds |

## Method

The model is a two-layer GCN with an additive residual connection, batch normalization, and dropout. Node representations are aggregated via concatenation of global mean and max pooling, followed by a fully connected classification head.

Since tissue graphs vary in size, BFS-based subgraph sampling is used: starting from a random seed node, neighbors are iteratively added until reaching 900 nodes. 30 subgraphs are extracted per FOV. FOV-level predictions are averaged across subgraphs, and patient-level predictions across FOVs.

Training uses AdamW with binary cross-entropy loss, learning rate reduction on plateau, and early stopping on validation AUC. Evaluation uses repeated 3-fold stratified cross-validation. Each fold is trained with multiple random seeds, and predictions are averaged across seeds to form an ensemble score. Performance is reported as mean AUC ± standard deviation across CV repeats.

## Model architecture

```
Input: one-hot cell type (dim = n_cell_types)
  → GCNConv → BatchNorm → ReLU → Dropout(0.15)
  → GCNConv → BatchNorm → ReLU + residual → Dropout(0.3)
  → [global_mean_pool ∥ global_max_pool]
  → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
```

## Requirements

```
torch
torch_geometric
scikit-learn
numpy
pandas
networkx
alphashape
shapely
scipy
tqdm
```
