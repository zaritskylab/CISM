# Graph Neural Network for Cell Classification

## Project Overview

This repository implements a Graph Neural Network (GNN) framework for spatial cell classification and analysis. The project focuses on the classification of cell populations from spatial tissue data, using a graph-based approach that captures the spatial relationships between different cell types.

## Features

- **Graph-based cell representation**: Converts cellular spatial data into graph structures where nodes represent cells and edges represent spatial relationships
- **Alpha-shape boundary processing**: Uses alpha shapes to identify and process the boundaries of cell clusters
- **Multiple cross-validation strategies**: Supports both k-fold and leave-one-out cross-validation
- **Performance metrics**: Calculates AUC, PR-AUC, and bootstrap confidence intervals
- **Model comparison**: Supports different GNN architectures (GCN, GraphSAGE, GAT, etc.)
- **Adaptive learning**: Implements early stopping, learning rate scheduling, and class weighting

## Repository Structure

- `gnn.py` - GNN model implementation
- `data_preprocessing.py` - Data preprocessing utilities
- `main.py` - Script for running experiments with command-line arguments
- `gnn_explainer.py` - Script for visualy explaining the results

## Requirements

```
torch
torch_geometric
numpy
pandas
scikit-learn
matplotlib
tqdm
networkx
scipy
alphashape
shapely
```

## Data Preprocessing

The `CellGraphDataset` class in `data_preprocessing.py` handles the conversion of cell data into graph structures. Key functionalities include:

- Loading cell data from CSV files
- Filtering specific cell types
- Creating graphs based on spatial proximity
- Processing graph boundaries using alpha shapes
- Supporting buffer operations for inflation/deflation of cell clusters

### Alpha Shape Processing

The dataset employs alpha shapes to identify the boundaries of cell clusters. The alpha parameter controls the "tightness" of the boundary around the points, with lower values creating tighter fits. The buffer value allows for inflation (positive values) or deflation (negative values) of these boundaries.

### Graph Construction

Graphs are constructed as follows:
1. Each cell becomes a node with its type and spatial coordinates
2. Delaunay triangulation establishes potential connections
3. Edges are created between cells within a maximum distance threshold
4. Connected components are identified as cell clusters
5. Alpha shapes define cluster boundaries
6. Optional filtering removes cells based on type or location

## Training and Evaluation

The framework supports:

- K-fold cross-validation with stratified sampling
- Leave-one-out cross-validation for small datasets
- Class weighting for imbalanced datasets
- Comprehensive performance metrics

### Cross-Validation

Both k-fold and leave-one-out cross-validation are implemented to evaluate model performance:

```python
# Example usage
results = perform_cross_validation(
    preprocessed_data=data,
    cv_type='kfold',  # or 'loo' for leave-one-out
    n_folds=5,
    test_size=0.2,
    val_ratio=0.15,
    epochs=300,
    lr=0.002,
    weight_decay=5e-5,
    hidden_channels=[64, 32, 16],
    dropout_rate=0.25,
    conv_type='gcn',  # 'gcn', 'sage', 'graph', or 'gat'
    residual=True,
    verbose=1
)
```

### Command-Line Usage

The framework can be run from the command line with various parameters:

```bash
python main.py --data_file mel_alpha_0.01_buffer_0_NPNN.pickle \
               --cv_type kfold \
               --n_folds 5 \
               --test_size 0.2 \
               --val_ratio 0.15 \
               --epochs 300 \
               --lr 0.002 \
               --weight_decay 5e-5 \
               --hidden_channels 64,32,16 \
               --dropout_rate 0.25 \
               --conv_type gcn \
               --residual \
               --verbose 1 \
               --group1 NP \
               --group2 NN
```

## Data Preparation Example

```python
# Import the dataset class
from data_preprocessing import CellGraphDataset

# Define groups to analyze
groups = ['NP', 'PN']

# Create dataset with specific preprocessing parameters
dataset = CellGraphDataset(
    'cell_type_data.csv', 
    groups, 
    max_distance=100,  # Maximum distance for edge creation
    cells_to_filter=['tumor'],  # Cell types to exclude
    alpha=0.01,  # Alpha parameter for boundary tightness
    buffer_value=0  # Buffer for inflation/deflation
)

# Preprocess and save the dataset
preprocessed_data = {i: data for i, data in enumerate(dataset)}
with open("processed_data.pickle", "wb") as output_file:
    pickle.dump(preprocessed_data, output_file)
```

## Results

The framework outputs comprehensive results including:
- ROC-AUC and PR-AUC scores
- 95% confidence intervals using bootstrapping
- Class-specific performance metrics
- Fold-specific results for k-fold CV
- Trained model and hyperparameters

Results are saved to a timestamped directory in JSON format for analysis and reproduction.

## Model Selection

The framework supports different GNN architectures:
- **GCN**: Graph Convolutional Networks - good general-purpose architecture
- **GraphSAGE**: Better for inductive learning scenarios
- **GAT**: Graph Attention Networks - uses attention mechanisms
- **Graph**: Standard graph convolution
