import warnings

import networkx as nx
import numpy as np
import pandas as pd
import torch

from graphs_preprocess.graph_process_helper_methods import process_graph
from graphs_preprocess.spatial_edges import build_spatial_edge_pairs

warnings.filterwarnings("ignore")


def build_fov_graph(
    data_p: pd.DataFrame,
    max_distance: int,
    remove_clusters_of_cell_type: str | None,
    alpha: float,
    buffer_value: float,
    plots_output_dir: str | None,
    patient: int,
    fov: int,
    *,
    x_col: str = 'centroid-0',
    y_col: str = 'centroid-1',
    cell_type_col: str = 'pred',
) -> tuple[np.ndarray, list[str], torch.Tensor]:
  """
  Build a Delaunay cell graph for a single patient FOV.

  Constructs a spatial graph from cell positions using Delaunay
  triangulation, optionally removes cluster interiors via alpha-shape
  filtering, and returns the surviving node data.

  Args:
    data_p (pd.DataFrame): Cell data for one patient-FOV with mapped
      coordinate and cell-type columns.
    max_distance (int): Maximum edge distance (each unit is 0.5 in microns).
    remove_clusters_of_cell_type (str | None): Cell type label whose
      spatial clusters are removed via alpha-shape filtering.
    alpha (float): Alpha parameter controlling alpha-shape tightness.
    buffer_value (float): Buffer applied to alpha shapes.
    plots_output_dir (str | None): Directory for before/after plots.
    patient (int): Patient identifier used in log messages and plots.
    fov (int): FOV identifier used in log messages and plots.
    x_col (str): Column containing the first spatial coordinate.
    y_col (str): Column containing the second spatial coordinate.
    cell_type_col (str): Column containing the cell-type label.

  Returns:
    (tuple[np.ndarray, list[str], torch.Tensor]): A tuple of:
      - coords: Float64 array of shape (N, 2) with cell (x, y).
      - cell_types: List of N cell type label strings.
      - edge_index: Long tensor of shape (2, E) with bidirectional
          Delaunay edges among surviving nodes.

  """
  # print(
  #   f"\n===== Processing Patient {patient}, FOV {fov} "
  #   f"with {len(data_p)} cells before filtering. ====="
  # )

  if data_p.empty:
    raise ValueError(
      f"Patient {patient!r}, FOV {fov!r}: no cells are available "
      "for graph construction."
    )

  G = nx.Graph()
  for i, row in data_p.iterrows():
    G.add_node(
      i,
      pos=(row[x_col], row[y_col]),
      cell_type=row[cell_type_col],
    )

  node_list = list(G.nodes())
  coords = np.array([G.nodes[n]['pos'] for n in node_list])
  edge_pairs = build_spatial_edge_pairs(
    coords,
    max_distance,
    context=f"Patient {patient}, FOV {fov}",
  )
  G.add_edges_from(
    (node_list[source], node_list[target])
    for source, target in edge_pairs
  )

  G_before = G.copy()

  if remove_clusters_of_cell_type is not None:
    G, clusters = process_graph(
      G,
      label_to_filter=remove_clusters_of_cell_type,
      buffer=buffer_value,
      alpha=alpha,
    )
    if len(clusters) > 0:
      # Keep optional visualization dependencies out of graph imports.
      from utils.plotting_graphs_utils import plot_before_after

      print(
        f"Found at least one cluster to remove - "
        f"Patient {patient}, FOV {fov}"
      )
      plot_before_after(
        G_before, G, patient, fov,
        remove_clusters_of_cell_type,
        plots_output_dir,
      )

  surviving_nodes = list(G.nodes())
  node_to_idx = {n: i for i, n in enumerate(surviving_nodes)}
  coords = np.array([G.nodes[n]['pos'] for n in surviving_nodes])
  cell_types = [G.nodes[n]['cell_type'] for n in surviving_nodes]

  edges = []
  for u, v in G.edges():
    i, j = node_to_idx[u], node_to_idx[v]
    edges.append([i, j])
    edges.append([j, i])

  if edges:
    edge_index = (
      torch.tensor(edges, dtype=torch.long).t().contiguous()
    )
  else:
    edge_index = torch.zeros((2, 0), dtype=torch.long)

  return coords, cell_types, edge_index
