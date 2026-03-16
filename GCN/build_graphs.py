"""
Standalone Graph Builder
========================
Builds patient-FOV graphs from CSV and saves:
  - graphs_with_meta.pkl  →  (data_dict, meta_df)

Usage:
    python build_graphs.py
    python build_graphs.py --csv my_data.csv --groups NP NN --alpha 0.01 --output my_graphs.pkl
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import alphashape
import torch

from scipy.spatial import Delaunay
from shapely.geometry import Point
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings('ignore')


def build_graphs(csv_path, groups, max_distance=100, alpha=0.01,
                 buffer_value=0, cells_to_filter=None, output_path='graphs_with_meta.pkl'):

    if cells_to_filter is None:
        cells_to_filter = []

    # --- Load & filter CSV ---
    df = pd.read_csv(csv_path)
    df = df[df['Group'].isin(groups)]
    print(f"Loaded {len(df)} cells from {df['patient number'].nunique()} patients")

    combos = (df.groupby(['patient number', 'fov', 'Group'])
                .size().reset_index()[['patient number', 'fov', 'Group']])
    print(f"Total patient-FOV combos: {len(combos)}")

    # --- Encoders ---
    encoder      = OneHotEncoder(sparse=False).fit(df[['pred']])
    group_encoder = {g: i for i, g in enumerate(sorted(df['Group'].unique()))}
    print(f"Cell types ({len(encoder.categories_[0])}): {list(encoder.categories_[0])}")
    print(f"Group encoding: {group_encoder}")

    # --- Helper: alpha shape ---
    def make_alpha_shape(points, alpha):
        if len(points) > 2:
            return alphashape.alphashape(points, alpha)
        return None

    # --- Helper: remove interior nodes ---
    def remove_interior_nodes(G, shapes, buffer_value):
        shapes = [s.buffer(buffer_value) if buffer_value != 0 else s
                  for s in shapes if s is not None]
        positions = nx.get_node_attributes(G, 'pos')
        to_remove = [n for s in shapes for n, p in positions.items()
                     if s is not None and Point(p).within(s)]
        G.remove_nodes_from(to_remove)
        return G

    # --- Build graphs ---
    data_dict = {}
    meta_rows = []
    skipped   = 0

    for _, row in tqdm(combos.iterrows(), total=len(combos), desc="Building graphs"):
        patient = row['patient number']
        fov     = row['fov']
        group   = row['Group']

        sub = df[(df['patient number'] == patient) & (df['fov'] == fov)]
        if cells_to_filter:
            sub = sub[~sub['pred'].isin(cells_to_filter)]

        if len(sub) < 3:
            skipped += 1
            continue

        # Build graph nodes
        G = nx.Graph()
        for i, r in sub.iterrows():
            G.add_node(i, pos=(r['centroid-0'], r['centroid-1']), cell_type=r['pred'])

        # Remove interior (tumor boundary) nodes via alpha shapes
        clusters = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        shapes   = [make_alpha_shape(
                        np.array(list(nx.get_node_attributes(cl, 'pos').values())), alpha)
                    for cl in clusters]
        G = remove_interior_nodes(G, shapes, buffer_value)

        if len(G.nodes) < 3:
            skipped += 1
            continue

        # Delaunay triangulation → edges within max_distance
        coords     = np.array([G.nodes[n]['pos'] for n in G.nodes()])
        cell_types = [G.nodes[n]['cell_type'] for n in G.nodes()]

        tri    = Delaunay(coords)
        ip, nb = tri.vertex_neighbor_vertices
        edges  = [[i, j] for i in range(len(coords))
                  for j in nb[ip[i]:ip[i+1]]
                  if np.linalg.norm(coords[i] - coords[j]) <= max_distance]

        if len(edges) == 0:
            skipped += 1
            continue

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x          = torch.tensor(encoder.transform([[ct] for ct in cell_types]), dtype=torch.float)
        y          = torch.tensor(group_encoder[group], dtype=torch.long)
        pos        = torch.tensor(coords, dtype=torch.float)

        graph_idx             = len(data_dict)
        data_dict[graph_idx]  = Data(x=x, edge_index=edge_index, pos=pos, y=y)
        meta_rows.append({
            'graph_idx': graph_idx,
            'patient':   patient,
            'fov':       fov,
            'group':     group,
            'label':     group_encoder[group],
        })

    meta_df = pd.DataFrame(meta_rows)

    # --- Summary ---
    print(f"\nBuilt {len(data_dict)} graphs  (skipped {skipped})")
    print(f"Patients: {meta_df['patient'].nunique()}")
    print(f"Groups:\n{meta_df.groupby('group').size().to_string()}")
    print(f"\nFOVs per patient:")
    fov_counts = meta_df.groupby('patient').size()
    print(f"  min={fov_counts.min()}  max={fov_counts.max()}  mean={fov_counts.mean():.1f}")
    print(f"\nNode features: {data_dict[0].x.shape[1]}  (one-hot cell types)")
    print(meta_df.groupby(['group', 'patient']).size().reset_index(name='n_fovs').to_string(index=False))

    # --- Save ---
    with open(output_path, 'wb') as f:
        pickle.dump((data_dict, meta_df), f)
    print(f"\nSaved to {output_path}")

    return data_dict, meta_df


def parse_args():
    parser = argparse.ArgumentParser(description='Build patient-FOV graphs from CSV')
    parser.add_argument('--csv',            type=str,   default='cell_type_18_7_2024.csv')
    parser.add_argument('--groups',         nargs='+',  default=['NP', 'NN'])
    parser.add_argument('--max_distance',   type=float, default=100)
    parser.add_argument('--alpha',          type=float, default=0.01)
    parser.add_argument('--buffer_value',   type=float, default=0)
    parser.add_argument('--cells_to_filter',nargs='*',  default=[])
    parser.add_argument('--output',         type=str,   default='graphs_with_meta.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Building graphs: {args.groups} | alpha={args.alpha} | "
          f"max_dist={args.max_distance} | buffer={args.buffer_value}")
    if args.cells_to_filter:
        print(f"Filtering out cell types: {args.cells_to_filter}")

    build_graphs(
        csv_path        = args.csv,
        groups          = args.groups,
        max_distance    = args.max_distance,
        alpha           = args.alpha,
        buffer_value    = args.buffer_value,
        cells_to_filter = args.cells_to_filter,
        output_path     = args.output,
    )
