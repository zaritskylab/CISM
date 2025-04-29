import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import Delaunay
import networkx as nx
import alphashape
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")


class CellGraphDataset(Dataset):
    def __init__(self, csv_path, groups, max_distance=100, transform=None, pre_transform=None,
                 cells_to_filter=None, alpha=0.01, buffer_value=0):
        super().__init__(transform, pre_transform)
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['Group'].isin(groups)]
        self.patient_fov_combos = self.df.groupby(['patient number', 'fov', 'Group']).size().reset_index()[
            ['patient number', 'fov', 'Group']]
        self.pred_encoder = OneHotEncoder(sparse=False).fit(self.df[['pred']])
        self.group_encoder = {group: i for i, group in enumerate(self.df['Group'].unique())}
        self.max_distance = max_distance
        self.cells_to_filter = cells_to_filter if cells_to_filter is not None else []
        self.alpha = alpha
        self.buffer_value = buffer_value

    def __len__(self):
        return len(self.patient_fov_combos)

    def __getitem__(self, idx):
        patient = self.patient_fov_combos.iloc[idx]['patient number']
        fov = self.patient_fov_combos.iloc[idx]['fov']
        group = self.patient_fov_combos.iloc[idx]['Group']

        data_p = self.df[(self.df['patient number'] == patient) & (self.df['fov'] == fov)]

        ### Filter out the specified cell types
        if self.cells_to_filter:
            data_p = data_p[~data_p['pred'].isin(self.cells_to_filter)]

        # Create initial graph
        G = nx.Graph()
        for i, row in data_p.iterrows():
            G.add_node(i, pos=(row['centroid-0'], row['centroid-1']), cell_type=row['pred'])
        G = self.process_graph(G, self.alpha, self.buffer_value)

        coords = np.array([G.nodes[n]['pos'] for n in G.nodes()])
        cell_types = [G.nodes[n]['cell_type'] for n in G.nodes()]

        tri = Delaunay(coords)
        indptr_neigh, neighbours = tri.vertex_neighbor_vertices

        edges = []
        for i in range(len(coords)):
            i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i + 1]]
            for j in i_neigh:
                if np.linalg.norm(coords[i] - coords[j]) <= self.max_distance:
                    edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = self.pred_encoder.transform([[ct] for ct in cell_types])
        x = torch.tensor(x, dtype=torch.float)

        y = torch.tensor(self.group_encoder[group], dtype=torch.long)
        pos = torch.tensor(coords, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, pos=pos, y=y)

    def transform_shapes_with_buffer(self, shapes, buffer_value):
        """
        Applies inflation or deflation to a list of Shapely geometric objects.
        """
        transformed_shapes = []
        for shape in shapes:
            transformed_shape = shape.buffer(buffer_value)
            transformed_shapes.append(transformed_shape)
        return transformed_shapes

    def process_graph(self, G, alpha, buffer_value=0):
        clusters = self.identify_clusters(G)
        alpha_shapes = self.calculate_alpha_shapes(clusters, alpha=alpha)
        if buffer_value != 0:
            alpha_shapes = self.transform_shapes_with_buffer(alpha_shapes, buffer_value)
        G = self.remove_nodes_inside_alpha_shape(G, alpha_shapes)
        return G

    def identify_clusters(self, G):
        return [G.subgraph(c).copy() for c in nx.connected_components(G)]

    def calculate_alpha_shapes(self, clusters, alpha=1.0):
        alpha_shapes = []
        for cluster in clusters:
            positions = nx.get_node_attributes(cluster, 'pos')
            points = np.array(list(positions.values()))
            if len(points) > 2:
                shape = alphashape.alphashape(points, alpha)
                alpha_shapes.append(shape)
        return alpha_shapes

    def remove_nodes_inside_alpha_shape(self, G, alpha_shapes):
        positions = nx.get_node_attributes(G, 'pos')
        for alpha_shape in alpha_shapes:
            nodes_to_remove = [node for node, pos in positions.items() if Point(pos).within(alpha_shape)]
            G.remove_nodes_from(nodes_to_remove)
        return G

    def filter_nodes_by_label(self, G, label):
        nodes_to_remove = [n for n, d in G.nodes(data=True) if d['cell_type'] == label]
        filtered_subgraph = G.subgraph(nodes_to_remove).copy()
        G.remove_nodes_from(nodes_to_remove)
        return filtered_subgraph, G


'''
# Example usage
def preprocess_dataset(dataset):
    preprocessed_data = {}
    for i, data in (enumerate(tqdm(dataset, total = len(dataset)))):
        preprocessed_data[i] = data
    return preprocessed_data

groups = ['NP', 'PN']
dataset = CellGraphDataset('cell_type_18_7_2024.csv', groups, max_distance=100, 
                          cells_to_filter=['tumor'], alpha=0.01, buffer_value=0)
preprocessed_data = preprocess_dataset(dataset)
with open(r"mel_alpha_0.01_buffer_0.pickle", "wb") as output_file:
    pickle.dump(preprocessed_data, output_file)
'''
