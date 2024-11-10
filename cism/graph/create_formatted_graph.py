import cv2
import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
from cism.graph.plugin_clean_tumor_clusters import process_graph

'''
Originally written by Yuval Tamir (tamiryuv@post.bgu.ac.il)
The code was adjusted by Amos Zamir (zamiramos@gmail.com)
 
Date: 16/12/2023
'''
class GraphBuilder():
    def __init__(self, cells_csv, 
                 common_cells_mapper,
                 colnames_mapper_dict):
            #### sepcific csv col names
            self.cols_mapper = colnames_mapper_dict
            self.cell_type_col = colnames_mapper_dict['cell_types']
            self.patients_col = colnames_mapper_dict['patient_id']
            #### the cells data csv
            self.cells = cells_csv
            #### the mapper between cell-type -> common cell type
            self.cells_mapper = common_cells_mapper
            self.cells['common_cell_types'] = self.cells[self.cell_type_col].apply(lambda x: self.cells_mapper[x])
            #### cell type 2 index
            common_cell_types = np.unique(self.cells['common_cell_types'])
            self.common_cell_type_mapper = {v : k for k,v in enumerate(common_cell_types)}
            
            
            
    def build_graph(self,
                    path_to_output_dir,
                    max_distance,
                    exclude_cell_type: str,
                    removed_cluster_buffer=0,
                    removed_cluster_alpha=0.01):
        from scipy.spatial import distance

        ## for every patient and fov get extract the neighbors
        for point in self.cells[self.patients_col].unique():
            for fov in self.cells['fov'].unique():
                data_p = self.cells[(self.cells[self.patients_col] == point) & (self.cells['fov'] == fov)]
                if len(data_p) < 1:
                    ## FOV and patient mismatch, skipping
                    continue
                coords = [(i, j) for i,j in zip(data_p['centroid-0'], data_p['centroid-1'])]
                points = np.array(coords)
                indptr_neigh, neighbours = Delaunay(points).vertex_neighbor_vertices
                edge = []
                node_ft = []
                for i,row in enumerate(data_p.iterrows()):
                        i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i+1]]
                        node_ft.append(self.common_cell_type_mapper[row[1]['common_cell_types']])
                        for cell in i_neigh:
                            pair = np.array([i, cell])
                            edge.append(pair)
                edges = np.asarray(edge).T
                edge_index = torch.tensor(edges, dtype=torch.long)
                x = torch.tensor(np.array(node_ft).reshape(-1,1), dtype=torch.float)
                left_cell = edge_index.T[:,0].data.tolist()
                right_cell = edge_index.T[:,1].data.tolist()
                input_list = edge_index.T[:,0].data.tolist()
                unique_values = sorted(list(set(input_list)))
                value_map = {unique_values[i]: i for i in range(len(unique_values))}
                output_list = [value_map[value] for value in input_list]
                mapper = {input_list[i] : output_list[i] for i in range(len(input_list))}

                G = nx.Graph()
                for left, right in edges.T:
                    G.add_node(left, pos=coords[left], cell_type=x[mapper[left]])
                    G.add_node(right, pos=coords[right], cell_type=x[mapper[right]])
                    # filter according to upper bound distance limitation
                    if (max_distance is None) or distance.euclidean(coords[left], coords[right]) <= max_distance:
                        G.add_edge(left, right)

                clusters = None
                if exclude_cell_type is not None:
                    G, clusters = process_graph(G,
                                                self.common_cell_type_mapper[exclude_cell_type],
                                                buffer=removed_cluster_buffer,
                                                alpha=removed_cluster_alpha)
                    if len(clusters) > 0:
                        print(f"Found at least one cluster to remove - Patient_{point}_FOV{fov}.txt")


                with open(f"{path_to_output_dir}Patient_{point}_FOV{fov}.txt", "w") as file1:
                    for idx,(left,right) in enumerate(zip(left_cell, right_cell)):
                        if (left not in G.nodes()) or (right not in G.nodes()):
                            continue
                        if (max_distance is None) or distance.euclidean(coords[left], coords[right]) <= max_distance:
                            file1.write(f'{left} {right} {int(x[mapper[left]])} {int(x[mapper[right]])}\n')
        print(self.common_cell_type_mapper)

    def visualize_voronoi(self, patient, fov = None):
        patients_col = self.cols_mapper['patient_id']
        self.data_p = self.cells[(self.cells[patients_col ] == patient) & (self.cells['fov'] == fov)]
        coords = [(i, j) for i,j in zip(self.data_p['centroid-0'], self.data_p['centroid-1'])]
        self.idx2cell_type = {k : v for k,v in enumerate(self.data_p['common_cell_types'])}
        self.points = np.array(coords)
        indptr_neigh, neighbours = Delaunay(self.points).vertex_neighbor_vertices
        vor = Voronoi(self.points)
        fig = voronoi_plot_2d(vor)
        plt.gca().invert_yaxis()
        plt.show()
        plt.clf()
        plt.triplot(self.points[:,0], self.points[:,1], Delaunay(self.points).simplices)
        plt.plot(self.points[:,0], self.points[:,1], 'o')
        for i,idx in enumerate(self.idx2cell_type):
            plt.text(self.points[i,0], self.points[i,1], self.idx2cell_type[idx])
        #plt.savefig('delaunay.png', dpi = 300)
        plt.gca().invert_yaxis()
        plt.show()
        plt.clf()

    def visualize_graph(self):
        indptr_neigh, neighbours = Delaunay(self.points).vertex_neighbor_vertices
        edge = []
        node_ft = []
        for i,row in enumerate(self.data_p.iterrows()):
                i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i+1]]
                node_ft.append(self.common_cell_type_mapper[row[1]['common_cell_types']])
                for cell in i_neigh:
                    pair = np.array([i, cell])
                    edge.append(pair)
        edges = np.asarray(edge).T
        edge_index = torch.tensor(edges, dtype=torch.long)
        x = torch.tensor(np.array(node_ft).reshape(-1,1), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.contiguous())
        g1 = torch_geometric.utils.to_networkx(data,to_undirected=True,node_attrs=['x'])
        COLOR_SCHEME = "Paired"
        nx.draw_networkx(g1, node_color=node_ft, node_size = 100, 
                         with_labels = False, cmap = COLOR_SCHEME)
        ax=plt.gca()
        norm = colors.Normalize(vmin=np.min(node_ft), vmax=np.max(node_ft))
        mappable = cm.ScalarMappable(norm=norm, cmap=COLOR_SCHEME)
        mappable.set_array([])

        nx.draw_networkx(g1, node_color=node_ft, node_size=100, with_labels=False, cmap=COLOR_SCHEME)
        plt.colorbar(mappable)
        plt.axis('off')
        plt.show()