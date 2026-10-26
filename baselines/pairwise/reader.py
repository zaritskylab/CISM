import os

import networkx as nx
import numpy as np
import pandas as pd

from cism.data_preparation import parse_network_filename
from cism.data_preparation.common import NETWORK_FILE_PATTERN
from .common import Columns


class GraphReader:
    @staticmethod
    def read_graph(file_path: str):
        graph = nx.Graph()
        with open(file_path, encoding="utf-8") as handle:
            while True:
                line = handle.readline().strip().split(" ")
                if len(line) < 4:
                    break
                node_left, node_right, node_type_left, node_type_right = line
                graph.add_node(node_left, type=node_type_left)
                graph.add_node(node_right, type=node_type_right)
                graph.add_edge(node_left, node_right)
                if not line:
                    break

        return graph

    @staticmethod
    def extract_pairwise_freq(cells_type: dict, graph: nx.Graph) -> np.ndarray:
        pairwise_counter = np.zeros((len(cells_type), len(cells_type)))
        for u, v in graph.edges():
            u_label = int(graph.nodes[u]["type"])
            v_label = int(graph.nodes[v]["type"])
            pairwise_counter[u_label, v_label] += 1
            pairwise_counter[v_label, u_label] += 1

        return pairwise_counter

    @staticmethod
    def get_graphs(
        full_graph_df: pd.DataFrame,
        raw_data_folder: str,
        raw_data_folder_type: str,
        disease: str,
        cells_type: dict,
    ):
        df = full_graph_df.copy()
        for _, _, files in os.walk(raw_data_folder + raw_data_folder_type):
            for file in files:
                # Only graph files carry patient/FOV metadata; skip patient_class.csv and any
                # other companion file that happens to sit in the dataset folder.
                if not NETWORK_FILE_PATTERN.match(file):
                    continue
                file_path = raw_data_folder + raw_data_folder_type + "/" + file
                temp_df = pd.DataFrame({"Disease": disease, "file": file, "graph": 0}, index=[0], dtype="O")
                temp_df.iloc[0].graph = GraphReader.read_graph(file_path)
                df = pd.concat([df, temp_df], ignore_index=True)

        df["Patient"] = df.file.transform(lambda x: parse_network_filename(x)[0])
        df = df[["Disease", "Patient", "file", "graph"]]
        df[Columns.PAIRWISE_COUNT] = df["graph"].transform(
            lambda graph: GraphReader.extract_pairwise_freq(cells_type=cells_type, graph=graph)
        )
        df["Patient_uId"] = df["Disease"] + df["Patient"]
        return df

    @staticmethod
    def get_normalized_matrix(graph_df: pd.DataFrame, cells_type: dict):
        normalized_matrix = graph_df["pairwise_freq"].sum() / (graph_df["pairwise_freq"].sum().sum())
        return pd.DataFrame(normalized_matrix, index=list(cells_type.values()), columns=list(cells_type.values()))

    @staticmethod
    def get_count_matrix(graph_df: pd.DataFrame, cells_type: dict):
        matrix = graph_df[Columns.PAIRWISE_COUNT].sum()
        return pd.DataFrame(matrix, index=list(cells_type.values()), columns=list(cells_type.values()))