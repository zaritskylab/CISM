import pandas as pd
import networkx as nx
import random


# https://notebook.community/arne-cl/discoursekernels/notebooks/efficient-subgraph-enumeration

def extend_subgraph(graph, k, subgraph_nodes, extension_nodes, node):
    """
    This function is the recursively called part of the ``ESU`` algorithm
    in Wernicke (2006).
    """
    if len(subgraph_nodes) == k:
        return graph.subgraph(subgraph_nodes)

    all_subgraphs = []
    while extension_nodes:
        # remove random node w from extension_nodes
        random_extension_node = random.choice(list(extension_nodes))
        extension_nodes.remove(random_extension_node)
        exclusive_neighbors = {neighbor for neighbor in exclusive_neighborhood(graph,
                                                                               random_extension_node,
                                                                               subgraph_nodes)
                               if neighbor > node}
        vbar_extension = extension_nodes | exclusive_neighbors  # union
        extended_subgraph_nodes = subgraph_nodes | {random_extension_node}
        subgraphs = extend_subgraph(graph, k, extended_subgraph_nodes, vbar_extension, node)

        if isinstance(subgraphs, list):
            all_subgraphs.extend(subgraphs)
        else:  # isinstance(subgraphs, nx.Graph)
            all_subgraphs.append(subgraphs)
    return all_subgraphs


def enumerate_all_size_k_subgraphs(graph, k):
    """
    returns all subgraphs of the given graph that have k nodes.
    The algorith is called ``ESU`` in Wernicke (2006).
    """
    assert all(isinstance(node, int) for node in graph.nodes())

    if not 1 <= k <= len(graph):
        return []

    all_subgraphs = []
    for node in graph.nodes():
        extension = {neighbor for neighbor in graph.neighbors(node)
                     if neighbor > node}
        subgraphs = extend_subgraph(graph, k, {node}, extension, node)
        if isinstance(subgraphs, list):
            all_subgraphs.extend(subgraphs)
        else:  # isinstance(subgraphs, nx.Graph)
            all_subgraphs.append(subgraphs)
    return all_subgraphs


def exclusive_neighborhood(graph, node, node_subset):
    """
    given a node v that doesn't belong to the given node subset V',
    returns all nodes that are neighbors of v, but don't belong to
    the node subset V' or its open neighborhood N(V').
    Based on Wernicke (2006).

    WARNING: different results for directed vs. undirected graphs
    """
    assert node not in node_subset
    open_nbh = open_neighborhood(graph, node_subset)

    exclusive_nbh = set()
    for neighbor in graph.neighbors(node):
        if neighbor not in node_subset and neighbor not in open_nbh:
            exclusive_nbh.add(neighbor)
    return exclusive_nbh


def open_neighborhood(graph, node_subset):
    """
    $N(V')$: returns the set of all nodes that are in the graph's node set
    (but not in the given subset) and are adjacent to at least one node
    in the subset. Based on Wernicke (2006).

    WARNING: different results for directed vs. undirected graphs
    """
    open_nbh = set()
    node_set = set(graph.nodes())
    nodes_not_in_subset = node_set - node_subset
    for node_not_in_subset in nodes_not_in_subset:
        if any(neighbor in node_subset
               for neighbor in graph.neighbors(node_not_in_subset)):
            open_nbh.add(node_not_in_subset)
    return open_nbh


def count_pattern(full_graph_df: pd.DataFrame, patient_num: str, motif: nx.Graph):
    total_count = 0
    for row_index, row in full_graph_df.iterrows():
        if row['Patient'] != patient_num:
            continue

        g_nx = row['graph']
        nb_nodes = motif.number_of_nodes()

        nodes_type = []

        for idx, node in motif.nodes(data=True):
            nodes_type.append(node['type'])

        nodes_type = sorted(nodes_type)

        h_nx: nx.Graph = g_nx.copy()
        # removing all the nodes with their edges from the graph h_nx:
        nodes_to_remove = []
        for idx, node in h_nx.nodes(data=True):
            if node['type'] not in nodes_type:
                nodes_to_remove.append(idx)
        h_nx.remove_nodes_from(nodes_to_remove)

        h_nx = nx.convert_node_labels_to_integers(h_nx, label_attribute='old')

        for SG in enumerate_all_size_k_subgraphs(h_nx, nb_nodes):
            if not nx.is_connected(SG):
                continue

            sg_nodes_type = []
            for idx, node in SG.nodes(data=True):
                sg_nodes_type.append(node['type'])

            if nodes_type == sorted(sg_nodes_type):
                total_count += 1

    return total_count
