import networkx as nx
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import alphashape
from shapely.geometry import Point


def filter_nodes_by_label(G: nx.Graph, label):
    """
    Filters out nodes with a specific label from the graph.

    :param G: The original graph.
    :param label: The label to filter by.
    :return: A tuple of (filtered subgraph, modified original graph).
    """
    nodes_to_remove = [n for n, d in G.nodes(data=True) if d['cell_type'] == label]
    filtered_subgraph = G.subgraph(nodes_to_remove).copy()
    G.remove_nodes_from(nodes_to_remove)
    return filtered_subgraph, G


def identify_clusters(G: nx.Graph):
    """
    Identifies clusters in the given graph using connected components.

    :param G: The subgraph of filtered nodes.
    :return: A list of subgraphs, each representing a cluster.
    """
    return [G.subgraph(c).copy() for c in nx.connected_components(G)]


def calculate_alpha_shapes(clusters, alpha=1.0):
    """
    Calculates the alpha shape for each cluster.

    :param clusters: A list of subgraphs, each representing a cluster.
    :param alpha: The alpha parameter controlling the tightness of the shape.
    :return: A list of vertices representing the alpha shapes.
    """
    alpha_shapes = []
    for cluster in clusters:
        positions = nx.get_node_attributes(cluster, 'pos')
        points = np.array(list(positions.values()))
        if len(points) > 2:  # Alpha shape needs at least 3 points
            shape = alphashape.alphashape(points, alpha)
            alpha_shapes.append(shape)
    return alpha_shapes


def remove_nodes_inside_alpha_shape(G, alpha_shapes):
    """
    Removes nodes from the graph that are inside the given alpha shape.

    :param G: The original graph.
    :param alpha_shapes: A Shapely polygon representing the alpha shape.
    :return: The modified graph with nodes removed.
    """
    positions = nx.get_node_attributes(G, 'pos')
    for alpha_shape in alpha_shapes:
        nodes_to_remove = [node for node, pos in positions.items() if Point(pos).within(alpha_shape)]
        G.remove_nodes_from(nodes_to_remove)
    return G


def transform_shapes_with_buffer(shapes, buffer_value):
    """
    Applies inflation or deflation to a list of Shapely geometric objects
    using the specified buffer value.

    Parameters:
    ----------
    shapes : list
        A list of Shapely geometric objects (Polygon, LineString, Point, etc.).
    buffer_value : float
        The buffer value to be applied. Positive for inflation (expanding the shape),
        negative for deflation (shrinking the shape).

    Returns:
    -------
    list
        A list of transformed shapes where each shape has been inflated or deflated
        based on the buffer_value.

    Example:
    -------
    shapes = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), LineString([(0, 0), (1, 1)])]
    buffer_value = 0.1
    transformed_shapes = transform_shapes_with_buffer(shapes, buffer_value)
    """
    transformed_shapes = []

    for shape in shapes:
        # Apply buffer transformation (inflation or deflation)
        transformed_shape = shape.buffer(buffer_value)
        transformed_shapes.append(transformed_shape)

    return transformed_shapes


def process_graph(G: nx.Graph, label_to_filter, buffer, alpha):
    """
    Main function to process the graph by filtering nodes, calculating convex hulls,
    and removing nodes inside those hulls.

    :param G: The original graph.
    :param label_to_filter: The label to filter by.
    :param buffer: The buffer value for transforming shapes.
    :param alpha: The alpha value for calculating alpha shapes.
    :return: The modified graph after all operations.
    """
    filtered_subgraph, G = filter_nodes_by_label(G, label_to_filter)
    print(f"Filtered out {len(filtered_subgraph.nodes())} nodes with label {label_to_filter}. Remaining nodes: {len(G.nodes())}")
    clusters = identify_clusters(filtered_subgraph)
    print(f"Identified {len(clusters)} clusters of label {label_to_filter}.")
    alpha_shapes = calculate_alpha_shapes(clusters, alpha=alpha)
    print(f"Calculated alpha shapes for {len(alpha_shapes)} clusters.")
    alpha_shapes = transform_shapes_with_buffer(shapes=alpha_shapes, buffer_value=buffer)
    G = remove_nodes_inside_alpha_shape(G, alpha_shapes)
    print(f"Removed nodes inside alpha shapes. Remaining nodes: {len(G.nodes())}")
    return G, clusters