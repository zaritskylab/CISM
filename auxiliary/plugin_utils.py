import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import Delaunay
from matplotlib.colors import ListedColormap
import copy
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg

motif = {
    ('A', 'B'): {}, 
    ('A', 'C'): {},
    ('B', 'C'): {},
    'A': {'type': 9},
    'B': {'type': 11},
    'C': {'type': 11}
}

def rotate_point(points, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Check if points is a single point or a list of points
    if isinstance(points[0], (int, float)):
        return np.dot(rotation_matrix, points)
    else:
        return [np.dot(rotation_matrix, point) for point in points]

def fig_to_np_array(fig):

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    image_data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    return image_data.reshape(canvas.get_width_height()[::-1] + (3,))

def convert_dict(input_dict):
    new_dict = {}
    for key, value in input_dict.items():
        new_dict[value['name']] = key
    return new_dict
def parse_motif_input(input_str, cell_type_dict):
    # Convert cell type names to indices
    type_to_index = convert_dict(cell_type_dict)

    # Split the input string into separate motifs
    motifs_str = input_str.strip().split("\n\n")
    motifs = []

    for motif_str in motifs_str:
        lines = motif_str.strip().split("\n")
        motif = {}

        for line in lines:
            line = line.strip()
            if "->" in line:
                src, dst = map(str.strip, line.split("->"))
                motif[(src, dst)] = {}
            elif "!>" in line:
                src, dst = map(str.strip, line.split("!>"))
                motif[(src, dst)] = {"not_connected": True}
            elif ".type =" in line:
                node, type_val = map(str.strip, line.split(".type ="))
                type_val = type_val.strip('"')
                # Check if type_val is a digit or needs conversion
                if type_val.isdigit():
                    motif[node] = {"type": int(type_val)}
                elif type_val in type_to_index:
                    motif[node] = {"type": type_to_index[type_val]}
                else:
                    raise ValueError(f"Unknown cell type: {type_val}")

        motifs.append(motif)

    return motifs
def parse_motif_input2(input_str, cell_type_dict):
    # Convert cell type names to indices
    type_to_index = convert_dict(cell_type_dict)

    lines = input_str.strip().split("\n")
    motif = {}
    
    for line in lines:
        line = line.strip()
        if "->" in line:
            src, dst = map(str.strip, line.split("->"))
            motif[(src, dst)] = {}
        elif "!>" in line:
            src, dst = map(str.strip, line.split("!>"))
            motif[(src, dst)] = {"not_connected": True}
        elif ".type =" in line:
            node, type_val = map(str.strip, line.split(".type ="))
            type_val = type_val.strip('"')
            # Check if type_val is a digit or needs conversion
            if type_val.isdigit():
                motif[node] = {"type": int(type_val)}
            elif type_val in type_to_index:
                motif[node] = {"type": type_to_index[type_val]}
            else:
                raise ValueError(f"Unknown cell type: {type_val}")
    
    return motif

def parse_motif_input1(input_str):
    lines = input_str.strip().split("\n")
    motif = {}
    
    for line in lines:
        line = line.strip()
        if "->" in line:
            src, dst = map(str.strip, line.split("->"))
            motif[(src, dst)] = {}
        elif "!>" in line:
            src, dst = map(str.strip, line.split("!>"))
            # You can adjust this part based on how you want to handle non-connections
            # For now, I'm just making a note in the motif dictionary
            motif[(src, dst)] = {"not_connected": True}
        elif ".type =" in line:
            node, type_val = map(str.strip, line.split(".type ="))
            motif[node] = {"type": int(type_val.strip('"'))}
    
    return motif

def create_tagged_image(cell_data, image, mapper_dict, patinet_number, fov = None):
        # Load the CSV data
        if fov is None:
            cell_data_specific_p = cell_data[cell_data[mapper_dict['patints_col']] == patinet_number]
        else:
            cell_data_specific_p = cell_data[(cell_data[mapper_dict['patints_col']] == patinet_number) \
                                                  & (cell_data[mapper_dict['fov_col']] == fov)]

        segmented_image_array = np.array(image)
        # Create a mapping from cell type to integer
        cell_types = cell_data_specific_p[mapper_dict['cell_types_col']].unique()
        cell_type_to_int = {cell_type: idx + 1 for idx, cell_type in enumerate(cell_types)}

        # Create a new image array for cell types
        cell_type_image_array = np.zeros_like(segmented_image_array)
        # Populate the new image array based on cell type mapping
        for _, row in cell_data_specific_p.iterrows():
            cell_label = row[mapper_dict['cell_index_col']]
            cell_type_int = cell_type_to_int[row[mapper_dict['cell_types_col']]]
            cell_type_image_array[segmented_image_array == cell_label] = cell_type_int
        return cell_type_image_array.astype(np.uint8)


def find_motifs(G_full, motifs):
    print(motifs)
    all_subgraphs = []

    for motif in motifs:
        motif_graph = nx.Graph()
        
        # Adding edges and nodes based on motif input
        for key, value in motif.items():
            if isinstance(key, tuple):  # Edges
                src, dst = key
                if "not_connected" not in value:  # Only add if it's a connection
                    motif_graph.add_edge(src, dst)
            else:  # Nodes
                node = key
                if 'type' in value:
                    motif_graph.add_node(node, type=value['type'])

        # Searching for the motif in the main graph
        matcher = nx.algorithms.isomorphism.GraphMatcher(G_full, motif_graph, node_match=node_matcher)
        subgraphs = matcher.subgraph_isomorphisms_iter()

        # Convert subgraphs to a list and add to all_subgraphs
        subgraphs_list = list(subgraphs)
        all_subgraphs.append(subgraphs_list)

    return all_subgraphs




def find_motifs1(G_full, motif):
    motif_graph = nx.Graph()
    
    # Adding edges and nodes based on motif input
    for key, value in motif.items():
        if isinstance(key, tuple):  # Edges
            src, dst = key
            if "not_connected" not in value:  # Only add if it's a connection
                motif_graph.add_edge(src, dst)
        else:  # Nodes
            node = key
            if 'type' in value:
                motif_graph.add_node(node, type=value['type'])

    # Searching for the motif in the main graph
    matcher = nx.algorithms.isomorphism.GraphMatcher(G_full, motif_graph, node_match=node_matcher)
    subgraphs = matcher.subgraph_isomorphisms_iter()

    # Convert subgraphs to a list
    subgraphs_list = list(subgraphs)

    return subgraphs_list


def generate_cell_type_structure_from_tagged(mapper):
        num_unique_types = len(mapper)
        
        colormap = cm.get_cmap('tab20c', num_unique_types)
        colors = [colormap(i) for i in range(num_unique_types)]
        
        # Background and Unknown are hardcoded
        cell_type_structure = {
            0: {'name': 'Background', 'color': 'black'},
            #1: {'name': 'Unknown', 'color': 'black'}
        }
        
        # Populate the dictionary with unique cell types and their colors
        for i, cell_type in enumerate(mapper.keys(), start=1):
            cell_type_structure[cell_type] = {'name': mapper[cell_type], 'color': colors[i-1]}
            
        return cell_type_structure


# Load the image
def build_graph(image, list_of_cells_to_exclude=[]):
    
    image_full = image#[500:1500, 500:1500]
    image_original = copy.deepcopy(image_full)
    # Preprocess the image based on the provided code
    #image_full[image_full == 4] = 17

    #image_full = np.where(image_full > 3, image_full - 1, image_full)
    if len(list_of_cells_to_exclude) > 0:
        image_full = np.where(np.isin(image_full, list_of_cells_to_exclude), 50, image_full)
    exc = 50 #if len(list_of_cells_to_exclude) > None0 else 
    # Extract cell contours and centroid coordinates
    cnts_full = cv2.findContours(image_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_full = cnts_full[0] if len(cnts_full) == 2 else cnts_full[1]
    print(len(cnts_full))
    coords_full = []
    point2cell_full = {}
    p2c = {}
    for idx, c in enumerate(cnts_full):
        ((x, y), r) = cv2.minEnclosingCircle(c)
        coords_full.append([int(x), int(y)])
        if image_full[int(y), int(x)] == exc:
            point2cell_full[idx] = exc
            p2c[idx] = image_original[int(y), int(x)]#0
        else:
            point2cell_full[idx] = image_full[int(y), int(x)]
            p2c[idx] = image_original[int(y), int(x)]
    # Generate the graph representation using Delaunay triangulation
    indptr_neigh_full, neighbours_full = Delaunay(np.array(coords_full)).vertex_neighbor_vertices

    # Generate edges and node features
    edges_full = []
    node_ft_full = []
    
    for i in range(len(coords_full)):
        if point2cell_full[i] == exc:
            continue
        else:
            i_neigh = neighbours_full[indptr_neigh_full[i]:indptr_neigh_full[i+1]]
            node_ft_full.append(point2cell_full[i])
            for cell in i_neigh:
                if point2cell_full[cell] == exc:
                    continue
                pair = np.array([i, cell])
                edges_full.append(pair)
    edges_full = np.asarray(edges_full).T


    G_full = nx.Graph()
    for left, right in edges_full.T:
        # Add nodes with their cell types
        G_full.add_node(left, cell_type=point2cell_full[left])
        G_full.add_node(right, cell_type=point2cell_full[right])
        
        # Add the edge
        G_full.add_edge(left, right)

    # Extract the nodes and edges added to the test graph
    nodes_in_test_graph = list(G_full.nodes(data=True))
    edges_in_test_graph = list(G_full.edges())

    return image_original, G_full, coords_full, point2cell_full, p2c

# Define the node matcher function
def node_matcher(node1, node2):
    return node1['cell_type'] == node2['type']

# Convert the motif dictionary into a graph structure
#def find_motifs(G_full, motif = motif):
#    motif_graph_8 = nx.Graph()
#    motif_graph_8.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C')])
#    for node, attr in motif.items():
#        if isinstance(attr, dict) and 'type' in attr:
#            motif_graph_8.add_node(node, type=attr['type'])

    # Search for the motif in the graph
#    subgraphs_8 = nx.algorithms.isomorphism.GraphMatcher(G_full, motif_graph_8, node_match=node_matcher).subgraph_isomorphisms_iter()

    # Convert subgraphs to a list and filter
#    subgraphs_8_list = list(subgraphs_8)
#    return subgraphs_8_list




def vis_graph_and_motifs(coords_full,
                         subgraphs_8_list,
                         image_original,
                         G_full,
                         point2cell_full,
                         cell_types):
    # Generate Voronoi diagram

    corrected_pos = {node: (coords_full[node][0], coords_full[node][1]) for node in G_full.nodes()}

    # Visualize the corrected graph overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    
    
    # Display the image with Voronoi overlay
    colors = ['black'] + [cell_types[key]['color'] for key in cell_types.keys()]
    custom_cmap = ListedColormap(colors)   
    print(custom_cmap) 
    ax1.imshow(image_original, cmap=custom_cmap, origin='upper')

    colors = []
    for cell_type, attributes in cell_types.items():
        ax2.plot([], [], 'o', color=attributes['color'], label=f"Cell Type {attributes['name']} ({cell_type})", markersize=10)
    ax2.legend(loc="upper left")
    ax2.axis('off')
    #voronoi_plot_2d(vor_full, ax=ax, show_vertices=False, line_colors='black', line_width=0.5, line_alpha=0.6, point_size=2)

    # Display the graph with emphasized motifs
    #node_colors = ['blue' if node not in [item for sublist in subgraphs_8_list for item in sublist.values()] else 'red' for node in G_full.nodes()]

    motif_nodes = [node for subgraph in subgraphs_8_list for node in subgraph.keys()]
    node_colors = ['white' if node in motif_nodes else 'none' for node in G_full.nodes()]
    nx.draw_networkx(G_full, pos=corrected_pos, node_size=5, with_labels=False, node_color=node_colors, edge_color='white', ax=ax1)
    for sg in subgraphs_8_list:
        bb = {v : k for k,v in sg.items()}
        motif_nodes = [bb['A'], bb['B'], bb['C']]
        motif_edges = [(motif_nodes[i], motif_nodes[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
        nx.draw_networkx_edges(G_full, pos=corrected_pos, edgelist=motif_edges, edge_color='red', width=2.5, ax=ax1)

    ax1.set_title('Tissue with Graph Overlay and Emphasized Motifs')
    ax1.axis('off')
    fig.tight_layout()
    return fig


def generate_color_map(cell_types):
    """
    Generate a color map for the provided cell types.
    """
    unique_cell_types = np.unique(cell_types)
    num_colors = len(unique_cell_types)
    
    # Generate a color palette with as many unique colors as there are cell types
    color_palette = plt.cm.tab20.colors + plt.cm.tab20c.colors  # Combine two color palettes to get more unique colors
    colors = color_palette * (num_colors // len(color_palette) + 1)  # Repeat palette if more colors are needed
    colors = colors[:num_colors]
    
    return {cell_type: colors[i] for i, cell_type in enumerate(unique_cell_types)}


def generate_cell_type_structure(cell_types):
    """
    Generate a dictionary structure similar to tnbc_cells_type for the provided cell types.
    """
    unique_cell_types = np.unique(cell_types)
    num_unique_types = len(unique_cell_types)
    
    # Generate a wide range of unique colors using a colormap
    colormap = cm.get_cmap('tab20c', num_unique_types)
    colors = [colormap(i) for i in range(num_unique_types)]
    
    # Background and Unknown are hardcoded
    cell_type_structure = {
        #0: {'name': 'Background', 'color': 'black'},
        #1: {'name': 'Unknown', 'color': 'black'}
    }
    
    # Populate the dictionary with unique cell types and their colors
    for i, cell_type in enumerate(unique_cell_types, start=0):
        cell_type_structure[i] = {'name': cell_type, 'color': colors[i]}
        
    return cell_type_structure


# 2. Define necessary functions
def build_cell_graph(data,
                     tnbc_cells_type,
                     max_distance,
                     exclude=[],
                     filter_out_cell_type_cluster=None,
                     buffer=0,
                     alpha=0.01):

    from scipy.spatial import distance

    """Build a graph from cell coordinates and cell types."""
    coords = data[['centroid-0', 'centroid-1']].values
    cell_types = data['common_cell_types'].values 
    cells_idx = data['cell_id']

    include_indices = [i for i, ctype in enumerate(cell_types) if ctype not in exclude]
    coords_included = coords[include_indices]
    #cell_types = cell_types[include_indices]
    
    idx2cell = {idx: cell_type for idx, cell_type in enumerate(cell_types)}
    cell_type_to_index = {v['name']: k for k, v in tnbc_cells_type.items()}
    
    # Use the above mapping to generate the desired dictionary
    p2c = {cell_idx: cell_type_to_index[cell_type] for cell_idx, cell_type in idx2cell.items()}
    
    points = np.array(coords)
    indptr_neigh, neighbours = Delaunay(points).vertex_neighbor_vertices
    edges = []

    for i, idx in enumerate(coords):
        if tnbc_cells_type[p2c[i]]['name'] in exclude:
            continue
        i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i+1]]
        for cell in i_neigh:
            if tnbc_cells_type[p2c[cell]]['name'] in exclude:
                continue
            else:
                pair = np.array([i, cell])
                edges.append(pair)
    edges = np.asarray(edges).T

    G = nx.Graph()
    for left, right in edges.T:
        G.add_node(left, pos=coords[left], cell_type=p2c[left])
        G.add_node(right, pos=coords[right], cell_type=p2c[right])
        # filter according to upper bound distance limitation
        if (max_distance is None) or distance.euclidean(coords[left], coords[right]) <= max_distance:
            G.add_edge(left, right)

    clusters = None
    if filter_out_cell_type_cluster is not None:
        from plugin_clean_tumor_clusters import process_graph
        G, clusters = process_graph(G,
                                    cell_type_to_index[filter_out_cell_type_cluster],
                                    buffer=buffer,
                                    alpha=alpha)
    
    return G, clusters, p2c, coords, coords_included


def plot_graph_with_colors(G, idx2cell, coords, color_map):
    """Plot the graph with nodes colored by cell type."""
    plt.figure(figsize=(10, 10))
    pos = {i: coords[i] for i in range(len(coords))}
    node_colors = [color_map[idx2cell[node]] for node in G.nodes()]
    nx.draw(G, pos, with_labels=False, node_size=20, node_color=node_colors, edge_color='gray')
    plt.title("Network Graph with Node Colors by Cell Type")
    plt.show()
