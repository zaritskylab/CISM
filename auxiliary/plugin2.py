from plugin_utils import *
import napari
import magicgui as mg
import imageio
import numpy as np
import itertools
import pandas as pd


@mg.magicgui
class GraphVisualizer:
    def __init__(self):
        self.ui = self.create_ui()
        self.viewer = napari.Viewer()

    def create_ui(self):        
        self.csv_file = mg.widgets.FileEdit(filter="CSV files (*.csv)", mode="r", label="CSV File Path")
        self.patient_id_input = mg.widgets.LineEdit(name="Patient ID", label="Patient ID")
        self.fov_input = mg.widgets.LineEdit(name="FOV", label="Field of View (FOV)")
        self.visualize_button = mg.widgets.PushButton(text="Visualize")
        self.visualize_button.clicked.connect(self.on_button_click)
        self.image_file = mg.widgets.FileEdit(filter="Images (*.png *.xpm *.jpg *.bmp *.tif *.tiff)", mode="r", label="Image Path")
        self.motif_input = mg.widgets.Textarea(value="", label="Motif Input")
        self.split_full_res = mg.widgets.CheckBox(value=False, text="Group cells by full cell type resolution")
        self.remove_cell_type_cluster = mg.widgets.CheckBox(value=False, text="Removing cell type clusters")
        self.remove_cell_type = mg.widgets.LineEdit(value="Tumor", label="Remove Cell type")
        self.remove_cluster_alpha = mg.widgets.LineEdit(value="1.0", label="Remove cluster alpha")
        self.remove_cluster_buffer = mg.widgets.LineEdit(value="0", label="Remove cluster buffer")
        self.data_choice_widget = mg.widgets.ComboBox(choices=["onlyCSV", "segmentedImgae", "labeledImgae"], name="Data Type")
        self.dataset_choice_widget = mg.widgets.ComboBox(choices=["Melanoma-MIBITOF", "TNBC-MIBITOF", "TNBC-IMC", "Other"], name="Dataset Name")
        self.max_distance = mg.widgets.LineEdit(name="max_distance", label="Max Distance")
        layout = mg.widgets.Container(widgets=[self.csv_file,
                                               self.patient_id_input,
                                               self.fov_input,
                                               self.image_file,
                                               self.visualize_button,
                                               self.motif_input,
                                               self.data_choice_widget,
                                               self.dataset_choice_widget,
                                               self.max_distance,
                                               self.split_full_res,
                                               self.remove_cell_type_cluster,
                                               self.remove_cell_type,
                                               self.remove_cluster_alpha,
                                               self.remove_cluster_buffer])
        return layout

    def _common_cells(self, dataset_name: str) -> dict:
        common_cells_mapper = None
        if dataset_name == 'Melanoma-MIBITOF':
            common_cells_mapper = {
              'DC sign Mac': 'MAC',
              'CD4 T cell': "CD4 T cell",
              'Memory%CD4 T cell': "Memory CD4 T Cell",
              'Tfh%CD4 T cell': "CD4 T cell",
              'CD4 APC': "CD4 APC",
              'Collagen_sma': "Stroma",
              'blood vessels': "Vessels",
              'Collagen': "Stroma",
              'CD8 T cell': "CD8 T cell",
              'SMA': "Stroma",
              'Hevs': "Hevs",
              'CD206_Mac': "MAC",
              'B cell': "B cell",
              'CD45RA+MHC2+': 'B cell',
              'Mac': "MAC",
              'Mono_CD14_DR': "MAC",
              'Neutrophil': "Neutrophil",
              'NK cell': "NK cell",
              'Follicular_Germinal_B_Cell': "Germinal Center B cell",
              'CD4 Treg%CD4 T cell': "CD4 Treg",
              'DCs': "DCs",
              'CD68_Mac': "MAC",
              'CD14_CD11c_DCs': "DCs",
              'CD11_CD11c_DCsign_DCs': "DCs",
              "tumor": "Tumor",
              "Unidentified": "Unidentified"
            }
        elif dataset_name == 'TNBC-MIBITOF':
            common_cells_mapper = {
              1: 'Unidentified',
              2: 'Endothelial',
              3: 'Mesenchyme',
              4: 'Tumor',
              5: 'Tregs',
              6: 'CD4 t cells',
              7: 'CD8 T cells',
              8: 'CD3 T cells',
              9: 'NK cells',
              10: 'B cells',
              11: 'Neutrophils',
              12: 'Macrophages',
              13: 'DC',
              14: 'DC/Mono',
              15: 'Mono/Neu',
              16: 'Immune other'
            }
        elif dataset_name == 'TNBC-IMC':
            common_cells_mapper = cells_mapper = {
              'Exhausted T-cell': 'T Cell',
              'NK cell (FOXP3)': "NK Cell",
              'Th1 T helper': "Th1 T Helper",
              'CAF': 'CAF',
              'GB Active T cell': 'T Cell',
              'Monocyte (Tbet)': 'Monocytes',
              'Macrophage': 'Macrophage',
              'T-cell': 'T Cell',
              'B7H4+ Cancer (CD11b)': 'Cancer',
              'Unassigned': 'Unassigned',
              'Endothelial': 'Endothelial',
              'Monocytes': 'Monocytes',
              'Memory T cell': 'T Cell',
              'Monocyte (HLA)': 'Monocytes',
              'CD44+ MAC ': "Macrophage",
              'B7H4+ Cancer': "Cancer",
              'Mem CD8 T cell': "CD8 T Cell",
              'Cancer stem cell': "Cancer",
              'Antigen presenting cell': "APC",
              'Reg T-cell': "Reg T Cell",
              'Cytotoxic T cell': "Cytotoxic T Cell",
              'E-Cad cancer': "Cancer",
              'FOXP3+ Cancer': "Cancer",
              'PanCK Cancer': "Cancer",
              'B-cell': "B Cell",
              'Alfa-SMA+ Monocytes': "SMA",
              'T-cell/Cancer': "Cancer",
              'Immune/Cancer': 'Immune/Cancer',
              'B/T/NK cell': "B/T/NK Cell",
              'Vim+ Cancer / NK': "Cancer/NK Cell",
              "T/B cells": "T/B Cell",
              "NK cells / Cancer": "Cancer/NK Cell"
            }
        else:
            common_cells_mapper = None
        return common_cells_mapper

    def on_button_click(self):
        print("Button clicked!")
        print(f"Motif input content: {self.motif_input.value}")
        csv_path = self.csv_file.value
        patient_id = int(self.patient_id_input.value)
        fov = self.fov_input.value
        image_full_path = self.image_file.value
        if self.data_choice_widget.value == 'onlyCSV':
            data = pd.read_csv(fr'{csv_path}')
            common_cells_mapper = self._common_cells(self.dataset_choice_widget.value)
            if common_cells_mapper is None:
                data['common_cell_types'] = data["pred"]
                cell_types = np.unique(data["pred"])
            else:
                data['common_cell_types'] = data["pred"].apply(lambda x: common_cells_mapper[x])
                cell_types = list(common_cells_mapper.values())
            if fov != "":
                data = data[(data['patient number'] == patient_id) & (data['fov'] == fov)]
            else:
                data = data[(data['patient number'] == patient_id)]
            coords = data[['centroid-0', 'centroid-1']].values
            tnbc_cells_type = generate_cell_type_structure(cell_types)
            print(tnbc_cells_type)
            #print(len(np.unique(cell_types)))
            max_distance = None if self.max_distance.value == "" else float(self.max_distance.value)

            filter_out_cell_type_cluster = None
            if self.remove_cell_type_cluster.value:
                filter_out_cell_type_cluster = self.remove_cell_type.value

            G_full, clusters, p2c, coords, coords_included = build_cell_graph(data,
                                                                    tnbc_cells_type,
                                                                    max_distance,
                                                                    exclude=[],
                                                                    filter_out_cell_type_cluster=filter_out_cell_type_cluster,
                                                                    alpha=float(self.remove_cluster_alpha.value))
           
        elif self.data_choice_widget.value == 'segmentedImgae':
            mapper = {'patints_col' : 'SampleID',
                        'cell_types_col' : 'cell_type',
                        'cell_index_col' : 'cellLabelInImage',
                        'fov_col' : 'FOV'}
            tb = pd.read_csv(fr'{csv_path}')
            cell_types = np.unique(tb['cell_type'])
            image_full = imageio.imread(image_full_path).astype(np.uint8)
            image_full = create_tagged_image(tb,image_full, mapper, patinet_number=patient_id, fov = fov)
            image_original, G_full, coords, point2cell_full, p2c = build_graph(image_full, list_of_cells_to_exclude = [4,5,6])
            tnbc_cells_type = generate_cell_type_structure(cell_types)
            print(tnbc_cells_type)
            
        elif self.data_choice_widget.value == 'labeledImgae':
            image_full = imageio.imread(image_full_path)  
            image_original, G_full, coords, point2cell_full, p2c = build_graph(image_full, list_of_cells_to_exclude = [])
            tnbc = pd.read_csv(fr'{csv_path}')
            cell_types = np.unique(tnbc['CellType'])
            mapper_cells = {
                1: 'Unidentified',
                2: 'Endothelial',
                3: 'Mesenchyme',
                4: 'Tumor',
                5: 'Tregs',
                6: 'CD4 t cells',
                7: 'CD8 T cells',
                8: 'CD3 T cells',
                9: 'NK cells',
                10: 'B cells',
                11: 'Neutrophils',
                12: 'Macrophages',
                13: 'DC',
                14: 'DC/Mono',
                15: 'Mono/Neu',
                16: 'Immune other'
            }
            tnbc_cells_type = generate_cell_type_structure_from_tagged(mapper_cells)
            print(tnbc_cells_type)

        split_full_res_checkbox = self.split_full_res.value

        cell_type_set = set(common_cells_mapper.values())
        if split_full_res_checkbox:
            cell_type_set = set(common_cells_mapper.keys())

        p2fc = {idx: cell_type for idx, cell_type in enumerate(data['pred'].values)}

        # add points. Classify each point to cell type and add to a separate layer
        cell_type_colors_list = dict.fromkeys(cell_type_set, None)
        for k, v in cell_type_colors_list.items():
            cell_type_colors_list[k] = []

        cell_type_coords_list = dict.fromkeys(cell_type_set, None)
        for k, v in cell_type_coords_list.items():
            cell_type_coords_list[k] = []

        for i in p2c:
            if i not in G_full.nodes():
                continue

            cell_t = p2c[i]
            cell_t_name = tnbc_cells_type[cell_t]['name']
            if split_full_res_checkbox:
                cell_t_name = p2fc[i]

            cell_type_colors_list[cell_t_name].append(tnbc_cells_type[cell_t]['color'])
            cell_type_coords_list[cell_t_name].append(coords[i])

        for cell_t in cell_type_set:
            if len(cell_type_coords_list[cell_t]) < 1:
                continue
            rotated_all_nodes_coords = rotate_point(np.array(cell_type_coords_list[cell_t]), 270)
            print(f'cell_type: {cell_t}, num: {len(rotated_all_nodes_coords)}')
            self.viewer.add_points(rotated_all_nodes_coords, size=15, face_color=cell_type_colors_list[cell_t], name=cell_t)

        def calculate_convex_hulls(cluster):
            """
            Calculates the convex hull for each cluster.

            :param clusters: A list of subgraphs, each representing a cluster.
            :return: A list of convex hull vertices.
            """
            hull_vertices = None
            positions = nx.get_node_attributes(cluster, 'pos')
            points = np.array(list(positions.values()))
            if len(points) > 2:  # Convex hull needs at least 3 points
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                hull_vertices = points[hull.vertices]
            return hull_vertices

        def calculate_alpha_shapes(cluster, buffer=0.0, alpha=1.0):
            """
            Calculates the alpha shape for each cluster.

            :param clusters: A list of subgraphs, each representing a cluster.
            :param alpha: The alpha parameter controlling the tightness of the shape.
            :return: A list of vertices representing the alpha shapes.
            """
            alpha_shape = None
            positions = nx.get_node_attributes(cluster, 'pos')
            points = np.array(list(positions.values()))
            if len(points) > 2:  # Alpha shape needs at least 3 points
                from alphashape import alphashape
                shape = alphashape(points, alpha)
                from shapely import Polygon
                if isinstance(shape, Polygon):
                    shape = shape.buffer(buffer)
                    alpha_shape = np.array(shape.exterior.coords)
            return alpha_shape

        if clusters is not None:
            cluster_idx = 0
            for cluster in clusters:
                cluster_coord_list = []
                for i in p2c:
                    if i in cluster.nodes():
                        cluster_coord_list.append(coords[i])

                rotated_all_nodes_coords = rotate_point(np.array(cluster_coord_list), 270)
                self.viewer.add_points(rotated_all_nodes_coords, size=15, name=f'cluster: {cluster_idx}')
                hull_vertices = calculate_alpha_shapes(cluster,
                                                       buffer=float(self.remove_cluster_buffer.value),
                                                       alpha=float(self.remove_cluster_alpha.value))
                if hull_vertices is not None:
                    hull_vertices = rotate_point(np.array(hull_vertices), 270)
                    self.viewer.add_shapes([hull_vertices], shape_type='polygon', edge_color='red', face_color='transparent',
                                          edge_width=10, name=f'cluster convex hull: {cluster_idx}')

                cluster_idx += 1

        '''            
        c = []
        for i in p2c:
            cell_t = p2c[i]
            c.append(tnbc_cells_type[cell_t]['color'])
        rotated_all_nodes_coords = rotate_point(coords, 270)
        self.viewer.add_points(rotated_all_nodes_coords, size=15, face_color=c, name="All Cells")
        '''
        node_ft_full = list(G_full.nodes(data=True))
        edges_full = list(G_full.edges())
        corrected_coords = []
        colors = []

        node_indices = [node[0] for node in node_ft_full]  # Extract the node indices
        # Only consider nodes that are part of the network subset
        for node, data in G_full.nodes(data=True):
            if node in node_indices:  # Check if node is part of the subset
                corrected_coords.append(coords[node])
                colors.append(tnbc_cells_type[data['cell_type']]['color'])

        
        #rotated_coords = rotate_point(corrected_coords, 270)
        #self.viewer.add_points(rotated_coords, size=15, face_color=colors, name="Only Commons")
        
        # Visualize edges as lines in Napari
        lines = []
        for edge in edges_full:
            lines.append([coords[edge[0]], coords[edge[1]]])
        
        rotated_lines = [rotate_point(line, 270) for line in lines]
        self.viewer.add_shapes(rotated_lines, shape_type='line', edge_color='white', name="Edges")
        #motifs = find_motif_in_graph(G_full, connections, non_connections)
        if not self.motif_input.value.strip():
            motif = {
                ('A', 'B'): {}, 
                ('A', 'C'): {},
                ('B', 'C'): {},
                'A': {'type': 9},
                'B': {'type': 11},
                'C': {'type': 11}
            }
        else:
            motif = parse_motif_input(self.motif_input.value, tnbc_cells_type)

        motifs = find_motifs(G_full, motif)
        all_motif_points = []
        all_motif_lines = []

        # Predefined colors for different motif types
        colors = ['lime', 'fuchsia', 'yellow', 'cyan', 'orange', 'hotpink', 'chartreuse', 'aqua', 'gold', 'coral']
        color_index = 0
        motif_node_size = 10  # Increase as needed
        motif_edge_width = 2  # Increase as needed
        for motif_type in motifs:
            motif_color = colors[color_index % len(colors)]
            color_index += 1

            for motif_instance in motif_type:
                # Extract the actual node indices from the motif instance
                motif_nodes = list(motif_instance.keys())
                motif_edges = list(itertools.combinations(motif_nodes, 2))
                # Rotate and visualize motif nodes
                rotated_motif_coords = [point for node in motif_nodes for point in rotate_point(coords[node], 270)]
                all_motif_points.extend(rotated_motif_coords)
                # Rotate and visualize motif edges
                rotated_motif_line_coords = []
                for line in motif_edges:
                    if G_full.has_edge(line[0], line[1]):
                        start_point = rotate_point(coords[line[0]], 270)
                        end_point = rotate_point(coords[line[1]], 270)
                        rotated_motif_line_coords.extend([start_point, end_point])
                all_motif_lines.extend(rotated_motif_line_coords)

            if len(all_motif_points) > 0:
                c = np.array(all_motif_points).reshape(-1,2)
                l = np.array(all_motif_lines).reshape(-1,2,2)
                self.viewer.add_shapes(l, shape_type='line', edge_color=motif_color,edge_width=motif_edge_width, name="Motif Edges")
                self.viewer.add_points(c, size=motif_node_size, face_color=motif_color, name="Motif Nodes", edge_color=motif_color)

            # Reset for the next motif type
            all_motif_points = []
            all_motif_lines = []

        ellipses = []
        colors = []
        labels = []
        y_coord = 30  # Starting y-coordinate for the center of the ellipse
        radius = 20  # Radius of each ellipse for both x and y axes
        for idx, details in tnbc_cells_type.items():
            ellipse_center = [y_coord, 30]
            ellipses.append([ellipse_center, [radius, radius]])
            
            colors.append(details['color'])
            labels.append(details['name'])
            y_coord += 60 

        # Add the Shapes layer with rectangles and colors
        shapes_layer = self.viewer.add_shapes(data=ellipses, shape_type='ellipse', edge_color=colors, face_color=colors, name="Legend")

        # Set the labels as the properties for the Shapes layer
        shapes_layer.properties = {'label': labels}
        shapes_layer.text = {'text': '{label}',
                             'size': 10,
                             'anchor': 'lower_right',
                             'color': 'white',
                             'translation': [-25, +200]}




if __name__ == "__main__":
    visualizer = GraphVisualizer()

    with napari.gui_qt():
        visualizer.viewer.window.add_dock_widget(visualizer.ui, area='right')
