import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np


def convert_to_labels(original_dict: dict, cells_type: dict):
    return {k: cells_type[int(v)] for k, v in original_dict.items()}


def draw_motif(motif: nx.Graph, cells_type: dict):
    plt.figure(figsize=(10, 6))
    new_labels = convert_to_labels(nx.get_node_attributes(motif, name='type'), cells_type)

    nx.draw(motif.to_undirected(),
            labels=new_labels,
            with_labels=True,
            node_size=4000,
            node_color='lightblue',
            font_color="black")
    plt.draw()
    plt.show()
    plt.close()


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = matplotlib.colormaps.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_colors_bar(discriminator,
                    ratio_gain_results: pd.DataFrame,
                    cells_type: dict):
    colors_data = None
    count = 0
    for idx, row in ratio_gain_results.iterrows():
        colors_vec_hash = row['hash']
        colors_vec = discriminator.cism.motifs_dataset[discriminator.cism.motifs_dataset.colors_vec_hash == colors_vec_hash].iloc[0].colors_vec
        count += len(colors_vec)
        if colors_data is None:
            colors_data = np.array(colors_vec)
        else:
            colors_data = np.append(colors_data, colors_vec)

    colors_data = colors_data.reshape(ratio_gain_results.shape[0], len(cells_type))
    colors_data = colors_data.transpose()
    print(colors_data.shape)
    number_of_colors = 5
    plt.grid(visible=True, color='black', linestyle='-', linewidth=1)
    plt.imshow(colors_data, cmap=discrete_cmap(number_of_colors, 'cubehelix_r'), vmin=0, vmax=number_of_colors)
    plt.colorbar(ticks=range(number_of_colors), fraction=0.018, pad=0.02)
    plt.clim(-0.5, number_of_colors - 0.5)
    plt.xticks(np.arange(-.5, ratio_gain_results.shape[0]-1, 1), labels=[])
    plt.yticks(ticks=np.arange(-.5, len(cells_type)-1, 1), labels=[])
    plt.show()
