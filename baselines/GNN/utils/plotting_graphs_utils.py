import matplotlib.pyplot as plt
import networkx as nx

from utils.plotting_utils import save_plot_to_job_dir

def plot_before_after(G_before: nx.Graph, G_after: nx.Graph, 
                      patient: int, fov: int, 
                      target_cell_type: str,
                      output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, G, title in zip(
        axes,
        [G_before, G_after],
        ['Before removal', 'After removal'],
    ):
        pos = nx.get_node_attributes(G, 'pos')
        colors = [
            'red' if G.nodes[n]['cell_type'] == target_cell_type else 'steelblue'
            for n in G.nodes()
        ]
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=colors, node_size=15,
            with_labels=False, edge_color='#cccccc',
        )
        ax.set_title(f"{title} — {G.number_of_nodes()} nodes")
        ax.set_aspect('equal')

    plt.suptitle(f"Patient {patient}, FOV {fov}")
    plt.tight_layout()
    save_plot_to_job_dir(plt.gcf(), f'patient_{patient}_fov_{fov}_{target_cell_type}.png', output_dir)
