import torch
import torch.nn.functional as F
import plotly.graph_objs as go
import networkx as nx
import numpy as np

def get_node_importance(model, data):
    """
    Compute saliency maps by calculating gradients with respect to the input nodes.
    This helps identify the most important nodes for the prediction.
    """
    data = data.to('cpu')
    model.to('cpu')
    data.x.requires_grad = True
    output = model(data)
    loss = F.binary_cross_entropy(output, data.y.float().view(-1, 1))
    loss.backward()
    importance_scores = data.x.grad.abs().cpu().numpy()
    
    return importance_scores

def visualize_graph_with_importance_interactive(data, importance_scores, cell_type_decoder, threshold='auto', output_html="graph.html"):
    """
    Visualizes the graph with node importance and exports it as an interactive HTML.
    Args:
        data: torch_geometric.data.Data object
        importance_scores: Gradient scores for the nodes
        cell_type_decoder: Dictionary to map one-hot encoded vectors back to cell type names
        threshold: A value above which a node is considered important, or 'auto' to set dynamically.
        output_html: Path to save the interactive graph as HTML.
    """
    # Create a graph using the edge index
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    pos = data.pos.cpu().numpy()

    # Add nodes with positions and importance scores
    for i, (x, coord) in enumerate(zip(importance_scores, pos)):
        cell_type = cell_type_decoder[np.argmax(data.x[i].detach().cpu().numpy())]
        importance = x.mean()
        G.add_node(i, pos=coord, cell_type=cell_type, importance=importance)

    # Add edges
    for i, j in zip(edge_index[0], edge_index[1]):
        G.add_edge(i, j)

    # Get node attributes
    pos = nx.get_node_attributes(G, 'pos')
    importance = np.array([G.nodes[n]['importance'] for n in G.nodes()])
    
    if threshold == 'auto':
        threshold = np.percentile(importance, 80)  # Top 20% most important nodes

    node_sizes = [15 if G.nodes[n]['importance'] > threshold else 5 for n in G.nodes()]
    node_colors = ['red' if G.nodes[n]['importance'] > threshold else 'blue' for n in G.nodes()]
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Cell Type: {G.nodes[node]['cell_type']}, Importance: {G.nodes[node]['importance']:.3f}")
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line_width=2),
        text=node_text,
        hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Interactive Graph Visualization with Node Importance',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    fig.write_html(output_html)
    print(f"Interactive graph saved to {output_html}")
'''
## Example usage, once you have a trained model:
with open('mel_alpha_0.01_buffer_0.pickle', 'rb') as p:
    dataset = pickle.load(p)
test_data = dataset[-1] #let's explain the last test sample
importance_scores = get_node_importance(model, test_data)
cell_type_decoder = {i: label for i, label in enumerate(dataset.pred_encoder.categories_[0])}
visualize_graph_with_importance_interactive(test_data, importance_scores, cell_type_decoder, threshold='auto', output_html="graph_visualization2.html")
'''