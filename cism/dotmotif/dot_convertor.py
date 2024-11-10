import dotmotif
import networkx as nx


def dotmotif_exist_edge(u:str, v: str) -> str:
    return f'{u} -> {v}\n'


def dotmotif_non_exist_edge(u:str, v: str) -> str:
    return f'{u} !> {v}\n'


def convert_nx_graph_to_dotmotif(graph: nx.Graph,
                                 induced: bool,
                                 ignore_attr: bool = False) -> (dotmotif.Motif, str, list):
    motif_string = ""

    included_nodes = set()

    #convert edges
    for node_i in graph.nodes():
        for node_j in graph.nodes():
            #skip self edges
            if node_i == node_j:
                continue

            #skip directed edges
            if isinstance(node_i, str) and len(node_i) == 0:
                if ord(node_i) > ord(node_j):
                    continue
            else:
                if node_i > node_j:
                    continue

            if graph.edges().get((node_i, node_j)) is not None:
                motif_string += dotmotif_exist_edge(node_i, node_j)
                included_nodes.add(node_i)
                included_nodes.add(node_j)

    if induced:
        for node_i in graph.nodes():
            for node_j in graph.nodes():
                #skip self edges
                if node_i == node_j:
                    continue

                #skip directed edges
                if isinstance(node_i, str) and len(node_i) == 0:
                    if ord(node_i) > ord(node_j):
                        continue
                else:
                    if node_i > node_j:
                        continue

                if (graph.edges().get((node_i, node_j)) is None
                    and ((node_i in included_nodes) and (node_j in included_nodes))):
                        motif_string += dotmotif_non_exist_edge(node_i, node_j)

    #convert nodes attributes
    nodes_type = set()
    if not ignore_attr:
        for node_i in graph.nodes():
            if (not induced) and (node_i not in included_nodes):
                continue
            for k, v in graph.nodes()[node_i].items():
                nodes_type.add(v)
                motif_string += f'{node_i}.{k} = "{v}"'
                motif_string += '\n'

    motif_result = dotmotif.Motif(motif_string)
    motif_result.ignore_direction = True

    return motif_result, motif_string, sorted(list(nodes_type))
