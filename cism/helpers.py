import base64
import pickle
from pathlib import Path
import shlex

import subprocess
import pandas as pd
import networkx as nx
import os
import numpy as np


def create_weight_matrix_from_motifs(discriminator,
                                     cells_type: dict,
                                     cell_identity_to_motif_hash: dict,
                                     motifs_ids: list,
                                     motifs_weight: pd.DataFrame) -> pd.DataFrame:
    pairwise_cell_types_count_matrix = pd.DataFrame(index=cells_type.values(),
                                                    columns=cells_type.values(),
                                                    dtype=float).fillna(0)

    for hash_id in list(cell_identity_to_motif_hash.keys()):
        motifs = cell_identity_to_motif_hash[hash_id]
        for motif_id in motifs:
            if motif_id not in motifs_ids:
                continue
            print(f'motif_id: {motif_id}, hash_id: {hash_id}')
            target_motif = discriminator.cism.motifs_dataset[discriminator.cism.motifs_dataset.ID == motif_id].iloc[0].motif
            target_motif = string_base64_pickle(target_motif)
            for edge in nx.Graph(target_motif).edges():
                left_node = target_motif.nodes[edge[0]]['type']
                right_node = target_motif.nodes[edge[1]]['type']
                pairwise_cell_types_count_matrix.loc[cells_type[int(left_node)], cells_type[int(right_node)]] += motifs_weight.loc[motif_id]

    return pairwise_cell_types_count_matrix/np.matrix(pairwise_cell_types_count_matrix).sum()

def run_fanmod(
        FANMOD_exe: str,
        FANMOD_path: str,
        iterations: str,
        motif_size: int,
        output_dir: str,
        raw_data_folder: str,
        file: str,
        patient_num: str) -> bool:
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"{FANMOD_path}/{FANMOD_exe} -i {raw_data_folder}/{file} -o {output_dir}/{patient_num}.csv -r {iterations} -s {motif_size} --colored_vertcies")
        process = subprocess.Popen(shlex.split(f"{FANMOD_path}/{FANMOD_exe} -i {raw_data_folder}/{file} -o {output_dir}/{patient_num}.csv -r {iterations} -s {motif_size} --colored_vertcies -d"))
        process.communicate()
        if process.returncode != 0:
            raise Exception("error occurs while trying to run fanmod+")

        return Path(output_dir + '/' + patient_num + '.csv').exists()
    except FileExistsError as e:
        print(f'{file} failed to create folder {output_dir} error: {e}')
        return False


def parse_leda(lines: [str]) -> nx.Graph:
    """
    The code was originally written in networkx.
    Here I just adjusted the parser to support the format that exported from the FANMOD tool.
    :param lines: leda lines
    :return: networkx graph object
    """
    lines = iter(
        [
            line.rstrip("\n")
            for line in lines
            if not (line.startswith("#") or line.startswith("\n") or line == "")
        ]
    )
    # skip the node label type and edge label type
    # skip LEDA.GRAPH
    # GraphBuilder
    for i in range(3):
        next(lines)
    du = int(next(lines))  # -1=directed, -2=undirected
    if du == -1:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    # Nodes
    n = int(next(lines))  # number of nodes
    node = {}
    for i in range(1, n + 1):  # LEDA counts from 1 to n
        symbol = next(lines).rstrip().strip("|{}|  ")
        if symbol == "":
            symbol = str(i)  # use int if no label - could be trouble
        node[i] = symbol

    graph.add_nodes_from([(i, {'type': s}) for i, s in node.items()])

    # Edges
    m = int(next(lines))  # number of edges
    for i in range(m):
        try:
            s, t, reversal, label = next(lines).split()
        except BaseException as err:
            raise Exception(f"Too few fields in LEDA.GRAPH edge {i+1}") from err

        # if reversal edge and directed then handle
        if bool(reversal) and du == -1:
            graph.add_edge(int(t), int(s), label=label[2:-2])

        graph.add_edge(int(s), int(t), label=label[2:-2])

    return graph


def parse_csv(filepath: str, patient_num: int, fov: str) -> pd.DataFrame:
    print(f'parse csv: {filepath} patient_num: {patient_num}, FOV: {fov}')
    df = pd.DataFrame()
    f = open(filepath, "r")
    line = f.readline()
    while not line.endswith('motifs were found.\n'):
        line = f.readline()

    total = line.split(' ')[0]

    line = f.readline()
    while not line.startswith('ID,'):
        line = f.readline()

    # line should point to the header
    # pbar = tqdm(total=int(total), mininterval=10)
    patient_num_list = []
    fov_list = []
    sample_id_list = []
    freq_list = []
    count_list = []
    mean_freq_list = []
    std_list = []
    z_score_list = []
    p_value_list = []
    hash_list = []
    motif_list = []
    while 1:
        line = f.readline()
        if line.rstrip('\n') == "":
            # no more lines
            break

        # line with result
        sample_id, freq, count, mean_freq, std, z_score, p_value = line.strip().split(',')

        leda_instance_lines = []
        while not line.startswith('#+leda'):
            line = f.readline()

        line = f.readline()
        while not line.startswith('#-leda'):
            leda_instance_lines.append(line)
            line = f.readline()

        graph: nx.Graph = parse_leda(leda_instance_lines)
        patient_num_list.append(patient_num)
        fov_list.append(fov)
        sample_id_list.append(sample_id)
        freq_list.append(freq)
        count_list.append(count)
        mean_freq_list.append(mean_freq)
        std_list.append(std)
        z_score_list.append(z_score if z_score != "undefined" else pd.NA)
        p_value_list.append(p_value)
        hash_list.append(nx.weisfeiler_lehman_graph_hash(graph, node_attr='type'))
        motif_list.append(graph)

        # empty line
        f.readline()
        # pbar.update(1)
    f.close()
    # pbar.close()

    df = pd.concat([df, pd.DataFrame.from_dict({
        'Patient': patient_num_list,
        'FOV': fov_list,
        'ID': sample_id_list,
        'Freq': freq_list,
        'Count': count_list,
        'Mean_Freq': mean_freq_list,
        'STD': std_list,
        'z_score': z_score_list,
        'p_value': p_value_list,
        'hash': hash_list,
        'motif': motif_list})],
        ignore_index=True)

    return df


def pickle_base64_stringify(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('ascii')


def string_base64_pickle(obj):
    return pickle.loads(base64.b64decode(obj.encode()))


def analyze_file(
                 FANMOD_exe: str,
                 FANMOD_path: str,
                 iterations: str,
                 motif_size: int,
                 file: str,
                 output_dir: str,
                 cache_dir: str,
                 force_run_fanmod: bool,
                 raw_data_folder: str,
                 force_parse: bool,
                 enable_parse: bool,
                 p_value: float = 0.05,
                 quantile_threshold: float = 0.9) -> pd.DataFrame:
    import re
    import pathlib

    empty_df = pd.DataFrame()
    _, file_type = re.split("[.]", file)
    if file_type != 'txt':
        return empty_df
    _, patient_num, fov, _ = re.split("[_ .]", file)

    pathlib.Path(output_dir + '/' + str(motif_size) + '/').mkdir(exist_ok=True, parents=True)
    result_file = output_dir + '/' + str(motif_size) + '/' + patient_num + '_' + fov + '.csv'
    if not pathlib.Path(result_file).exists() or force_run_fanmod or pathlib.Path(result_file).stat().st_size < 2048:
        if not run_fanmod(FANMOD_exe=FANMOD_exe,
                          FANMOD_path=FANMOD_path,
                          iterations=iterations,
                          motif_size=motif_size,
                          output_dir=output_dir + '/' + str(motif_size) + '/',
                          raw_data_folder=raw_data_folder,
                          file=file,
                          patient_num=patient_num + '_' + fov):
            print(f'failed to run fanmod. File: {raw_data_folder}/{file}')
            return empty_df

    if not enable_parse:
        return empty_df

    # check if cache is exist
    pathlib.Path(cache_dir + '/' + str(motif_size) + '/').mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir + '/' + str(motif_size) + '/' + patient_num + '_' + fov + '.df_cache'
    if force_parse or not pathlib.Path(cache_file).exists():
        if not pathlib.Path(cache_dir).exists():
            os.makedirs(cache_dir, exist_ok=True)
        df = parse_csv(filepath=result_file, patient_num=patient_num, fov=fov)
        df['nunique_colors'] = df.motif.apply(
            lambda row: len(set(nx.get_node_attributes(row, name='type').values())))
        df['motif'] = df['motif'].transform(lambda graph: pickle_base64_stringify(graph))
        df.to_csv(cache_file)
        df = None

    df = pd.read_csv(cache_file, index_col=[0], engine='pyarrow')
    '''
    In case that FANMOD+ cannot generate random graphs (field of view have few nodes), 
    since there are not enough edges or vertices the Mean_Freq will be equal to nan.
    we cleanup those result since they are not significant.
    '''
    df = df[~df['Mean_Freq'].isna()]

    df['ID'] = df.ID.astype('int64')
    df['Patient'] = df.Patient.astype('int16')
    df.Freq = pd.to_numeric(df.Freq, errors='coerce')
    df.p_value = pd.to_numeric(df.p_value, errors='coerce')
    df.z_score = pd.to_numeric(df.z_score, errors='coerce')
    df.Count = pd.to_numeric(df.Count, errors='coerce')
    df['FOV'] = df['FOV'].astype('category')
    df['Mean_Freq'] = df['Mean_Freq'].astype('float32')
    df['STD'] = df['STD'].astype('float32')
    df['z_score'] = df['z_score'].astype('float32')
    df = df.drop('hash', axis=1)

    # filters
    # df['motif'] = df['motif'].apply(lambda row: string_base64_pickle(row))
    df = df[df.p_value < p_value]
    df['p_value'] = df['p_value'].astype('float32')
    quantile_slice = df['Freq'].quantile(q=quantile_threshold)
    df = df[df['Freq'] <= quantile_slice]

    return df
