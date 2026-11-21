import pandas as pd

from baselines.pairwise.reader import GraphReader
from figures.helpers.datasets import DatasetConfig


def read_graphs(config: DatasetConfig) -> pd.DataFrame:
    """One row per network file of the dataset: its graph, pairwise cell type counts and Patient_uId."""
    return GraphReader.get_graphs(full_graph_df=pd.DataFrame(),
                                  raw_data_folder=str(config.graphs_dir.parent),
                                  raw_data_folder_type=f'/{config.graphs_dir.name}',
                                  disease=config.patient_uid_prefix,
                                  cells_type=config.cells_type)
