import torch
import warnings
import pandas as pd
import numpy as np

from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import OneHotEncoder

from graphs_preprocess._dataset_contract import (
    DEFAULT_COLUMN_MAP,
    prepare_dataset_table,
)
from graphs_preprocess._graph_builder import build_fov_graph
from graphs_preprocess.node_feature_inspector import format_vector

warnings.filterwarnings("ignore")

class CellGraphDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 groups: list[str],
                 max_distance: float = 100,
                 microns_per_pixel: float = 1.0,
                 cells_to_filter: list[str] | None = None,
                 remove_clusters_of_cell_type: str | None = None,
                 alpha : float = 0.01,
                 buffer_value : float = 0,
                 df: pd.DataFrame | None = None,
                 plots_output_dir: str | None = None,
                 column_map: dict[str, str] | None = None,
                 ):
        super().__init__()

        # Merge with defaults so callers only need to override the
        # columns that actually differ for their dataset.
        self.column_map = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
        print("\nColumns recognized by CellGraphDataset:")
        for key, value in self.column_map.items():
            print(f"  {key}: {value}")
        self.patient_col = self.column_map["patient"]
        self.fov_col = self.column_map["fov"]
        self.label_col = self.column_map["label"]
        self.cell_type_col = self.column_map["cell_type"]
        self.x_col = self.column_map["x"]
        self.y_col = self.column_map["y"]

        self.df, self.patient_fov_combos, self.column_map = (
            prepare_dataset_table(
                csv_path=csv_path,
                df=df,
                groups=groups,
                column_map=self.column_map,
            )
        )

        self.pred_encoder = OneHotEncoder(sparse_output=False).fit(self.df[[self.cell_type_col]])
        #self.group_encoder = {group: i for i, group in enumerate(self.df['Group'].unique())} --- This line can reverse the labels
        self.group_encoder = {group: i for i, group in enumerate(groups)}
        
        # max_distance input is in whether the input csv coordinates are defined.
        self.max_distance = max_distance
        
        self.cells_to_filter = cells_to_filter
        self.remove_clusters_of_cell_type = remove_clusters_of_cell_type
        self.alpha = alpha
        self.buffer_value = buffer_value
        self.plots_output_dir = plots_output_dir

    def __len__(self):
        return len(self.patient_fov_combos)

    @property
    def cell_names(self) -> list[str]:
        """
        Return the ordered list of cell type feature names.

        Returns:
          (list[str]): Cell type names in the same order as the
            one-hot encoder output columns.

        """
        return self.pred_encoder.categories_[0].tolist()

    def __getitem__(self, idx):
        patient = self.patient_fov_combos.iloc[idx][self.patient_col]
        fov = self.patient_fov_combos.iloc[idx][self.fov_col]
        group = self.patient_fov_combos.iloc[idx][self.label_col]
        data_p = self.df[
            (self.df[self.patient_col] == patient) &
            (self.df[self.fov_col] == fov)
        ]
        selected_labels = data_p[self.label_col].unique()
        if len(selected_labels) != 1 or selected_labels[0] != group:
            raise RuntimeError(
                f"Internal label inconsistency for patient {patient!r}, "
                f"FOV {fov!r}: {selected_labels.tolist()}."
            )

        # print(
        #     f"\n===== Processing Patient {patient}, FOV {fov} "
        #     f"(Group: {group}) with {len(data_p)} cells "
        #     f"before filtering. ====="
        # )

        # Filter out the specified cell types
        if self.cells_to_filter:
            data_p = data_p[
                ~data_p[self.cell_type_col].isin(self.cells_to_filter)
            ]
        if data_p.empty:
            raise ValueError(
                f"Patient {patient!r}, FOV {fov!r}: no cells remain "
                "after filtering."
            )

        coords, cell_types, edge_index = build_fov_graph(
            data_p,
            max_distance=self.max_distance,
            remove_clusters_of_cell_type=self.remove_clusters_of_cell_type,
            alpha=self.alpha,
            buffer_value=self.buffer_value,
            plots_output_dir=self.plots_output_dir,
            patient=patient,
            fov=fov,
            x_col=self.x_col,
            y_col=self.y_col,
            cell_type_col=self.cell_type_col,
        )

        x = self.pred_encoder.transform([[ct] for ct in cell_types])
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(self.group_encoder[group], dtype=torch.long)
        pos = torch.tensor(coords, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, pos=pos, y=y)

    def print_node_feature_example(
            self, cell_type: str | None = None) -> None:
        """
        Print an example one-hot node feature vector and its meaning.

        Picks the first available cell type when cell_type is None.

        Args:
          cell_type (str | None): Cell type label to illustrate.
            Uses the first encoder category when None.

        Raises:
          ValueError: If cell_type is not in the encoder's categories.

        """
        categories = self.pred_encoder.categories_[0]

        if cell_type is None:
            cell_type = categories[0]
        elif cell_type not in categories:
            raise ValueError(
                f"Cell type '{cell_type}' not found. "
                f"Available: {list(categories)}"
            )

        vec = self.pred_encoder.transform([[cell_type]])[0]
        active_idx = int(np.argmax(vec))
        dim = len(vec)

        print("\n=== Node Feature Example (One-Hot Encoding) ===")
        print(f"  Cell type  : \"{cell_type}\"")
        print(f"  Feature dim: {dim}")
        print(f"  Vector     : {format_vector(vec)}")
        print()
        print("  Active features:")
        print(
            f"    Index {active_idx:>3} → \"{categories[active_idx]}\""
        )
        print()
        print(
            "  Interpretation: one-hot encoding places exactly 1 at "
            "the index\n"
            "  that corresponds to the cell type, 0 everywhere else."
        )
        print()
        print("  All feature columns  (index → cell type label):")
        for i, cat in enumerate(categories):
            marker = "  ← active" if i == active_idx else ""
            print(f"    [{i:>3}]  {cat}{marker}")
