import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import pickle

class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class GCN_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_channels=[75, 150, 50], output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.output_size = output_size
        self.gcn_blocks = torch.nn.ModuleList()
        in_channels = self.input_size
        for out_channels in self.hidden_channels:
            self.gcn_blocks.append(GCNBlock(in_channels, out_channels))
            in_channels = out_channels
        self.output_layer = torch.nn.Linear(self.hidden_channels[-1], self.output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for gcn_block in self.gcn_blocks:
            x = gcn_block(x, edge_index)
        x = global_max_pool(x, data.batch)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

