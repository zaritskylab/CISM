import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class Enhanced_GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels=[64, 32, 16], output_size=1, 
                 dropout_rate=0.25, conv_type='gcn', residual=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.conv_type = conv_type
        
        # Create convolution layers based on conv_type
        if conv_type == 'gcn':
            self.conv1 = GCNConv(input_size, hidden_channels[0])
            self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
            self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        elif conv_type == 'sage':
            self.conv1 = SAGEConv(input_size, hidden_channels[0])
            self.conv2 = SAGEConv(hidden_channels[0], hidden_channels[1])
            self.conv3 = SAGEConv(hidden_channels[1], hidden_channels[2])
        elif conv_type == 'graph':
            self.conv1 = GraphConv(input_size, hidden_channels[0])
            self.conv2 = GraphConv(hidden_channels[0], hidden_channels[1])
            self.conv3 = GraphConv(hidden_channels[1], hidden_channels[2])
        elif conv_type == 'gat':
            self.conv1 = GATv2Conv(input_size, hidden_channels[0])
            self.conv2 = GATv2Conv(hidden_channels[0], hidden_channels[1])
            self.conv3 = GATv2Conv(hidden_channels[1], hidden_channels[2])
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels[0])
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels[1])
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels[2])
        
        # Feature attention mechanism for combining pooled features
        self.attn = torch.nn.Linear(hidden_channels[2] * 3, 3)
        
        # Multiple fully connected layers for better representation
        self.fc1 = torch.nn.Linear(hidden_channels[2], hidden_channels[2])
        self.fc2 = torch.nn.Linear(hidden_channels[2], hidden_channels[2] // 2)
        self.output_layer = torch.nn.Linear(hidden_channels[2] // 2, output_size)
        
        # Dropout layers with different rates for different depths
        self.dropout1 = torch.nn.Dropout(dropout_rate * 0.7)  # Less dropout in early layers
        self.dropout2 = torch.nn.Dropout(dropout_rate * 0.8)
        self.dropout3 = torch.nn.Dropout(dropout_rate)
        
        # Parameter initialization
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters for better training with small datasets"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GNN layer
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.1)  # Leaky ReLU often works better than ELU
        x1 = self.dropout1(x1)
        
        # Second GNN layer
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.1)
        x2 = self.dropout2(x2)
        
        # Residual connection
        if self.residual and x1.size(-1) == x2.size(-1):
            x2 = x2 + x1
        
        # Third GNN layer
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.1)
        
        # Another residual connection
        if self.residual and x2.size(-1) == x3.size(-1):
            x3 = x3 + x2
        
        # Multiple pooling strategies
        x_mean = global_mean_pool(x3, batch)
        x_max = global_max_pool(x3, batch)
        x_sum = global_add_pool(x3, batch)
        
        # Concatenate pooled features
        pooled = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Apply feature attention to weight different pooling strategies
        attn_weights = F.softmax(self.attn(pooled), dim=1)
        x = attn_weights[:, 0].unsqueeze(-1) * x_mean + \
            attn_weights[:, 1].unsqueeze(-1) * x_max + \
            attn_weights[:, 2].unsqueeze(-1) * x_sum
        
        # MLP layers with dropout
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        
        # Output layer
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        
        return x
