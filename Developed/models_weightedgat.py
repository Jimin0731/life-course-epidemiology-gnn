import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool

class WeightedGATGNN(nn.Module):
    """GAT with edge weight incorporation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
                 num_heads: int = 4, dropout: float = 0.1):
        super(WeightedGATGNN, self).__init__()
        
        self.gat1 = GATv2Conv(
            input_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1
        )
        self.gat2 = GATv2Conv(
            hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=1
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        edge_attr = edge_weight.view(-1, 1)

        x, attention_weights_1 = self.gat1(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        x = self.relu(x)
        x = self.dropout(x)
        
        x, attention_weights_2 = self.gat2(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        x = self.relu(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, (attention_weights_1, attention_weights_2)
