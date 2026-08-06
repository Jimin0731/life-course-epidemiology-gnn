import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool

class EconomicGNN(nn.Module):
    """Graph Neural Network for Economic Data Analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
                 num_heads: int = 4, dropout: float = 0.1, use_edge_weight: bool = True):
        super(EconomicGNN, self).__init__()
        
        self.use_edge_weight = use_edge_weight
        
        if use_edge_weight:
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads,
                         dropout=dropout, concat=False) 
            self.conv2 = GATConv(hidden_dim * 1, hidden_dim, heads=1, # hidden_dim * num_heads -> hidden_dim * 1
                         dropout=dropout, concat=False) 

        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Add LayerNorm layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        
        attention_weights_1 = None
        attention_weights_2 = None

        if self.use_edge_weight:
            # GCNConv path: edge weights optional, no attention returned
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
        else:
            # GATConv path: capture attention weights
            x, attention_weights_1 = self.conv1(x, edge_index, return_attention_weights=True)
        
        # After passing conv1, apply LayerNorm
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if self.use_edge_weight:
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
        else:
            x, attention_weights_2 = self.conv2(x, edge_index, return_attention_weights=True)
            
        # After passing conv2, apply LayerNorm
        x = self.norm2(x)
        x = self.relu(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, (attention_weights_1, attention_weights_2)

