import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool

class HybridGNN(nn.Module):
    """Hybrid model combining GCN and GAT"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
                 num_heads: int = 4, dropout: float = 0.1):
        super(HybridGNN, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.gat = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.input_proj(x)
        x = self.relu(x)
        
        h_gcn = self.gcn1(x, edge_index, edge_weight=edge_weight)
        h_gcn = self.relu(h_gcn)
        h_gcn = self.dropout(h_gcn)
        h_gcn = self.gcn2(h_gcn, edge_index, edge_weight=edge_weight)
        h_gcn = self.layer_norm1(h_gcn)
        
        h_gat = self.gat(x, edge_index)
        h_gat = self.relu(h_gat)
        h_gat = self.layer_norm2(h_gat)
        
        h_combined = torch.cat([h_gcn, h_gat], dim=1)
        h = self.fusion(h_combined)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        
        return h