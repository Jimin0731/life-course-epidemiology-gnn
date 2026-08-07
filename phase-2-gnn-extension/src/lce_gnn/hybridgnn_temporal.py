import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
# Local relative import to keep package self-contained
from .temporal_encoders import (
    LSTMEncoder,
    GRUEncoder,
    Conv1DEncoder,
    TransformerEncoder,
)


class HybridGNN(nn.Module):
    """Hybrid model combining GCN and GAT with Temporal Encoding"""
    
    def __init__(self, input_dim: int, seq_len: int = 41, hidden_dim: int = 64, 
                 output_dim: int = 32, num_heads: int = 4, dropout: float = 0.1,
                 temporal_encoder: str = 'lstm'):
        """
        Args:
            input_dim: Feature dimension per time step
            seq_len: Length of time series
            hidden_dim: Hidden dimension for both temporal and graph layers
            output_dim: Final output dimension
            num_heads: Number of attention heads for GAT
            dropout: Dropout rate
            temporal_encoder: Type of temporal encoder ('lstm', 'gru', 'cnn', 'transformer')
        """
        super(HybridGNN, self).__init__()
        
        self.temporal_encoder_type = temporal_encoder
        
        # ===== Temporal Encoder =====
        if temporal_encoder == 'lstm':
            self.temporal_encoder = LSTMEncoder(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_layers=2, 
                dropout=dropout,
                bidirectional=True
            )
        elif temporal_encoder == 'gru':
            self.temporal_encoder = GRUEncoder(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_layers=2, 
                dropout=dropout,
                bidirectional=True
            )
        elif temporal_encoder == 'cnn':
            self.temporal_encoder = Conv1DEncoder(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                kernel_sizes=[3, 5, 7],
                dropout=dropout
            )
        elif temporal_encoder == 'transformer':
            self.temporal_encoder = TransformerEncoder(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_heads=num_heads,
                num_layers=2,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown temporal encoder: {temporal_encoder}")
        
        # Get the output dimension from temporal encoder
        temporal_output_dim = self.temporal_encoder.output_dim
        
        # ===== Input Projection =====
        self.input_proj = nn.Linear(temporal_output_dim, hidden_dim)
        
        # ===== GCN Branch =====
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # ===== GAT Branch =====
        self.gat = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        
        # ===== Fusion Layer =====
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # ===== Feed-Forward Layers =====
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, seq_len, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        
        Returns:
            Node embeddings
        """
        # ===== Step 1: Temporal Encoding =====
        x = self.temporal_encoder(x)  # [num_nodes, temporal_output_dim]
        
        # ===== Step 2: Input Projection =====
        x = self.input_proj(x)  # [num_nodes, hidden_dim]
        x = self.relu(x)
        
        # ===== Step 3: GCN Branch =====
        h_gcn = self.gcn1(x, edge_index, edge_weight=edge_weight)
        h_gcn = self.relu(h_gcn)
        h_gcn = self.dropout(h_gcn)
        h_gcn = self.gcn2(h_gcn, edge_index, edge_weight=edge_weight)
        h_gcn = self.layer_norm1(h_gcn)
        
        # ===== Step 4: GAT Branch =====
        h_gat = self.gat(x, edge_index)
        h_gat = self.relu(h_gat)
        h_gat = self.layer_norm2(h_gat)
        
        # ===== Step 5: Fusion =====
        h_combined = torch.cat([h_gcn, h_gat], dim=1)  # [num_nodes, hidden_dim * 2]
        h = self.fusion(h_combined)  # [num_nodes, hidden_dim]
        h = self.relu(h)
        h = self.dropout(h)
        
        # ===== Step 6: Final Prediction =====
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)  # [num_nodes, output_dim]
        
        return h
