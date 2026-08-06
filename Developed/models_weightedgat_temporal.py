import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from .temporal_encoders1 import (
    LSTMEncoder,
    GRUEncoder,
    Conv1DEncoder,
    TransformerEncoder,
)


class WeightedGATGNN(nn.Module):
    """GAT with edge weight incorporation and Temporal Encoding"""
    
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
        super(WeightedGATGNN, self).__init__()
        
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
        
        # ===== GAT Layers =====
        # edge_weight is passed as 1-d edge attribute (edge_dim=1).
        self.gat1 = GATv2Conv(
            temporal_output_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,
        )
        self.gat2 = GATv2Conv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            edge_dim=1,
        )
        
        # ===== Feed-Forward Layers =====
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, seq_len, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        
        Returns:
            Node embeddings and attention weights
        """
        # ===== Step 1: Temporal Encoding =====
        x = self.temporal_encoder(x)  # [num_nodes, temporal_output_dim]
        
        # ===== Step 2: GAT Layers =====
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        edge_attr = edge_weight.view(-1, 1)

        x, attention_weights_1 = self.gat1(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
        )
        x = self.relu(x)
        x = self.dropout(x)
        
        x, attention_weights_2 = self.gat2(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
        )
        x = self.relu(x)
        
        # ===== Step 3: Final Prediction =====
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, (attention_weights_1, attention_weights_2)
