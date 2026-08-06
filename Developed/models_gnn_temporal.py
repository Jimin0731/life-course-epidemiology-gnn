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


class EconomicGNN(nn.Module):
    """Graph Neural Network for Economic Data Analysis with Temporal Encoding"""
    
    def __init__(self, input_dim: int, seq_len: int = 41, hidden_dim: int = 64, 
                 output_dim: int = 32, num_heads: int = 4, dropout: float = 0.1, 
                 use_edge_weight: bool = True, temporal_encoder: str = 'lstm'):
        """
        Args:
            input_dim: Feature dimension per time step (e.g., 1 for univariate series)
            seq_len: Length of time series (e.g., 41)
            hidden_dim: Hidden dimension for both temporal and graph layers
            output_dim: Final output dimension
            num_heads: Number of attention heads for GAT
            dropout: Dropout rate
            use_edge_weight: Whether to use edge weights in GCN
            temporal_encoder: Type of temporal encoder ('lstm', 'gru', 'cnn', 'transformer')
        """
        super(EconomicGNN, self).__init__()
        
        self.use_edge_weight = use_edge_weight
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
        
        # ===== Graph Neural Network =====
        if use_edge_weight:
            self.conv1 = GCNConv(temporal_output_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = GATConv(temporal_output_dim, hidden_dim, heads=num_heads,
                         dropout=dropout, concat=False) 
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1,
                         dropout=dropout, concat=False) 

        # ===== Feed-Forward Layers =====
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, seq_len, input_dim] for single graph
               or [batch_size * num_nodes, seq_len, input_dim] for batched graphs
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        
        Returns:
            Node embeddings and attention weights (if applicable)
        """
        # ===== Step 1: Temporal Encoding =====
        # Encode time series for each node
        x_encoded = self.temporal_encoder(x)  # [num_nodes, temporal_output_dim]
        
        # ===== Step 2: Graph Convolution =====
        attention_weights_1 = None
        attention_weights_2 = None

        if self.use_edge_weight:
            # GCNConv path
            x_encoded = self.conv1(x_encoded, edge_index, edge_weight=edge_weight)
        else:
            # GATConv path: capture attention weights
            x_encoded, attention_weights_1 = self.conv1(x_encoded, edge_index, return_attention_weights=True)
        
        x_encoded = self.norm1(x_encoded)
        x_encoded = self.relu(x_encoded)
        x_encoded = self.dropout(x_encoded)

        if self.use_edge_weight:
            x_encoded = self.conv2(x_encoded, edge_index, edge_weight=edge_weight)
        else:
            x_encoded, attention_weights_2 = self.conv2(x_encoded, edge_index, return_attention_weights=True)
            
        x_encoded = self.norm2(x_encoded)
        x_encoded = self.relu(x_encoded)
        
        # ===== Step 3: Final Prediction Layers =====
        x_encoded = self.fc1(x_encoded)
        x_encoded = self.relu(x_encoded)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.fc2(x_encoded)
        
        return x_encoded, (attention_weights_1, attention_weights_2)
