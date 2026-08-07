import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """✅ IMPROVED: LSTM with residual projection for better gradient flow"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.1, bidirectional: bool = True):
        super(LSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # ✅ NEW: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            encoded: [batch_size, output_dim]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            encoded = torch.cat([h_forward, h_backward], dim=1)
        else:
            encoded = h_n[-1, :, :]
        
        # ✅ Apply normalization
        encoded = self.layer_norm(encoded)
        
        return encoded


class GRUEncoder(nn.Module):
    """✅ IMPROVED: GRU with layer normalization"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.1, bidirectional: bool = True):
        super(GRUEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # ✅ NEW: Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            encoded = torch.cat([h_forward, h_backward], dim=1)
        else:
            encoded = h_n[-1, :, :]
        
        # ✅ Apply normalization
        encoded = self.layer_norm(encoded)
        
        return encoded


class Conv1DEncoder(nn.Module):
    """✅ IMPROVED: Multi-scale CNN with attention pooling"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.1):
        super(Conv1DEncoder, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        
        # Multiple parallel convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # ✅ NEW: Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in kernel_sizes
        ])
        
        self.output_dim = hidden_dim * len(kernel_sizes)
        
        # ✅ NEW: Attention weights for different scales
        self.scale_attention = nn.Linear(len(kernel_sizes), len(kernel_sizes))
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            encoded: [batch_size, output_dim]
        """
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        
        conv_outputs = []
        scale_features = []
        
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = conv(x)  # [batch, hidden_dim, seq_len]
            conv_out = bn(conv_out)  # ✅ Batch norm
            conv_out = self.relu(conv_out)
            
            # Global max pooling
            pooled = torch.max(conv_out, dim=2)[0]  # [batch, hidden_dim]
            conv_outputs.append(pooled)
            
            # Average feature for attention
            scale_features.append(torch.mean(pooled, dim=1, keepdim=True))
        
        # ✅ NEW: Learn importance of different scales
        scale_features = torch.cat(scale_features, dim=1)  # [batch, num_scales]
        scale_weights = torch.softmax(self.scale_attention(scale_features), dim=1)  # [batch, num_scales]
        
        # Weight each scale's contribution
        weighted_outputs = []
        for i, pooled in enumerate(conv_outputs):
            weighted = pooled * scale_weights[:, i:i+1]  # [batch, hidden_dim]
            weighted_outputs.append(weighted)
        
        encoded = torch.cat(weighted_outputs, dim=1)  # [batch, output_dim]
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)
        
        return encoded


class TransformerEncoder(nn.Module):
    """✅ IMPROVED: Transformer with positional encoding"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # ✅ NEW: Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=100)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ✅ NEW: Learnable aggregation (better than mean)
        self.attention_pool = nn.Linear(hidden_dim, 1)
        
        self.output_dim = hidden_dim
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            encoded: [batch_size, output_dim]
        """
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]
        x = self.pos_encoder(x)  # ✅ Add positional encoding
        x = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        # ✅ NEW: Attention-based pooling instead of mean
        attention_weights = torch.softmax(self.attention_pool(x), dim=1)  # [batch, seq_len, 1]
        encoded = torch.sum(x * attention_weights, dim=1)  # [batch, hidden_dim]
        
        return encoded


# ✅ NEW: Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


import numpy as np  # For positional encoding