import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """LSTM-based temporal encoder for time series data"""
    
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
        
        # Output dimension is hidden_dim * 2 if bidirectional, else hidden_dim
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [num_nodes, seq_len, input_dim]
        Returns:
            encoded: [batch_size, output_dim] or [num_nodes, output_dim]
        """
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            # h_n shape: [num_layers * num_directions, batch, hidden_dim]
            h_forward = h_n[-2, :, :]  # Last layer forward
            h_backward = h_n[-1, :, :]  # Last layer backward
            encoded = torch.cat([h_forward, h_backward], dim=1)
        else:
            # Use only the last layer's hidden state
            encoded = h_n[-1, :, :]
        
        return encoded


class GRUEncoder(nn.Module):
    """GRU-based temporal encoder for time series data"""
    
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
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [num_nodes, seq_len, input_dim]
        Returns:
            encoded: [batch_size, output_dim] or [num_nodes, output_dim]
        """
        gru_out, h_n = self.gru(x)
        
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            encoded = torch.cat([h_forward, h_backward], dim=1)
        else:
            encoded = h_n[-1, :, :]
        
        return encoded


class Conv1DEncoder(nn.Module):
    """1D CNN-based temporal encoder for time series data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.1):
        super(Conv1DEncoder, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        
        # Multiple parallel convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Output dimension is hidden_dim * number of kernels
        self.output_dim = hidden_dim * len(kernel_sizes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [num_nodes, seq_len, input_dim]
        Returns:
            encoded: [batch_size, output_dim] or [num_nodes, output_dim]
        """
        # Conv1d expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x)  # [batch, hidden_dim, seq_len]
            conv_out = self.relu(conv_out)
            # Global max pooling over time dimension
            pooled = torch.max(conv_out, dim=2)[0]  # [batch, hidden_dim]
            conv_outputs.append(pooled)
        
        # Concatenate all kernel outputs
        encoded = torch.cat(conv_outputs, dim=1)  # [batch, output_dim]
        encoded = self.layer_norm(encoded)
        encoded = self.dropout(encoded)
        
        return encoded


class TransformerEncoder(nn.Module):
    """Transformer-based temporal encoder for time series data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = hidden_dim
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [num_nodes, seq_len, input_dim]
        Returns:
            encoded: [batch_size, output_dim] or [num_nodes, output_dim]
        """
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]
        x = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        # Use mean pooling over time dimension
        encoded = torch.mean(x, dim=1)  # [batch, hidden_dim]
        
        return encoded