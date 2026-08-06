import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
import numpy as np


# ==============================================================================
# Positional Encoding
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# Cross-Attention: Temporal ↔ Graph 정보를 동시에 학습
# ==============================================================================
class TemporalGraphCrossAttention(nn.Module):
    """
    핵심 아이디어:
    - 각 시간 스텝에서 이웃 노드 정보를 함께 참조 (Cross-Attention)
    - Temporal sequence가 Graph 구조를 인식하며 학습
    - Graph가 Temporal 패턴을 인식하며 학습
    
    구조:
    1. Temporal Self-Attention: 시계열 내부 패턴 학습
    2. Graph Cross-Attention: 이웃 노드의 temporal 패턴을 query로 활용
    3. Fusion: 두 정보를 합쳐서 최종 표현 생성
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, seq_len: int = 41):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = hidden_dim

        # --- Step 1: Input projection ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=seq_len + 10)

        # --- Step 2: Temporal Self-Attention (각 노드 독립적으로) ---
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm: 더 안정적
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)

        # --- Step 3: Graph Cross-Attention ---
        # 이웃 노드의 temporal 표현을 참조하는 cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # --- Step 4: GAT for graph message passing ---
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads,
                               dropout=dropout, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1,
                               dropout=dropout, concat=False)

        # --- Step 5: Fusion ---
        # Temporal 표현 + Graph 표현을 합쳐서 최종 출력
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Attention pooling (시간 축 압축)
        self.temporal_pool = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, seq_len, input_dim]
            edge_index: [2, num_edges]
        Returns:
            out: [num_nodes, hidden_dim]
            attention_weights: tuple
        """
        num_nodes, seq_len, input_dim = x.shape

        # === Step 1: Input projection + Positional Encoding ===
        x = self.input_proj(x)          # [num_nodes, seq_len, hidden_dim]
        x = self.pos_encoder(x)         # [num_nodes, seq_len, hidden_dim]

        # === Step 2: Temporal Self-Attention ===
        # 각 노드의 시계열 패턴을 독립적으로 학습
        h_temporal = self.temporal_encoder(x)  # [num_nodes, seq_len, hidden_dim]

        # === Step 3: Graph-aware Cross-Attention ===
        # 핵심: 이웃 노드들의 temporal 표현을 참조
        h_cross_list = []
        for node_i in range(num_nodes):
            # 현재 노드의 temporal sequence
            query = h_temporal[node_i:node_i+1]  # [1, seq_len, hidden_dim]

            # 이웃 노드 찾기
            neighbors = edge_index[1][edge_index[0] == node_i]
            if len(neighbors) == 0:
                h_cross_list.append(query.squeeze(0))
                continue

            # 이웃 노드들의 temporal 표현을 key/value로 사용
            neighbor_feats = h_temporal[neighbors]  # [n_neighbors, seq_len, hidden_dim]

            # neighbor들을 하나의 sequence로 flatten
            # [1, n_neighbors * seq_len, hidden_dim]
            kv = neighbor_feats.reshape(1, -1, self.hidden_dim)

            # Cross-attention: 현재 노드(query) ↔ 이웃 노드(key/value)
            attended, _ = self.cross_attn(query, kv, kv)  # [1, seq_len, hidden_dim]
            attended = self.cross_attn_norm(attended + query)  # residual
            h_cross_list.append(attended.squeeze(0))

        h_cross = torch.stack(h_cross_list, dim=0)  # [num_nodes, seq_len, hidden_dim]

        # === Step 4: Attention Pooling (시간 축 압축) ===
        # Temporal 표현
        attn_w = torch.softmax(self.temporal_pool(h_temporal), dim=1)  # [num_nodes, seq_len, 1]
        h_temp_pooled = (h_temporal * attn_w).sum(dim=1)               # [num_nodes, hidden_dim]
        h_temp_pooled = self.norm1(h_temp_pooled)

        # Cross-attention 표현
        attn_w2 = torch.softmax(self.temporal_pool(h_cross), dim=1)
        h_cross_pooled = (h_cross * attn_w2).sum(dim=1)                # [num_nodes, hidden_dim]
        h_cross_pooled = self.norm2(h_cross_pooled)

        # === Step 5: GAT Graph Message Passing ===
        # Cross-attention으로 이미 그래프 정보가 들어갔지만
        # GAT으로 한 번 더 명시적 그래프 학습
        h_graph, attn1 = self.gat1(h_cross_pooled, edge_index,
                                    return_attention_weights=True)
        h_graph = F.gelu(h_graph)
        h_graph = self.dropout(h_graph)
        h_graph, attn2 = self.gat2(h_graph, edge_index,
                                    return_attention_weights=True)

        # === Step 6: Fusion ===
        # Temporal 정보 + Graph 정보 합치기
        h_combined = torch.cat([h_temp_pooled, h_graph], dim=1)  # [num_nodes, hidden_dim*2]
        out = self.fusion(h_combined)                              # [num_nodes, hidden_dim]

        return out, (attn1, attn2)


# ==============================================================================
# SpatioTemporalGNN: 최종 모델
# ==============================================================================
class SpatioTemporalGNN(nn.Module):
    """
    Cross-Attention 기반 Temporal-Graph 동시 학습 모델

    기존 모델과의 차이:
    - 기존: Temporal 인코딩 완료 → GNN (순차적, 정보 손실)
    - 신규: Temporal ↔ Graph 동시 학습 (Cross-Attention으로 상호 참조)
    """

    def __init__(self, input_dim: int, seq_len: int = 41,
                 hidden_dim: int = 64, output_dim: int = 1,
                 num_heads: int = 4, dropout: float = 0.1,
                 temporal_encoder: str = 'spatiotemporal'):
        super().__init__()

        self.core = TemporalGraphCrossAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
            seq_len=seq_len
        )

        # Final prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.output_dim = hidden_dim

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [num_nodes, seq_len, input_dim]
            edge_index: [2, num_edges]
        Returns:
            out: [num_nodes, output_dim]
            attention_weights: tuple
        """
        h, attention_weights = self.core(x, edge_index)
        out = self.head(h)
        return out, attention_weights