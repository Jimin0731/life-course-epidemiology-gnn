"""
Simple test script to verify temporal GNN models work correctly
"""
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# Import temporal encoders (package-relative)
from lce_gnn.temporal_encoders import (
    LSTMEncoder,
    GRUEncoder,
    Conv1DEncoder,
    TransformerEncoder,
)

# Import temporal-aware GNN models (package-relative)
from lce_gnn.models_gnn_temporal import EconomicGNN
from lce_gnn.hybridgnn_temporal import HybridGNN
from lce_gnn.models_weightedgat_temporal import WeightedGATGNN


def test_temporal_encoders():
    """Test all temporal encoders"""
    print("\n" + "="*60)
    print("Testing Temporal Encoders")
    print("="*60)
    
    batch_size = 8
    seq_len = 41
    input_dim = 1
    hidden_dim = 64
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    encoders = {
        'LSTM': LSTMEncoder(input_dim, hidden_dim),
        'GRU': GRUEncoder(input_dim, hidden_dim),
        'CNN': Conv1DEncoder(input_dim, hidden_dim),
        'Transformer': TransformerEncoder(input_dim, hidden_dim)
    }
    
    for name, encoder in encoders.items():
        output = encoder(x)
        print(f"{name:12} - Input: {x.shape} -> Output: {output.shape} "
              f"(expected dim: {encoder.output_dim})")
        assert output.shape == (batch_size, encoder.output_dim), \
            f"{name} output shape mismatch!"
    
    print("✅ All temporal encoders passed!")


def test_temporal_gnn_models():
    """Test all GNN models with temporal encoding"""
    print("\n" + "="*60)
    print("Testing Temporal GNN Models")
    print("="*60)
    
    # Create dummy graph data
    num_nodes = 10
    seq_len = 41
    input_dim = 1
    batch_size = 1
    
    # Node features with temporal dimension
    x = torch.randn(num_nodes, seq_len, input_dim)
    
    # Simple edge index (chain graph)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    models = {
        'EconomicGNN_LSTM': (EconomicGNN, 'lstm'),
        'EconomicGNN_GRU': (EconomicGNN, 'gru'),
        'EconomicGNN_CNN': (EconomicGNN, 'cnn'),
        'HybridGNN_LSTM': (HybridGNN, 'lstm'),
        'HybridGNN_GRU': (HybridGNN, 'gru'),
        'WeightedGAT_LSTM': (WeightedGATGNN, 'lstm'),
        'WeightedGAT_CNN': (WeightedGATGNN, 'cnn'),
    }
    
    for name, (model_class, temporal_encoder) in models.items():
        model = model_class(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=64,
            output_dim=1,
            temporal_encoder=temporal_encoder
        )
        
        # Forward pass
        out = model(x, edge_index)
        
        # Handle different return types
        if isinstance(out, tuple):
            output = out[0]
            print(f"{name:22} - Input: {x.shape} -> Output: {output.shape} "
                  f"(with attention weights)")
        else:
            output = out
            print(f"{name:22} - Input: {x.shape} -> Output: {output.shape}")
        
        assert output.shape == (num_nodes, 1), f"{name} output shape mismatch!"
    
    print("✅ All temporal GNN models passed!")


def test_batched_data():
    """Test with batched graph data (realistic scenario)"""
    print("\n" + "="*60)
    print("Testing Batched Data Processing")
    print("="*60)
    
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool
    
    # Create multiple graphs
    num_graphs = 4
    num_nodes = 10
    seq_len = 41
    input_dim = 1
    
    # Simple chain graph structure
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # Create data list
    data_list = []
    for i in range(num_graphs):
        x = torch.randn(num_nodes, seq_len, input_dim)
        y = torch.tensor([float(i % 2)], dtype=torch.float)  # Binary label
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    # Create DataLoader
    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    
    # Test with EconomicGNN + LSTM
    model = EconomicGNN(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=64,
        output_dim=1,
        temporal_encoder='lstm'
    )
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Forward pass
            out = model(batch.x, batch.edge_index)
            
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
            
            # Graph-level pooling
            pred_graph = global_mean_pool(pred, batch.batch)
            
            print(f"Batch {batch_idx}: "
                  f"Input shape: {batch.x.shape}, "
                  f"Node predictions: {pred.shape}, "
                  f"Graph predictions: {pred_graph.shape}, "
                  f"True labels: {batch.y.shape}")
            
            assert pred_graph.shape[0] == batch.y.shape[0], \
                "Batch size mismatch!"
    
    print("✅ Batched data processing passed!")


def test_different_input_dimensions():
    """Test models with different input dimensions (multivariate time series)"""
    print("\n" + "="*60)
    print("Testing Different Input Dimensions")
    print("="*60)
    
    num_nodes = 10
    seq_len = 41
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5]
    ], dtype=torch.long)
    
    for input_dim in [1, 3, 5]:
        x = torch.randn(num_nodes, seq_len, input_dim)
        
        model = EconomicGNN(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=64,
            output_dim=1,
            temporal_encoder='lstm'
        )
        
        out = model(x, edge_index)
        output = out[0] if isinstance(out, tuple) else out
        
        print(f"Input dim: {input_dim} - Input: {x.shape} -> Output: {output.shape}")
        assert output.shape == (num_nodes, 1), f"Output shape mismatch for input_dim={input_dim}!"
    
    print("✅ Different input dimensions passed!")


if __name__ == "__main__":
    print("\n" + "🧪 Starting Temporal GNN Tests" + "\n")
    
    try:
        test_temporal_encoders()
        test_temporal_gnn_models()
        test_batched_data()
        test_different_input_dimensions()
        
        print("\n" + "="*60)
        print("🎉 All tests passed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
