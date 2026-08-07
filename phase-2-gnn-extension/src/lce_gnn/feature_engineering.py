"""
Feature Engineering for GNN - Ready to Use
27 features로 XGBoost 경쟁

Usage:
    from feature_engineering import SuperRichFeatureEngineer
    X_enriched = SuperRichFeatureEngineer.extract_features(X_tensor)
"""

import torch
import torch.nn as nn

class SuperRichFeatureEngineer:
    """Extract engineered temporal and cross-node features for comparison experiments."""
    
    @staticmethod
    def extract_features(X_tensor):
        """
        Extract 27 features from time series data
        
        Args:
            X_tensor: [n_samples, n_nodes, seq_len]
        
        Returns:
            X_enriched: [n_samples, n_nodes, seq_len, 27]
        """
        n_samples, n_nodes, seq_len = X_tensor.shape
        device = X_tensor.device
        features = []
        
        print(f"  🔧 Feature Engineering: ", end="")
        
        # ===== Group 1: Basic (2 features) =====
        # 1. Original value
        features.append(X_tensor.unsqueeze(-1))
        
        # 2. Standardized
        mean = X_tensor.mean(dim=2, keepdim=True)
        std = X_tensor.std(dim=2, keepdim=True) + 1e-6
        standardized = (X_tensor - mean) / std
        features.append(standardized.unsqueeze(-1))
        
        # ===== Group 2: Moving Averages (4 features) =====
        for window in [3, 5, 7, 10]:
            ma = SuperRichFeatureEngineer._moving_average(X_tensor, window)
            features.append(ma.unsqueeze(-1))
        
        # ===== Group 3: Trends (3 features) =====
        # 3. 1st difference (velocity)
        diff1 = torch.diff(X_tensor, dim=2, prepend=X_tensor[:, :, :1])
        features.append(diff1.unsqueeze(-1))
        
        # 4. 2nd difference (acceleration)
        diff2 = torch.diff(diff1, dim=2, prepend=diff1[:, :, :1])
        features.append(diff2.unsqueeze(-1))
        
        # 5. Momentum (diff * value)
        momentum = diff1 * X_tensor
        features.append(momentum.unsqueeze(-1))
        
        # ===== Group 4: Cumulative (1 feature) =====
        # 6. Cumulative mean
        cumsum = torch.cumsum(X_tensor, dim=2)
        positions = torch.arange(1, seq_len+1, device=device).float()
        cumsum_norm = cumsum / positions.view(1, 1, -1)
        features.append(cumsum_norm.unsqueeze(-1))
        
        # ===== Group 5: Rolling Statistics (8 features) =====
        for window in [5, 10]:
            for stat in ['mean', 'std', 'min', 'max']:
                roll = SuperRichFeatureEngineer._rolling_stat(X_tensor, window, stat)
                features.append(roll.unsqueeze(-1))
        
        # ===== Group 6: Normalization (2 features) =====
        # 7. Z-score
        z_score = (X_tensor - mean) / std
        features.append(z_score.unsqueeze(-1))
        
        # 8. Min-max normalization
        min_val = X_tensor.min(dim=2, keepdim=True)[0]
        max_val = X_tensor.max(dim=2, keepdim=True)[0]
        minmax = (X_tensor - min_val) / (max_val - min_val + 1e-6)
        features.append(minmax.unsqueeze(-1))
        
        # ===== Group 7: Position (1 feature) =====
        # 9. Temporal position encoding
        position = torch.arange(seq_len, device=device).float() / seq_len
        position = position.view(1, 1, -1, 1).expand(n_samples, n_nodes, -1, -1)
        features.append(position)
        
        # ===== Group 8: Cross-Node Features (3 features) =====
        # Cross-node context for graph-based models.
        if n_nodes > 1:
            # 10. Deviation from graph mean
            graph_mean = X_tensor.mean(dim=1, keepdim=True)
            deviation = X_tensor - graph_mean
            features.append(deviation.unsqueeze(-1))
            
            # 11. Distance to max node
            max_node = X_tensor.max(dim=1, keepdim=True)[0]
            dist_max = X_tensor - max_node
            features.append(dist_max.unsqueeze(-1))
            
            # 12. Distance to min node
            min_node = X_tensor.min(dim=1, keepdim=True)[0]
            dist_min = X_tensor - min_node
            features.append(dist_min.unsqueeze(-1))
        else:
            # Single node: add dummy features
            features.extend([
                torch.zeros_like(X_tensor).unsqueeze(-1),
                torch.zeros_like(X_tensor).unsqueeze(-1),
                torch.zeros_like(X_tensor).unsqueeze(-1)
            ])
        
        # Concatenate all features
        X_enriched = torch.cat(features, dim=-1)
        
        print(f"{X_tensor.shape} → {X_enriched.shape} ({X_enriched.shape[-1]} features)")
        
        return X_enriched
    
    @staticmethod
    def _moving_average(x, window):
        """Fast moving average using conv1d"""
        # x: [n_samples, n_nodes, seq_len]
        n_samples, n_nodes, seq_len = x.shape
        padding = window // 2
        x_pad = torch.nn.functional.pad(x, (padding, padding), mode='replicate')

        # avg_pool1d expects [N, C, L]. Pool over temporal axis L only.
        x_pool = x_pad.reshape(n_samples * n_nodes, 1, x_pad.shape[-1])
        ma = torch.nn.functional.avg_pool1d(
            x_pool, kernel_size=window, stride=1, padding=0
        )

        ma = ma.reshape(n_samples, n_nodes, -1)

        # Keep temporal length equal to input when even window sizes are used.
        if ma.shape[-1] > seq_len:
            ma = ma[:, :, :seq_len]
        elif ma.shape[-1] < seq_len:
            ma = torch.nn.functional.pad(ma, (0, seq_len - ma.shape[-1]), mode='replicate')

        return ma
    
    @staticmethod
    def _rolling_stat(x, window, stat='mean'):
        """Compute rolling statistics"""
        # x: [n_samples, n_nodes, seq_len]
        n_samples, n_nodes, seq_len = x.shape
        result = torch.zeros_like(x)
        
        padding = window // 2
        x_pad = torch.nn.functional.pad(x, (padding, padding), mode='replicate')
        
        for i in range(seq_len):
            window_data = x_pad[:, :, i:i+window]
            
            if stat == 'mean':
                result[:, :, i] = window_data.mean(dim=2)
            elif stat == 'std':
                result[:, :, i] = window_data.std(dim=2)
            elif stat == 'min':
                result[:, :, i] = window_data.min(dim=2)[0]
            elif stat == 'max':
                result[:, :, i] = window_data.max(dim=2)[0]
        
        return result


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Feature Engineering Test")
    print("=" * 70)
    
    # Test with sample data
    n_samples = 100
    n_nodes = 20
    seq_len = 41
    
    X_test = torch.randn(n_samples, n_nodes, seq_len)
    
    print(f"\nInput:")
    print(f"  Shape: {X_test.shape}")
    print(f"  Size:  {X_test.numel() * 4 / 1024:.2f} KB")
    
    # Extract features
    import time
    start = time.time()
    
    X_enriched = SuperRichFeatureEngineer.extract_features(X_test)
    
    elapsed = time.time() - start
    
    print(f"\nOutput:")
    print(f"  Shape: {X_enriched.shape}")
    print(f"  Size:  {X_enriched.numel() * 4 / 1024:.2f} KB")
    print(f"  Time:  {elapsed:.3f} seconds")
    
    print(f"\n✅ Feature Groups:")
    print(f"  1. Basic (2): Original, Standardized")
    print(f"  2. Moving Averages (4): MA(3,5,7,10)")
    print(f"  3. Trends (3): Diff1, Diff2, Momentum")
    print(f"  4. Cumulative (1): Cumsum normalized")
    print(f"  5. Rolling Stats (8): Mean/Std/Min/Max for W=5,10")
    print(f"  6. Normalization (2): Z-score, MinMax")
    print(f"  7. Position (1): Temporal encoding")
    print(f"  8. Cross-Node (3): Graph-mean, Max-dist, Min-dist")
    print(f"\n  Total: {X_enriched.shape[-1]} features")
    
    print("\n✅ Ready to use!")
    print("\nNext steps:")
    print("  1. Save this file as 'feature_engineering.py'")
    print("  2. In Run_all_n_temporal.py:")
    print("     from feature_engineering import SuperRichFeatureEngineer")
    print("     X_enriched = SuperRichFeatureEngineer.extract_features(X_tensor)")
    print("  3. Change input_dim=1 to input_dim=27 in model initialization")
    print("  4. Evaluate against held-out data; this self-test makes no performance claim.")
