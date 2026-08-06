"""
Node Importance Analysis via Attention Weights
- GAT/GATv2 attention weights를 추출하여 어떤 노드가 예측에 중요한지 분석
- Rule별, 모델별 노드 중요도 시각화
"""

import torch
import numpy as np
import pandas as pd
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, to_undirected

# ==============================================================================
# Import models
# ==============================================================================
try:
    from Data_simulation.models import EconomicGNN, WeightedGATGNN
    print("✅ Models loaded (Data_simulation.models)")
except ImportError:
    from models import EconomicGNN, WeightedGATGNN
    print("✅ Models loaded (models)")

try:
    from Data_simulation.feature_engineering import SuperRichFeatureEngineer
except ImportError:
    from feature_engineering import SuperRichFeatureEngineer

# ==============================================================================
# CONFIG
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAG_DIR = os.path.join(BASE_DIR, "data")
# Prefer v4 checkpoints when present, otherwise fall back to legacy folder.
_CKPT_V4_DIR = os.path.join(BASE_DIR, "models_ckpt_v4", "models_ckpt")
CKPT_DIR = _CKPT_V4_DIR if os.path.isdir(_CKPT_V4_DIR) else os.path.join(BASE_DIR, "models_ckpt")
OUTPUT_DIR = os.path.join(BASE_DIR, "node_importance_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DAG 노드 & 엣지
NODE_NAMES = [
    "sex", "ethnicity", "marriage", "education", "class",
    "admissions", "admissions2", "admissions3", "admissions4"
]
EDGES = [
    ("class", "admissions2"),
    ("education", "admissions2"),
    ("sex", "admissions3"),
    ("ethnicity", "admissions3"),
    ("admissions2", "admissions4"),
    ("admissions", "admissions4"),
]

# ✅ 실제 존재하는 checkpoint 기준으로 모델 & Rule 설정
# use_edge_weight / hidden_dim / input_dim은 checkpoint에서 자동 추론
MODELS_TO_ANALYZE = [
    ("WeightedGAT_LSTM", WeightedGATGNN, 'lstm', None),
]

N_SIZE = 10000
# ✅ 실제 파일 존재 확인된 Rule만 (Order1은 gru만 있어서 제외)
RULES_TO_ANALYZE = ["Period1", "Period2", "Repeats1", "Timing1", "Timing2"]

# ==============================================================================
# HELPER: Build graph
# ==============================================================================
def build_edge_index():
    name2idx = {n: i for i, n in enumerate(NODE_NAMES)}
    src = [name2idx[s] for s, d in EDGES]
    dst = [name2idx[d] for s, d in EDGES]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(NODE_NAMES))
    return edge_index, name2idx


def infer_checkpoint_family(state_dict):
    """Infer model family from checkpoint parameter names."""
    if any(k.startswith("gat1.") for k in state_dict.keys()):
        return "weightedgat"
    if any(k.startswith("conv1.") for k in state_dict.keys()):
        return "economic"
    return "unknown"


def requested_family_from_class(model_class):
    if model_class.__name__ == "WeightedGATGNN":
        return "weightedgat"
    if model_class.__name__ == "EconomicGNN":
        return "economic"
    return "unknown"


def infer_dims_from_checkpoint(state_dict, temporal_encoder):
    """Infer input/hidden/output dimensions from checkpoint weights."""
    if temporal_encoder == "cnn":
        key = "temporal_encoder.conv_layers.0.weight"
        if key not in state_dict:
            raise KeyError(f"Missing key: {key}")
        hidden_dim = state_dict[key].shape[0]
        input_dim = state_dict[key].shape[1]
    elif temporal_encoder == "lstm":
        key = "temporal_encoder.lstm.weight_ih_l0"
        if key not in state_dict:
            raise KeyError(f"Missing key: {key}")
        hidden_dim = state_dict[key].shape[0] // 4
        input_dim = state_dict[key].shape[1]
    elif temporal_encoder == "gru":
        key = "temporal_encoder.gru.weight_ih_l0"
        if key not in state_dict:
            raise KeyError(f"Missing key: {key}")
        hidden_dim = state_dict[key].shape[0] // 3
        input_dim = state_dict[key].shape[1]
    else:
        key = "temporal_encoder.input_projection.weight"
        if key not in state_dict:
            raise KeyError(f"Missing key: {key}")
        hidden_dim = state_dict[key].shape[0]
        input_dim = state_dict[key].shape[1]

    output_dim = state_dict.get("fc2.weight", torch.empty(1, 1)).shape[0]
    return input_dim, hidden_dim, output_dim


def infer_use_edge_weight_from_checkpoint(state_dict):
    """EconomicGNN path inference: True=GCN, False=GAT."""
    has_gat_attention_params = any(
        k.startswith("conv1.att") or k.startswith("conv1.att_")
        for k in state_dict.keys()
    )
    return not has_gat_attention_params


def patch_missing_weightedgat_edge_params(model, state_dict):
    """
    Some older WeightedGAT checkpoints were trained without edge_attr projection
    params (lin_edge). For compatibility, initialize missing lin_edge weights to 0.
    """
    model_sd = model.state_dict()
    patched_keys = []
    for key in ("gat1.lin_edge.weight", "gat2.lin_edge.weight"):
        if key not in state_dict and key in model_sd:
            state_dict[key] = torch.zeros_like(model_sd[key])
            patched_keys.append(key)
    return patched_keys

# ==============================================================================
# CORE: Extract attention weights from trained model
# ==============================================================================
def extract_attention_weights(model_name, model_class, temporal_encoder,
                               use_edge_weight, rule_name, n_size):
    """저장된 checkpoint에서 attention weights 추출"""

    ckpt_path = os.path.join(
        CKPT_DIR,
        f"best_model_N{n_size}_{rule_name}_{temporal_encoder}.pth"
    )
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️  Checkpoint not found: {ckpt_path}")
        return None

    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw_ckpt["state_dict"] if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt else raw_ckpt

    ckpt_family = infer_checkpoint_family(state_dict)
    expected_family = requested_family_from_class(model_class)
    if ckpt_family != expected_family:
        print(
            f"  ⚠️  Skip {rule_name}/{temporal_encoder}: checkpoint family={ckpt_family}, "
            f"requested={expected_family}"
        )
        return None

    try:
        input_dim, hidden_dim, output_dim = infer_dims_from_checkpoint(state_dict, temporal_encoder)
    except KeyError as e:
        print(f"  ⚠️  Skip {rule_name}/{temporal_encoder}: cannot infer dims ({e})")
        return None

    # Data
    data_folder = os.path.join(BASE_DIR, f"data_{n_size}")
    x_path = os.path.join(data_folder, "data_X.npy")
    y_path = os.path.join(data_folder, f"data_{rule_name}_Y.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"  ⚠️  Data not found for {rule_name}")
        return None

    X_np = np.load(x_path)[:500]   # 빠른 분석을 위해 500개만
    Y_np = np.load(y_path)[:500]

    X_tensor = torch.tensor(X_np, dtype=torch.float)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float)

    n_samples, n_nodes, seq_len = X_tensor.shape

    # Match input feature dimension to checkpoint architecture.
    if input_dim == 1:
        X_model = X_tensor.unsqueeze(-1)
    else:
        X_enriched = SuperRichFeatureEngineer.extract_features(X_tensor)
        if X_enriched.shape[-1] != input_dim:
            print(
                f"  ⚠️  Skip {rule_name}/{temporal_encoder}: checkpoint expects input_dim={input_dim}, "
                f"but engineered features={X_enriched.shape[-1]}"
            )
            return None
        X_model = X_enriched

    edge_index, _ = build_edge_index()

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ckpt_family == "economic":
        inferred_use_edge_weight = infer_use_edge_weight_from_checkpoint(state_dict)
        if use_edge_weight is not None and use_edge_weight != inferred_use_edge_weight:
            print(
                f"  ℹ️  use_edge_weight override ignored: arg={use_edge_weight}, "
                f"checkpoint={inferred_use_edge_weight}"
            )
        model = model_class(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            temporal_encoder=temporal_encoder,
            use_edge_weight=inferred_use_edge_weight,
        ).to(device)
    elif ckpt_family == "weightedgat":
        model = model_class(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            temporal_encoder=temporal_encoder,
        ).to(device)
    else:
        print(f"  ⚠️  Unsupported checkpoint family: {ckpt_family}")
        return None

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Backward compatibility for WeightedGAT checkpoints without lin_edge.
        if ckpt_family == "weightedgat" and "lin_edge.weight" in str(e):
            patched = patch_missing_weightedgat_edge_params(model, state_dict)
            if patched:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                extra_missing = [k for k in missing if k not in patched]
                if extra_missing or unexpected:
                    print(
                        f"  ⚠️  Failed to load checkpoint after patch. "
                        f"missing={extra_missing}, unexpected={unexpected}"
                    )
                    return None
                print(f"  ℹ️  Patched missing keys for compatibility: {patched}")
            else:
                print(f"  ⚠️  Failed to load checkpoint: {e}")
                return None
        else:
            print(f"  ⚠️  Failed to load checkpoint: {e}")
            return None

    model.eval()

    # Collect attention weights
    all_attn1 = []
    all_attn2 = []
    attn_edge_index = None

    with torch.no_grad():
        for i in range(min(100, n_samples)):
            x_i = X_model[i].to(device)
            ei = edge_index.to(device)

            out = model(x_i, ei)

            if isinstance(out, tuple) and len(out) == 2:
                _, (attn1, attn2) = out
                if attn1 is not None:
                    # attn1: (edge_index, attention_weights)
                    ei_out, aw = attn1
                    if attn_edge_index is None:
                        attn_edge_index = ei_out.cpu().numpy()
                    all_attn1.append(aw.cpu().numpy())
                if attn2 is not None:
                    ei_out, aw = attn2
                    all_attn2.append(aw.cpu().numpy())

    if not all_attn1:
        print(
            f"  ⚠️  No attention weights extracted for {model_name}/{rule_name} "
            f"(likely non-attention checkpoint)"
        )
        return None

    # Average attention weights across samples
    avg_attn1 = np.mean(all_attn1, axis=0).mean(axis=-1)  # [num_edges]
    avg_attn2 = np.mean(all_attn2, axis=0).mean(axis=-1) if all_attn2 else None

    # use the exact edge_index returned by attention layer when available
    ei_np = attn_edge_index if attn_edge_index is not None else edge_index.numpy()

    return {
        'edge_index': ei_np,
        'avg_attn1': avg_attn1,
        'avg_attn2': avg_attn2,
        'node_names': NODE_NAMES,
    }


# ==============================================================================
# COMPUTE: Node importance = sum of incoming attention weights
# ==============================================================================
def compute_node_importance(result):
    """
    Compute source-node importance via outgoing attention mass.
    Incoming attention is softmax-normalized per destination in GAT, which can
    become nearly constant across nodes.
    """
    n_nodes = len(result['node_names'])
    node_imp = np.zeros(n_nodes)

    ei = result['edge_index']
    attn = result['avg_attn1']

    for edge_idx in range(ei.shape[1]):
        src = ei[0, edge_idx]
        dst = ei[1, edge_idx]
        if src < n_nodes and dst < n_nodes and src != dst:
            node_imp[src] += attn[edge_idx]

    # Normalize
    if node_imp.sum() > 0:
        node_imp = node_imp / node_imp.sum()

    return node_imp


# ==============================================================================
# VISUALIZE: Node importance heatmap per rule
# ==============================================================================
def plot_node_importance_heatmap(importance_dict, model_name):
    """Rule별 노드 중요도 히트맵"""
    rules = list(importance_dict.keys())
    nodes = NODE_NAMES

    matrix = np.array([importance_dict[r] for r in rules])  # [n_rules, n_nodes]

    fig, ax = plt.subplots(figsize=(14, len(rules) * 0.8 + 2))

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=matrix.max())

    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes, rotation=35, ha='right', fontsize=11)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules, fontsize=11)

    # Value annotations
    for i in range(len(rules)):
        for j in range(len(nodes)):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Normalized Attention Weight', shrink=0.8)
    ax.set_title(f'Node Importance by Rule — {model_name}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Node', fontsize=12)
    ax.set_ylabel('Rule', fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'node_importance_heatmap_{model_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  💾 Saved: {path}")


# ==============================================================================
# VISUALIZE: DAG graph with attention weights
# ==============================================================================
def plot_dag_with_attention(result, rule_name, model_name):
    """DAG 그래프에 attention weight를 엣지 두께로 표시"""
    G = nx.DiGraph()
    G.add_nodes_from(NODE_NAMES)
    for src, dst in EDGES:
        G.add_edge(src, dst)

    # Node importance
    node_imp = compute_node_importance(result)
    node_sizes = [300 + node_imp[i] * 3000 for i in range(len(NODE_NAMES))]
    node_colors = [node_imp[i] for i in range(len(NODE_NAMES))]

    # Edge attention weights (layer 1)
    ei = result['edge_index']
    attn = result['avg_attn1']
    edge_weights = {}
    for edge_idx in range(ei.shape[1]):
        src_idx = ei[0, edge_idx]
        dst_idx = ei[1, edge_idx]
        if src_idx != dst_idx and src_idx < len(NODE_NAMES) and dst_idx < len(NODE_NAMES):
            src_name = NODE_NAMES[src_idx]
            dst_name = NODE_NAMES[dst_idx]
            key = (src_name, dst_name)
            edge_weights[key] = edge_weights.get(key, 0) + attn[edge_idx]

    # Layout
    pos = {
        'sex':         (0, 2),
        'ethnicity':   (0, 1),
        'marriage':    (0, 0),
        'education':   (1, 2),
        'class':       (1, 1),
        'admissions':  (2, 0),
        'admissions2': (3, 2),
        'admissions3': (3, 1),
        'admissions4': (5, 1),
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw edges with attention-based width
    max_w = max(edge_weights.values()) if edge_weights else 1
    for src, dst in EDGES:
        w = edge_weights.get((src, dst), 0.01)
        width = 1 + (w / max_w) * 8
        alpha = 0.3 + (w / max_w) * 0.7
        nx.draw_networkx_edges(
            G, pos, edgelist=[(src, dst)],
            width=width, alpha=alpha,
            edge_color='#E74C3C',
            arrows=True, arrowsize=20,
            ax=ax,
            connectionstyle='arc3,rad=0.1'
        )

    # Draw nodes
    cmap = plt.cm.YlOrRd
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap, vmin=0, vmax=max(node_imp) if max(node_imp) > 0 else 1,
        ax=ax, alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=max(node_imp)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Node Importance', shrink=0.7)

    ax.set_title(f'DAG — Node Importance & Edge Attention\n{model_name} | Rule: {rule_name}',
                 fontsize=13, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'dag_attention_{model_name}_{rule_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  💾 Saved: {path}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("🔍 Node Importance Analysis Start\n")
    print(f"📂 Using checkpoints from: {CKPT_DIR}\n")

    for model_name, model_class, temporal_encoder, use_edge_weight in MODELS_TO_ANALYZE:
        print(f"\n{'='*60}")
        print(f"📌 Model: {model_name}")
        print(f"{'='*60}")

        importance_dict = {}

        for rule_name in RULES_TO_ANALYZE:
            print(f"\n  🎯 Rule: {rule_name}")
            result = extract_attention_weights(
                model_name, model_class, temporal_encoder,
                use_edge_weight, rule_name, N_SIZE
            )

            if result is None:
                continue

            # Node importance
            node_imp = compute_node_importance(result)
            importance_dict[rule_name] = node_imp

            # Print top nodes
            top_idx = np.argsort(node_imp)[::-1][:3]
            print(f"  🏆 Top 3 important nodes:")
            for rank, idx in enumerate(top_idx):
                print(f"     {rank+1}. {NODE_NAMES[idx]}: {node_imp[idx]:.4f}")

            # DAG visualization
            plot_dag_with_attention(result, rule_name, model_name)

        # Heatmap across rules
        if importance_dict:
            plot_node_importance_heatmap(importance_dict, model_name)

    # Save CSV summary
    print("\n📊 Saving summary CSV...")
    rows = []
    for model_name, model_class, temporal_encoder, use_edge_weight in MODELS_TO_ANALYZE:
        for rule_name in RULES_TO_ANALYZE:
            result = extract_attention_weights(
                model_name, model_class, temporal_encoder,
                use_edge_weight, rule_name, N_SIZE
            )
            if result is None:
                continue
            node_imp = compute_node_importance(result)
            for i, name in enumerate(NODE_NAMES):
                rows.append({
                    'Model': model_name,
                    'Rule': rule_name,
                    'Node': name,
                    'Importance': round(node_imp[i], 5)
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'node_importance_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Summary saved: {csv_path}")
    print("\n🎉 Analysis Complete!")
