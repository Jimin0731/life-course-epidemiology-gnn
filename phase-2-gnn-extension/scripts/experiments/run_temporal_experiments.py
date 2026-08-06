import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import glob
import sys
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, precision_recall_curve

PHASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PHASE_DIR / "src"
CONFIG_DIR = PHASE_DIR / "config"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from n_adaptive_config import EPOCHS_CONFIG, get_hyperparameters
from lce_gnn import EconomicGNN, SpatioTemporalGNN, WeightedGATGNN
from lce_gnn.feature_engineering import SuperRichFeatureEngineer



class FocalLoss(nn.Module):
    """
    Focal Loss: 쉬운 샘플(음성)은 무시하고 어려운 샘플(양성)에 집중
    - alpha: 양성 클래스 가중치 (불균형이 심할수록 높게)
    - gamma: focusing parameter (높을수록 어려운 샘플에 더 집중)
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # pos_weight 반영한 BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1 - self.alpha, device=logits.device))
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

# ==============================================================================
# 🚀 IMPROVED SETTINGS - Adaptive Configuration
# ==============================================================================
N_SIZES = [500, 1000, 2000, 5000, 10000, 20000]

BASE_DIR = str(PHASE_DIR)

def resolve_data_dir():
    candidates = [
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "scripts", "data", "data"),
        os.path.join(os.getcwd(), "data"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "dag_edges.csv")) and os.path.exists(
            os.path.join(candidate, "dag_node_names.txt")
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find DAG files (dag_edges.csv, dag_node_names.txt). "
        f"Tried: {candidates}"
    )

DAG_DIR = resolve_data_dir()

def resolve_dataset_dir(n_size):
    folder_name = f"data_{n_size}"
    candidates = [
        os.path.join(BASE_DIR, folder_name),
        os.path.join(BASE_DIR, "scripts", "data", folder_name),
        os.path.join(os.getcwd(), folder_name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None

RESULT_FILE = os.path.join(BASE_DIR, "results", "generated", "temporal_model_metrics.csv")

MODELS_TO_TEST = [
    ("EconomicGNN_CNN", EconomicGNN, 'cnn'),
    ("WeightedGAT_LSTM", WeightedGATGNN, 'lstm'),
    ("SpatioTemporalGNN", SpatioTemporalGNN, 'spatiotemporal'),
]

torch.set_num_threads(6)
print(f"🔧 Using {torch.get_num_threads()} CPU threads")

# ==============================================================================
# HELPER: Find optimal threshold for F1
# ==============================================================================
def find_optimal_threshold(y_true, y_probs):
    """Find threshold that maximizes F1 score"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1

# ==============================================================================
# ✅ FIX 3: Improved Experiment Function
# ==============================================================================
def run_experiment(model_class, temporal_encoder, n_size, rule_name, x_path, y_path, edge_index):
    """Run experiment with N-adaptive configuration"""
    
    # Get adaptive hyperparameters
    hparams = get_hyperparameters(n_size, temporal_encoder)
    epochs = EPOCHS_CONFIG[n_size]
    
    print(f"      📐 Config: hidden={hparams['hidden_dim']}, batch={hparams['batch_size']}, "
          f"lr={hparams['lr']}, epochs={epochs}", end=" | ")
    
    # 1. Data loading
    X_np = np.load(x_path)  
    Y_np = np.load(y_path)

    if X_np.shape[0] != Y_np.shape[0]:
        min_size = min(X_np.shape[0], Y_np.shape[0])
        X_np = X_np[:min_size]
        Y_np = Y_np[:min_size]

    # ✅ FIX 4: 샘플링 threshold 상향 (더 많은 데이터 활용)
    if X_np.shape[0] > 60000:  # 50000 → 60000
        print(f"sampling {X_np.shape[0]}→60000", end=" | ")
        indices = np.random.choice(X_np.shape[0], 60000, replace=False)
        X_np = X_np[indices]
        Y_np = Y_np[indices]
    
    X_tensor = torch.tensor(X_np, dtype=torch.float)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float)
    X_enriched = SuperRichFeatureEngineer.extract_features(X_tensor)

    
    n_samples, n_nodes, seq_len = X_tensor.shape
    input_dim = X_enriched.shape[-1]
    

    
    # Create data list
    data_list = []
    for i in range(n_samples):
        x_i = X_enriched[i]  
        y_i = Y_tensor[i, -1]
        data_list.append(Data(
            x=x_i,
            edge_index=edge_index, 
            y=torch.tensor([y_i], dtype=torch.float)
        ))
    
    labels = np.array([d.y.item() for d in data_list])
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    n_pos_train = max(1, int(len(pos_indices) * 0.8))
    n_neg_train = int(len(neg_indices) * 0.8)

    train_idx = np.concatenate([pos_indices[:n_pos_train], neg_indices[:n_neg_train]])
    val_idx   = np.concatenate([pos_indices[n_pos_train:], neg_indices[n_neg_train:]])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_data = [data_list[i] for i in train_idx]
    val_data   = [data_list[i] for i in val_idx]

    # ✅ NEW: Oversampling - pos_rate < 5%이면 양성 샘플 복제
    if pos_rate < 0.05 and pos_rate > 0:
        pos_data = [d for d in train_data if d.y.item() == 1]
        if len(pos_data) > 0:
            # 양성:음성 비율을 최소 1:10으로 맞춤 (최대 20배 복제)
            neg_count = len(train_data) - len(pos_data)
            target_pos = max(len(pos_data), neg_count // 10)
            repeat = min(int(target_pos / len(pos_data)), 20)
            train_data = train_data + pos_data * (repeat - 1)
            import random; random.shuffle(train_data)
            new_pos_rate = sum(d.y.item() for d in train_data) / len(train_data)
            print(f"oversample {pos_rate:.3f}→{new_pos_rate:.3f}", end=" | ")

    val_pos = sum(d.y.item() for d in val_data)
    train_pos = sum(d.y.item() for d in train_data)
    if val_pos == 0:
        return 0.5, 0.0, 0.5, 0.0, sum(labels) / len(labels)
    
    # ✅ Adaptive batch size
    train_loader = DataLoader(train_data, batch_size=hparams['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=hparams['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=False)
    
    # 2. Model initialization with adaptive hidden_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_class(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=hparams['hidden_dim'],  # ⚡ N-adaptive!
        output_dim=1,
        temporal_encoder=temporal_encoder
    ).to(device)
    
    # ✅ Adaptive learning rate
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=hparams['lr'],  # ⚡ N-adaptive!
                                weight_decay=1e-4)
    
    # ✅ FIX 5: More aggressive LR scheduling for large N
    if n_size >= 5000:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4  # 3→4 (더 여유)
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
    
    # Loss function with stronger pos_weight for large N
    y_train = [d.y.item() for d in train_data]
    pos_rate = sum(y_train) / (len(y_train) + 1e-9)
    
    if pos_rate == 0: 
        pos_weight = torch.tensor([1.0]).to(device)
    else:
        # pos_weight: 불균형 비율 반영
        raw_weight = (1 - pos_rate) / pos_rate
        if n_size >= 5000:
            weight_multiplier = 2.0
        else:
            weight_multiplier = 1.5
        pos_weight = torch.tensor([min(raw_weight * weight_multiplier, 100.0)]).to(device)

    # ✅ NEW: Focal Loss 사용 (불균형 데이터에 훨씬 효과적)
    # pos_rate < 5%: gamma=3.0 (더 강하게 어려운 샘플에 집중)
    # pos_rate >= 5%: gamma=2.0 (기본값)
    if pos_rate < 0.05:
        criterion = FocalLoss(alpha=0.75, gamma=3.0, pos_weight=pos_weight)
    elif pos_rate < 0.15:
        criterion = FocalLoss(alpha=0.50, gamma=2.0, pos_weight=pos_weight)
    else:
        criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    
    # 3. Training loop with adaptive patience
    best_auroc = 0.5
    best_f1 = 0.0
    best_f1_threshold = 0.5
    best_brier = 0.0
    patience_counter = 0
    max_patience = hparams['patience']  # ⚡ N-adaptive!
    
    for epoch in range(epochs):  # ⚡ N-adaptive epochs!
        # Training
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index)
            
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out

            pred_graph = global_max_pool(pred, batch.batch) if pred.dim() == 2 else pred
            loss = criterion(pred_graph.view(-1), batch.y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        # Validation
        model.eval()
        y_true, y_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)

                if isinstance(out, tuple):
                    pred = out[0]
                else:
                    pred = out

                pred_graph = global_max_pool(pred, batch.batch) if pred.dim() == 2 else pred

                y_true.append(batch.y.cpu())
                y_logits.append(pred_graph.cpu())
        
        try:
            if len(y_true) > 0:
                y_true_np = torch.cat(y_true).numpy()
                y_logits_np = torch.cat(y_logits).numpy().flatten()
                y_probs_np = torch.sigmoid(torch.tensor(y_logits_np)).numpy()
                
                if len(np.unique(y_true_np)) > 1:
                    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true_np, y_probs_np)
                    y_preds_optimal = (y_probs_np > optimal_threshold).astype(int)

                    auroc = roc_auc_score(y_true_np, y_probs_np)
                    f1_optimal = f1_score(y_true_np, y_preds_optimal, zero_division=0)
                    brier = brier_score_loss(y_true_np, y_probs_np)

                    scheduler.step(auroc)

                    # ✅ NEW: F1 기준으로 early stopping (AUROC → F1)
                    improved = False
                    if f1_optimal > best_f1:
                        improved = True
                    elif f1_optimal == best_f1 and auroc > best_auroc:
                        improved = True

                    if improved:
                        best_auroc = auroc
                        best_f1 = f1_optimal
                        best_f1_threshold = optimal_threshold
                        best_brier = brier
                        patience_counter = 0

                        if not os.path.exists("models_ckpt"):
                            os.makedirs("models_ckpt")
                        torch.save(model.state_dict(),
                                 f"models_ckpt/best_model_N{n_size}_{rule_name}_{temporal_encoder}.pth")
                    else:
                        patience_counter += 1

                    if patience_counter >= max_patience:
                        print(f"[early@{epoch+1}]", end="")
                        break
                        
        except Exception as e:
            pass
            
    return best_auroc, best_f1, best_f1_threshold, best_brier, pos_rate

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    results = []

    # Load DAG
    print(f"Loading DAG...")
    dag_edges = pd.read_csv(os.path.join(DAG_DIR, "dag_edges.csv"))
    dag_nodes = [line.strip() for line in open(os.path.join(DAG_DIR, "dag_node_names.txt"))]
    name2idx = {n:i for i,n in enumerate(dag_nodes)}
    
    src = dag_edges["src"].map(name2idx).values
    dst = dag_edges["dst"].map(name2idx).values
    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(dag_nodes))

    # Experiment loop
    processed_n = 0
    for n_size in N_SIZES:
        data_folder = resolve_dataset_dir(n_size)
        if data_folder is None:
            print(f"⚠️  Skip N={n_size}: cannot find data_{n_size} folder")
            continue
        processed_n += 1
            
        x_file = os.path.join(data_folder, "data_X.npy")
        y_files = sorted(glob.glob(os.path.join(data_folder, "data_*_Y.npy")))
        if not y_files:
            print(f"⚠️  Skip N={n_size}: no Y files in {data_folder}")
            continue
        
        print(f"\n📂 Processing N={n_size} (epochs={EPOCHS_CONFIG[n_size]}) ... [{data_folder}]")
        
        for y_file in y_files:
            rule_name = os.path.basename(y_file).replace("data_", "").replace("_Y.npy", "")
            print(f"  🎯 {rule_name}")
            
            for model_name, model_class, temporal_encoder in MODELS_TO_TEST:
                print(f"    ▶ {model_name}: ", end="", flush=True)
                try:
                    exp_start = time.time()
                    auroc, f1, threshold, brier, pos_rate = run_experiment(
                        model_class, temporal_encoder, n_size, rule_name, 
                        x_file, y_file, edge_index
                    )
                    exp_time = time.time() - exp_start
                    print(f" ✅ AUROC:{auroc:.3f} F1:{f1:.3f}@{threshold:.2f} Brier:{brier:.3f} ({exp_time:.1f}s)")
                    
                    results.append({
                        "N": n_size,
                        "Rule": rule_name,
                        "Model": model_name,
                        "Temporal_Encoder": temporal_encoder,
                        "AUROC": auroc,
                        "F1_Score": f1,
                        "Optimal_Threshold": threshold,
                        "Brier_Score": brier,
                        "Pos_Rate": pos_rate,
                        "Training_Time_Sec": exp_time
                    })
                except Exception as e:
                    print(f"❌ Error: {e}")

    if processed_n == 0:
        print("❌ No dataset folders were found. Expected data_500/.../data_10000")

    # Save results
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
        df.to_csv(RESULT_FILE, index=False)
        
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print(f"✅ IMPROVED Experiment complete! Results: '{RESULT_FILE}'")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print("="*70)
        
        # Performance comparison
        print("\n📊 Performance Summary by N:")
        summary = df.groupby(['N', 'Model']).agg({
            'AUROC': 'mean',
            'F1_Score': 'mean'
        }).round(3)
        print(summary)
        
        print("\n📈 Overall Performance by Model:")
        overall = df.groupby('Model').agg({
            'AUROC': ['mean', 'std'],
            'F1_Score': ['mean', 'std']
        }).round(3)
        print(overall)
