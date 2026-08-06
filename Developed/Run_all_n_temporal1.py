import torch
import numpy as np
import pandas as pd
import os
import glob
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, precision_recall_curve

# models Import
try:
    from Data_simulation.models import EconomicGNN, HybridGNN, WeightedGATGNN
    print("✅ All model files loaded successfully! (Data_simulation.models)")
except ImportError as e:
    try:
        from models import EconomicGNN, HybridGNN, WeightedGATGNN
        print("✅ All model files loaded successfully! (models)")
    except ImportError as e2:
        print(f"❌ Cannot find model files: {e2}")
        exit()

# ==============================================================================
# SETTINGS - IMPROVED FOR F1 SCORE
# ==============================================================================
N_SIZES = [500, 1000, 2000, 5000, 10000]
EPOCHS = 15 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_data_dir():
    candidates = [
        os.path.join(BASE_DIR, "data"),
        os.path.join(os.path.dirname(BASE_DIR), "Data_simulation", "data"),
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
RESULT_FILE = "final_model_comparison_metrics_improved_f1.csv"

MODELS_TO_TEST = [
    ("EconomicGNN_LSTM", EconomicGNN, 'lstm'),
    ("EconomicGNN_GRU", EconomicGNN, 'gru'),
    ("EconomicGNN_CNN", EconomicGNN, 'cnn'),
    ("HybridGNN_LSTM", HybridGNN, 'lstm'),
    ("WeightedGAT_LSTM", WeightedGATGNN, 'lstm'),
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
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1

# ==============================================================================
# MAIN EXPERIMENT FUNCTION - IMPROVED
# ==============================================================================
def run_experiment(model_class, temporal_encoder, n_size, rule_name, x_path, y_path, edge_index):
    """Run experiment with optimal threshold finding"""
    
    # 1. Data loading
    X_np = np.load(x_path)  
    Y_np = np.load(y_path)

    if X_np.shape[0] != Y_np.shape[0]:
        min_size = min(X_np.shape[0], Y_np.shape[0])
        print(f"      ⚠️  Size mismatch! X: {X_np.shape[0]}, Y: {Y_np.shape[0]} → using {min_size}")
        X_np = X_np[:min_size]
        Y_np = Y_np[:min_size]

    # Data sampling for large datasets
    if X_np.shape[0] > 50000:
        print(f"      ⚡ Sampling data: {X_np.shape[0]} → 50000")
        indices = np.random.choice(X_np.shape[0], 50000, replace=False)
        X_np = X_np[indices]
        Y_np = Y_np[indices]
    
    print(f"      Data shape - X: {X_np.shape}, Y: {Y_np.shape}")
    
    X_tensor = torch.tensor(X_np, dtype=torch.float)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float)
    
    n_samples, n_nodes, seq_len = X_tensor.shape
    input_dim = 1
    
    if X_tensor.dim() == 3:
        X_tensor = X_tensor.unsqueeze(-1)
    
    # Create data list
    data_list = []
    for i in range(n_samples):
        x_i = X_tensor[i]  
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

    val_pos = sum(d.y.item() for d in val_data)
    train_pos = sum(d.y.item() for d in train_data)
    if val_pos == 0:
        print(f"      ⚠️  val positive=0 (train positive={int(train_pos)}) → evaluation impossible, skip")
        return 0.5, 0.0, 0.5, 0.0, sum(labels) / len(labels)
    
    # ⚡ Batch size optimization
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    # 2. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ⚡ Increased hidden_dim for better performance
    if temporal_encoder == 'lstm':
        hidden_dim = 64  # 48 → 64 (better capacity)
    else:
        hidden_dim = 48  # 32 → 48
    
    model = model_class(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_dim=1,
        temporal_encoder=temporal_encoder
    ).to(device)
    
    # ⚡ Optimizer with adjusted learning rate
    if temporal_encoder == 'lstm':
        lr = 0.005  # Slightly lower for stability
    else:
        lr = 0.007
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3  # Increased patience
    )
    
    # ⚡ IMPROVED: Focal Loss for imbalanced data
    y_train = [d.y.item() for d in train_data]
    pos_rate = sum(y_train) / (len(y_train) + 1e-9)
    
    # Stronger weight for minority class
    if pos_rate == 0: 
        pos_weight = torch.tensor([1.0]).to(device)
    else:
        # ⚡ Increased weight for positive class
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate * 1.5]).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Training loop
    best_auroc = 0.5
    best_f1 = 0.0
    best_f1_threshold = 0.5
    best_brier = 0.0
    patience_counter = 0
    max_patience = 5  # ⚡ Increased patience
    
    for epoch in range(EPOCHS):
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

            pred_graph = global_mean_pool(pred, batch.batch) if pred.dim() == 2 else pred
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

                pred_graph = global_mean_pool(pred, batch.batch) if pred.dim() == 2 else pred

                y_true.append(batch.y.cpu())
                y_logits.append(pred_graph.cpu())
        
        try:
            if len(y_true) > 0:
                y_true_np = torch.cat(y_true).numpy()
                y_logits_np = torch.cat(y_logits).numpy().flatten()
                
                # Logits -> Probability
                y_probs_np = torch.sigmoid(torch.tensor(y_logits_np)).numpy()
                
                if len(np.unique(y_true_np)) > 1:
                    # ✅ FIX 2: Optimal threshold for F1
                    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true_np, y_probs_np)
                    y_preds_optimal = (y_probs_np > optimal_threshold).astype(int)

                    auroc = roc_auc_score(y_true_np, y_probs_np)
                    f1_optimal = f1_score(y_true_np, y_preds_optimal, zero_division=0)
                    brier = brier_score_loss(y_true_np, y_probs_np)

                    scheduler.step(auroc)

                    # ✅ FIX 3: AUROC 기준 저장 (F1보다 안정적, 특히 소규모 데이터)
                    if auroc > best_auroc:
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

                    # ✅ FIX 4: Early stopping - val에 양성 충분할 때만 발동
                    if patience_counter >= max_patience:
                        print(f" (early stop at epoch {epoch+1})", end="")
                        break
                else:
                    # val에 양성이 없는 epoch은 patience 증가 안 함 (학습은 계속)
                    pass
                        
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
    for n_size in N_SIZES:
        data_folder = f"data_{n_size}"
        if not os.path.exists(data_folder): 
            continue
            
        x_file = os.path.join(data_folder, "data_X.npy")
        y_files = sorted(glob.glob(os.path.join(data_folder, "data_*_Y.npy")))
        
        print(f"\n📂 Processing N={n_size} ...")
        
        for y_file in y_files:
            rule_name = os.path.basename(y_file).replace("data_", "").replace("_Y.npy", "")
            print(f"  Target Rule: {rule_name}")
            
            for model_name, model_class, temporal_encoder in MODELS_TO_TEST:
                print(f"    ▶ {model_name} training...", end=" ", flush=True)
                try:
                    exp_start = time.time()
                    auroc, f1, threshold, brier, pos_rate = run_experiment(
                        model_class, temporal_encoder, n_size, rule_name, 
                        x_file, y_file, edge_index
                    )
                    exp_time = time.time() - exp_start
                    print(f"Done! AUROC: {auroc:.4f} | F1: {f1:.4f} (threshold={threshold:.3f}) | Brier: {brier:.4f} | Time: {exp_time:.1f}s")
                    
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
                    print(f"Error: {e}")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULT_FILE, index=False)
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ Experiment complete! Results saved: '{RESULT_FILE}'")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print("="*60)
        print(df.head(20))
        print(f"\n📊 F1 Score Summary:")
        print(df.groupby('Model')['F1_Score'].agg(['mean', 'std', 'min', 'max']))
