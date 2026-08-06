import torch
import numpy as np
import pandas as pd
import os
import glob
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

# models Import (package structure)
try:
    from Data_simulation.models import EconomicGNN, HybridGNN, WeightedGATGNN
    print(" All model files loaded successfully! (Data_simulation.models)")
except ImportError as e:
    try:
        from models import EconomicGNN, HybridGNN, WeightedGATGNN
        print(" All model files loaded successfully! (models)")
    except ImportError as e2:
        print(f" Cannot find model files: {e2}")
        exit()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean': return torch.mean(F_loss)
        else: return F_loss

# ==============================================================================
# [setting] experiment paraemeters
# ==============================================================================
N_SIZES = [500, 1000, 2000, 5000, 10000]       # N siezes to test
EPOCHS = 15                  # epoch
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

DAG_DIR = resolve_data_dir()             # DAG
RESULT_FILE = "final_model_comparison_metrics.csv" # file save

# name, class
MODELS_TO_TEST = [
    ("EconomicGNN", EconomicGNN),
    ("HybridGNN", HybridGNN),
    ("WeightedGAT", WeightedGATGNN)
]

# ==============================================================================
# [function] one experiment run
# ==============================================================================
def run_experiment(model_class, n_size, rule_name, x_path, y_path, edge_index):
    # 1. data load
    X_np = np.load(x_path) 
    Y_np = np.load(y_path)
    
    X_tensor = torch.tensor(X_np, dtype=torch.float)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float)
    
    data_list = []
    for i in range(X_tensor.shape[0]):
        x_i = X_tensor[i] 
        y_i = Y_tensor[i, -1]
        data_list.append(Data(x=x_i, edge_index=edge_index, y=torch.tensor([y_i], dtype=torch.float)))
    
    # Train/Val Split
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    val_data = data_list[train_size:]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # 2. initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = X_tensor.shape[2]
    model = model_class(input_dim=input_dim, output_dim=1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # computation weight (Imbalanced Data)
    #y_train = [d.y.item() for d in train_data]
    #pos_rate = sum(y_train) / (len(y_train) + 1e-9)
    #if pos_rate == 0: pos_weight = torch.tensor([1.0]).to(device)
    #else: pos_weight = torch.tensor([(1 - pos_rate) / pos_rate]).to(device)
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    
    
    # 3. leanrning loop
    best_auroc = 0.5
    best_f1 = 0.0
    best_brier = 0.0
    
    for epoch in range(EPOCHS):
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
                
                # Logits -> Probability (Sigmoid)
                y_probs_np = torch.sigmoid(torch.tensor(y_logits_np)).numpy()
                
                # Probability -> Binary Prediction (Threshold 0.5)
                y_preds_np = (y_probs_np > 0.5).astype(int)
                
                if len(np.unique(y_true_np)) > 1:
                    # Metric Calculation
                    auroc = roc_auc_score(y_true_np, y_probs_np)
                    f1 = f1_score(y_true_np, y_preds_np, zero_division=0)
                    brier = brier_score_loss(y_true_np, y_probs_np)
                    
                    # save the best indicators based on AUROC 
                    if auroc > best_auroc: 
                        best_auroc = auroc
                        best_f1 = f1
                        best_brier = brier

                        if not os.path.exists("models_ckpt"): os.makedirs("models_ckpt")
                        torch.save(model.state_dict(), f"models_ckpt/best_model_N{n_size}_{rule_name}.pth")
        except Exception as e:
            # print(f"Metric calculation error: {e}")
            pass
            
    return best_auroc, best_f1, best_brier, pos_rate

# ==============================================================================
# [main] implemenation logic
# ==============================================================================
if __name__ == "__main__":
    results = []

    # 1. DAG load
    print(f"Loading DAG...")
    dag_edges = pd.read_csv(os.path.join(DAG_DIR, "dag_edges.csv"))
    dag_nodes = [line.strip() for line in open(os.path.join(DAG_DIR, "dag_node_names.txt"))]
    name2idx = {n:i for i,n in enumerate(dag_nodes)}
    
    src = dag_edges["src"].map(name2idx).values
    dst = dag_edges["dst"].map(name2idx).values
    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(dag_nodes))

    # 2. experiment loop
    for n_size in N_SIZES:
        data_folder = f"data_{n_size}"
        if not os.path.exists(data_folder): continue
            
        x_file = os.path.join(data_folder, "data_X.npy")
        y_files = sorted(glob.glob(os.path.join(data_folder, "data_*_Y.npy")))
        
        print(f"\n📂 Processing N={n_size} ...")
        
        for y_file in y_files:
            rule_name = os.path.basename(y_file).replace("data_", "").replace("_Y.npy", "")
            print(f"  Target Rule: {rule_name}")
            
            for model_name, model_class in MODELS_TO_TEST:
                print(f"    ▶ {model_name} training...", end=" ", flush=True)
                try:
                    score, f1, brier, pos_rate = run_experiment(model_class, n_size, rule_name, x_file, y_file, edge_index)
                    print(f"Done! AUROC: {score:.4f} | F1: {f1:.4f} | Brier: {brier:.4f}")
                    
                    results.append({
                        "N": n_size,
                        "Rule": rule_name,
                        "Model": model_name,
                        "AUROC": score,
                        "F1_Score": f1,
                        "Brier_Score": brier,
                        "Pos_Rate": pos_rate
                    })
                except Exception as e:
                    print(f"Error: {e}")

    # 3. save the result
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULT_FILE, index=False)
        print("\n" + "="*50)
        print(f" Experiment complete! The result saved.: '{RESULT_FILE}'")
        print("="*50)
        print(df.head())
