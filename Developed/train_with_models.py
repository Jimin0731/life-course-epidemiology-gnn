import numpy as np
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.utils import add_self_loops, to_undirected

try:
    from Data_simulation.models import EconomicGNN, HybridGNN, WeightedGATGNN
except ImportError:
    from models import EconomicGNN, HybridGNN, WeightedGATGNN

BASE_DIR = Path(__file__).resolve().parent

def resolve_data_dir():
    candidates = [
        BASE_DIR / "data",
        BASE_DIR.parent / "Data_simulation" / "data",
        Path.cwd() / "data",
    ]
    for candidate in candidates:
        if (candidate / "dag_edges.csv").exists() and (candidate / "dag_node_names.txt").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find DAG files (dag_edges.csv, dag_node_names.txt). "
        f"Tried: {[str(c) for c in candidates]}"
    )

DATA_DIR = resolve_data_dir()

X = np.load(DATA_DIR / "data_X.npy").astype(np.float32)          # (2000, 9, 40)
Y = np.load(DATA_DIR / "data_Order1_Y.npy").astype(np.float32)   # (2000, 40)

N, num_nodes, feat_dim = X.shape
assert num_nodes == 9 and feat_dim == 40

y = Y[:, 39]  # (N,)

with open(DATA_DIR / "dag_node_names.txt", "r") as f:
    node_names = [line.strip() for line in f if line.strip()]

name2idx = {name: i for i, name in enumerate(node_names)}

edges_df = pd.read_csv(DATA_DIR / "dag_edges.csv")

src = edges_df["src"].map(name2idx).values
dst = edges_df["dst"].map(name2idx).values

edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

edge_index = to_undirected(edge_index)
edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_names))

print(f" Enhanced Graph! Final Edges: {edge_index.shape[1]}")
print(f" DAG Loaded! Nodes: {len(node_names)}, Edges: {edge_index.shape[1]}")
print("Edge Index Shape:", edge_index.shape)  

dataset = []
for i in range(N):
    data = Data(
        x=torch.from_numpy(X[i]),                # (9,40)
        edge_index=edge_index,                   # (2,E)
        y=torch.tensor([y[i]], dtype=torch.float32)  # (1,)
    )
    dataset.append(data)

# split
train_size = int(0.8 * N)
val_size = N - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gnn = EconomicGNN(input_dim=40, hidden_dim=64, output_dim=64).to(device)

# graph-level binary classifier head
clf = nn.Linear(64, 1).to(device)

# loss (pos_weight)
pos_rate = float(y.mean())
pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)
pos_weight = torch.tensor([20.0]) 
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(list(gnn.parameters()) + list(clf.parameters()), lr=1e-3)

def forward_batch(batch):
    out = gnn(batch.x, batch.edge_index)

    node_emb = out[0] if isinstance(out, tuple) else out

    graph_emb = global_mean_pool(node_emb, batch.batch)
    logits = clf(graph_emb).squeeze(-1)
    return logits


@torch.no_grad()
def eval_metrics(loader):
    gnn.eval(); clf.eval()
    ys, ps = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = forward_batch(batch)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ytrue = batch.y.view(-1).detach().cpu().numpy()

        ys.append(ytrue)
        ps.append(prob)

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan"), 0.5

    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)

    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y, (p >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_t = float(thresholds[best_idx])

    return float(auroc), float(auprc), best_f1, best_t


@torch.no_grad()
def eval_loss(loader):
    gnn.eval(); clf.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = forward_batch(batch)
        loss = criterion(logits, batch.y.view(-1))
        total += float(loss) * batch.num_graphs
        n += batch.num_graphs
    return total / n

best_score = 0
save_path = "best_model.pth"

for epoch in range(1, 11):
    gnn.train(); clf.train()
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = forward_batch(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        opt.step()

    tr = eval_loss(train_loader)
    va = eval_loss(val_loader)
    auroc, auprc, best_f1, best_t = eval_metrics(val_loader)
    val_auroc = auroc
    print(
    f"epoch {epoch:02d} | train_loss {tr:.4f} | val_loss {va:.4f} "
    f"| AUROC {auroc:.3f} | AUPRC {auprc:.3f} | bestF1 {best_f1:.3f} @t={best_t:.2f} "
    f"| pos_rate {pos_rate:.4f}"
)
    print("="*30)
    if val_auroc > best_score:
        best_score = val_auroc
        torch.save({"gnn": gnn.state_dict(), "clf": clf.state_dict()}, save_path)
        print(f"  --> 🎉 New Best Model Saved! (AUROC: {best_score:.4f})")

print(f"Loading Best Model with AUROC: {best_score:.4f}")
checkpoint = torch.load(save_path, map_location=device)
gnn.load_state_dict(checkpoint["gnn"])
clf.load_state_dict(checkpoint["clf"])

@torch.no_grad()
def flip_check(loader):
    gnn.eval(); clf.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = forward_batch(batch)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        ytrue = batch.y.view(-1).detach().cpu().numpy()
        ys.append(ytrue); ps.append(p)

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    if len(np.unique(y)) < 2:
        print("flip_check: only one class in y")
        return

    print("AUROC(y,p)  :", roc_auc_score(y, p))
    print("AUROC(y,1-p):", roc_auc_score(y, 1 - p))
    print("AUPRC(y,p)  :", average_precision_score(y, p))
    print("AUPRC(y,1-p):", average_precision_score(y, 1 - p))

print("done")
