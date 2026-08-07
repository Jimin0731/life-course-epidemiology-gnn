import torch
import numpy as np
import pandas as pd
import os
import glob
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score

# ==============================================================================
# [설정] 실험 파라미터
# ==============================================================================
N_SIZES = [1000, 2000]       # R에서 생성한 폴더 이름에 맞춰 설정
EPOCHS = 15                  # 실험 속도를 위해 15 정도로 설정 (필요시 증가)
DAG_DIR = "data"             # dag_edges.csv가 있는 폴더 (아까 만든 곳)
RESULT_FILE = "final_experiment_results.csv" # 결과 저장 파일명

# ==============================================================================
# [모델 정의] 간단한 GNN 모델 (또는 기존 EconomicGNN 사용 가능)
# ==============================================================================
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(SimpleGNN, self).__init__()
        # Time steps(40)을 feature로 사용
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2) 
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # x: (Total_Nodes, Time_Steps)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # 그래프 단위 예측을 위해 Pooling (Readout)
        x = global_mean_pool(x, batch)
        return self.fc(x)

# ==============================================================================
# [함수] 한 번의 실험(Experiment)을 수행
# ==============================================================================
def run_experiment(n_size, rule_name, x_path, y_path, edge_index):
    # 1. 데이터 로드
    X_np = np.load(x_path) # (N, 9, 41) - 마지막 차원은 Time+1일 수 있음
    Y_np = np.load(y_path) # (N, 41)
    
    # 텐서 변환 (N, Nodes, Time)
    # R 코드에서 (N, 9, 41)로 저장했으므로 Time=41 (0~40)
    X_tensor = torch.tensor(X_np, dtype=torch.float)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float)
    
    # 데이터셋 구성 (PyG Data list)
    data_list = []
    for i in range(X_tensor.shape[0]):
        # Feature: (9, 41)
        # Target: 마지막 시점(40)의 값 사용 -> Y_tensor[i, -1]
        x_i = X_tensor[i] 
        y_i = Y_tensor[i, -1] 
        
        data_list.append(Data(x=x_i, edge_index=edge_index, y=torch.tensor([y_i], dtype=torch.float)))
    
    # Train/Val 분리 (80:20)
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    val_data = data_list[train_size:]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # 2. 모델 및 학습 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGNN(input_dim=X_tensor.shape[2]).to(device) # Input dim = Time steps
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # 불균형 데이터 가중치 계산
    y_train = [d.y.item() for d in train_data]
    pos_rate = sum(y_train) / len(y_train)
    pos_weight = torch.tensor([(1 - pos_rate) / (pos_rate + 1e-6)]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. 학습 루프
    best_auroc = 0.5
    best_auprc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            
        # 검증 (Validation)
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                y_true.append(batch.y.cpu())
                y_pred.append(out.cpu())
        
        try:
            y_true = torch.cat(y_true).numpy()
            y_pred = torch.sigmoid(torch.cat(y_pred)).numpy()
            
            # AUROC 계산 (클래스가 하나뿐이면 에러 발생 가능)
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)
                if auroc > best_auroc: 
                    best_auroc = auroc
                    best_auprc = auprc
        except:
            pass # 계산 에러 무시
            
    return best_auroc, best_auprc, pos_rate

# ==============================================================================
# [메인] 전체 실행 로직
# ==============================================================================
if __name__ == "__main__":
    results = []

    # 1. DAG 구조 로드 (공통)
    print(f"Loading DAG from {DAG_DIR}...")
    try:
        dag_edges = pd.read_csv(os.path.join(DAG_DIR, "dag_edges.csv"))
        dag_nodes = [line.strip() for line in open(os.path.join(DAG_DIR, "dag_node_names.txt"))]
        name2idx = {n:i for i,n in enumerate(dag_nodes)}
        
        src = dag_edges["src"].map(name2idx).values
        dst = dag_edges["dst"].map(name2idx).values
        edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
        
        # 그래프 강화 (무방향 + Self-loop)
        edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(dag_nodes))
        print("✅ DAG Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading DAG: {e}")
        exit()

    # 2. 모든 N 사이즈와 규칙에 대해 실험 반복
    for n_size in N_SIZES:
        data_folder = f"data_{n_size}"
        
        if not os.path.exists(data_folder):
            print(f"⚠️ Warning: Folder {data_folder} not found. Skipping...")
            continue
            
        x_file = os.path.join(data_folder, "data_X.npy")
        # 해당 폴더 내의 모든 Y 파일 찾기
        y_files = glob.glob(os.path.join(data_folder, "data_*_Y.npy"))
        
        print(f"\n📂 Processing {data_folder} ({len(y_files)} rules found)")
        
        for y_file in sorted(y_files):
            # 파일명에서 규칙 이름 추출 (data_repeat2_Y.npy -> repeat2)
            rule_name = os.path.basename(y_file).replace("data_", "").replace("_Y.npy", "")
            
            print(f"  ▶ Running {rule_name}...", end=" ", flush=True)
            
            try:
                auroc, auprc, pos_rate = run_experiment(n_size, rule_name, x_file, y_file, edge_index)
                print(f"Done! AUROC: {auroc:.4f}")
                
                results.append({
                    "N": n_size,
                    "Rule": rule_name,
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "Pos_Rate": pos_rate
                })
            except Exception as e:
                print(f"Error: {e}")

    # 3. 결과 저장
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULT_FILE, index=False)
        print("\n" + "="*50)
        print(f"🎉 All experiments finished! Results saved to '{RESULT_FILE}'")
        print("="*50)
        print(df)
    else:
        print("\n❌ No results generated.")