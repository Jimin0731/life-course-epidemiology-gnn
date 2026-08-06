import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/jiminbyun/Personal Project/data")

X = np.load(DATA_DIR / "data_X.npy", allow_pickle=True)

print("=== X (node features) ===")
print("type:", type(X))
print("shape:", X.shape)
print("dtype:", X.dtype)
print("first row:", X[0])
print()

Y_order1 = np.load(DATA_DIR / "data_Order1_Y.npy", allow_pickle=True)

print("=== Y (Order1) ===")
print("type:", type(Y_order1))
print("shape:", Y_order1.shape)
print("dtype:", Y_order1.dtype)
print("first 10:", Y_order1[:10])