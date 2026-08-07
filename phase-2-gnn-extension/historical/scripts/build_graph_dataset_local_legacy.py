import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

x_path = DATA_DIR / "x_out.npy"
y_path = DATA_DIR / "y_out.npy"

x_out = np.load(x_path, allow_pickle=True)
y_out = np.load(y_path, allow_pickle=True)

print("x_out type:", type(x_out), "shape:", getattr(x_out, "shape", None))
print("y_out type:", type(y_out), "shape:", getattr(y_out, "shape", None))
print("y_out sample:", y_out[:10])
