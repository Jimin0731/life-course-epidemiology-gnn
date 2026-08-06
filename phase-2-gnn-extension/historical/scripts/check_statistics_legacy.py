import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

X = np.load(DATA_DIR / "data_X.npy")
Y = np.load(DATA_DIR / "data_Order1_Y.npy")

print("X:", X.shape, X.dtype)
print("Y:", Y.shape, Y.dtype)

print("X finite ratio:", np.isfinite(X).mean())
print("Y finite ratio:", np.isfinite(Y).mean())

pos_rate = Y.mean(axis=0)
print("Y positive rate (min/mean/max):", pos_rate.min(), pos_rate.mean(), pos_rate.max())

y0 = Y[:, 0]
print("y0 positive rate:", y0.mean())
print("unique y0:", np.unique(y0))
