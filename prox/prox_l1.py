import numpy as np

print(">>> prox_l1.py loaded")

def prox_l1(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

def lasso_objective(A, b, x, lam):
    return 0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.linalg.norm(x, 1)
