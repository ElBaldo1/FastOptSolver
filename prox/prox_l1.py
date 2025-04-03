# prox/prox_l1.py

import numpy as np

def prox_l1(x: np.ndarray, t: float) -> np.ndarray:
    """
    Proximal operator for ℓ1 norm: λ * ||x||_1
    Implements soft-thresholding: shrinkage toward zero

    Assumptions:
    - t = λ * step_size is passed in already
    - Applies element-wise operation: soft(x_i, t) = sign(x_i) * max(|x_i| - t, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)
