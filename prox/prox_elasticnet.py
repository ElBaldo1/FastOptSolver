import numpy as np
from .prox_l1 import prox_l1

def prox_elasticnet(x: np.ndarray, lam: float, alpha: float) -> np.ndarray:
    """
    Prox operator for Elastic Net: lam * [alpha * ||x||_1 + (1 - alpha) * ||x||_2^2]
    Equivalent to soft-thresholding applied to a scaled input.
    """
    l1 = alpha * lam
    l2 = (1 - alpha) * lam
    scaled = x / (1 + 2 * l2)
    return prox_l1(scaled, l1)