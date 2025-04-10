"""
Elastic Net Proximal Operator Module

Implements the proximal operator for the Elastic Net regularization, 
combining ℓ1 and ℓ2 regularization.
"""
import numpy as np
from .prox_l1 import prox_l1

def prox_elasticnet(x: np.ndarray, lam: float, alpha: float) -> np.ndarray:
    """
    Computes the proximal operator for Elastic Net regularization defined as:
    
        lam * [alpha * ||x||_1 + (1 - alpha) * ||x||_2^2]

    Parameters:
        x (np.ndarray): Input vector.
        lam (float): Regularization parameter.
        alpha (float): Mixing parameter between ℓ1 and ℓ2 (0 <= alpha <= 1).

    Returns:
        np.ndarray: The result after applying the Elastic Net proximal operator.
    """
    l1 = alpha * lam
    l2 = (1 - alpha) * lam
    scaled = x / (1 + 2 * l2)
    return prox_l1(scaled, l1)