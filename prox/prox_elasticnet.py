import numpy as np
from prox_l1 import prox_l1

def prox_elasticnet(v, lam1, lam2):
    """
    Proximity operator for Elastic Net: λ1‖x‖₁ + (λ2/2)‖x‖²

    Solves: prox_{λ1‖·‖₁ + (λ2/2)‖·‖²}(v)

    Parameters:
    ----------
    v : np.ndarray
        Input vector.
    lam1 : float
        L1 regularization parameter.
    lam2 : float
        L2 regularization parameter.

    Returns:
    -------
    x : np.ndarray
        Output of the proximity operator.
    """
    scaled_v = v / (1.0 + lam2)
    return prox_l1(scaled_v, lam1 / (1.0 + lam2))
