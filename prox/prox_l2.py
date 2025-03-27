import numpy as np

def prox_l2(v, lam):
    """
    Proximity operator for the L2 squared norm: (λ / 2) * ‖x‖²

    Solves: prox_{λ * (1/2) * ‖·‖²}(v) = argmin_x (1/2) * ‖x - v‖² + (λ / 2) * ‖x‖²

    Parameters:
    ----------
    v : np.ndarray
        Input vector.
    lam : float
        Regularization parameter (λ ≥ 0).

    Returns:
    -------
    x : np.ndarray
        Result of the proximity operator.
    """
    return v / (1.0 + lam)
