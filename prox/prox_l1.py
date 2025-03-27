import numpy as np

def prox_l1(v, lam):
    """
    Proximity operator for the L1 norm (soft-thresholding).

    Solves: prox_{λ‖·‖₁}(v) = argmin_x (1/2) * ‖x - v‖² + λ‖x‖₁

    Parameters:
    ----------
    v : np.ndarray
        Input vector.
    lam : float
        Regularization parameter (λ ≥ 0).

    Returns:
    -------
    x : np.ndarray
        Result of applying the soft-thresholding operator to v.
    """
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)
