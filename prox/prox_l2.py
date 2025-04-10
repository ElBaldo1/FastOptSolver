import numpy as np

def prox_l2(x: np.ndarray, t: float) -> np.ndarray:
    """
    Computes the proximal operator for the ℓ2 norm regularization (using shrinkage).
    
    Parameters:
        x (np.ndarray): Input vector.
        t (float): Threshold parameter (λ * step_size).

    Returns:
        np.ndarray: The result after applying the shrinkage operator.
    """
    return x / (1 + 2 * t)