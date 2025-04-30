import numpy as np

def prox_l1(x: np.ndarray, t: float) -> np.ndarray:
    """
    Computes the ℓ1 proximal operator using soft-thresholding.
    
    Parameters:
        x (np.ndarray): Input vector.
        t (float): Threshold parameter (typically λ * step_size).

    Returns:
        np.ndarray: The result after applying the soft-thresholding operator element-wise.
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)