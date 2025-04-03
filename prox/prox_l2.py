# prox/prox_l2.py

import numpy as np

def prox_l2(x: np.ndarray, t: float) -> np.ndarray:
    """
    Proximal operator for λ * ||x||_2^2:
    Closed-form solution is shrinkage toward zero.

    Assumptions:
    - t = λ * step_size
    - Since ∇(λ * ||x||^2) = 2λx, the shrinkage factor becomes 1 / (1 + 2t)
    """
    return x / (1 + 2 * t)
