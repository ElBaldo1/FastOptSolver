import numpy as np

def grad_f(A, b, x):
    """
    Gradient of the smooth part of the LASSO objective:
        f(x) = 0.5 * ||Ax - b||^2

    Parameters:
        A : np.ndarray, shape (m, n)
            Data matrix
        b : np.ndarray, shape (m,)
            Target vector
        x : np.ndarray, shape (n,)
            Current iterate

    Returns:
        grad : np.ndarray, shape (n,)
            Gradient vector: Aᵗ (A x - b)
    """
    return A.T @ (A @ x - b)
