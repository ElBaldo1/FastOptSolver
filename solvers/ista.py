import numpy as np
from prox.prox_l1 import prox_l1

def ista(A, b, lam, step_size, max_iter=1000, tol=1e-6, verbose=False):
    """
    ISTA algorithm for solving: min_x 0.5 * ||Ax - b||^2 + lambda * ||x||_1

    Parameters:
    ----------
    A : np.ndarray (m x n)
        Design matrix.
    b : np.ndarray (m,)
        Observation vector.
    lam : float
        Regularization parameter (lambda ≥ 0).
    step_size : float
        Step size (should be <= 1 / ||A.T @ A|| for convergence).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance on the norm of the update step.
    verbose : bool
        Whether to print iteration logs.

    Returns:
    -------
    x : np.ndarray (n,)
        Estimated solution.
    history : dict
        Contains 'objective' and 'residual' lists over iterations.
    """
    m, n = A.shape
    x = np.zeros(n)
    history = {'objective': [], 'residual': []}

    for k in range(max_iter):
        grad = A.T @ (A @ x - b)
        x_new = prox_l1(x - step_size * grad, lam * step_size)

        obj = 0.5 * np.linalg.norm(A @ x_new - b)**2 + lam * np.linalg.norm(x_new, 1)
        res = np.linalg.norm(x_new - x)

        history['objective'].append(obj)
        history['residual'].append(res)

        if verbose:
            print(f"Iter {k:4d}: Obj = {obj:.6f}, Residual = {res:.2e}")

        if res < tol:
            break

        x = x_new

    return x, history
