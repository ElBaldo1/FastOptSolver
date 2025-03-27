import numpy as np
from prox.prox_l1 import prox_l1

def fista(A, b, lam, step_size, max_iter=1000, tol=1e-6, verbose=False):
    """
    FISTA algorithm for: min_x 0.5 * ||Ax - b||^2 + λ * ||x||_1

    Parameters:
    ----------
    A : np.ndarray (m x n)
    b : np.ndarray (m,)
    lam : float
        Regularization parameter
    step_size : float
        Step size (should be ≤ 1 / ||A^T A||₂)
    max_iter : int
        Max number of iterations
    tol : float
        Stopping threshold
    verbose : bool
        If True, print iteration info

    Returns:
    -------
    x : np.ndarray (n,)
        Final solution
    history : dict
        'objective', 'residual' over iterations
    """
    m, n = A.shape
    x = np.zeros(n)
    y = np.zeros(n)
    t = 1.0
    history = {'objective': [], 'residual': []}

    for k in range(max_iter):
        grad = A.T @ (A @ y - b)
        x_new = prox_l1(y - step_size * grad, lam * step_size)

        obj = 0.5 * np.linalg.norm(A @ x_new - b)**2 + lam * np.linalg.norm(x_new, 1)
        res = np.linalg.norm(x_new - x)
        history['objective'].append(obj)
        history['residual'].append(res)

        if verbose:
            print(f"Iter {k:4d}: Obj = {obj:.6f}, Residual = {res:.2e}")

        if res < tol:
            break

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

    return x, history
