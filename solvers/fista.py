import numpy as np
from solvers.gradient import grad_f
from prox.prox_l1 import prox_l1, lasso_objective

def fista(A, b, lam, step_size, max_iter=1000, tol=1e-6, verbose=False):
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for LASSO:
        min_x 0.5 * ||Ax - b||^2 + lam * ||x||_1

    Parameters
    ----------
    A : np.ndarray (m, n)
        Design matrix.
    b : np.ndarray (m,)
        Target vector.
    lam : float
        Regularization strength.
    step_size : float
        Step size (must satisfy step_size <= 1 / ||AᵗA||).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance on update norm.
    verbose : bool
        If True, print iteration log.

    Returns
    -------
    x : np.ndarray (n,)
        Final estimated solution.
    history : dict
        - 'objective': list of objective values.
        - 'residual': list of residuals ||x_k+1 - x_k||.
    """
    m, n = A.shape
    x = np.zeros(n)
    y = np.zeros(n)
    t = 1.0
    history = {'objective': [], 'residual': []}

    for k in range(max_iter):
        # Gradient step on y
        grad = grad_f(A, b, y)

        # Prox step (soft-thresholding)
        x_new = prox_l1(y - step_size * grad, lam * step_size)

        # Track objective and residual
        obj = lasso_objective(A, b, x_new, lam)
        res = np.linalg.norm(x_new - x)

        history['objective'].append(obj)
        history['residual'].append(res)

        if verbose:
            print(f"[FISTA] Iter {k:4d} | Obj: {obj:.6f} | Residual: {res:.2e}")

        if res < tol:
            break

        # Acceleration
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

    return x, history
