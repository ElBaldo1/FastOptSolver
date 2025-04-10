import numpy as np
from solvers.gradient import grad_f
from prox.prox_l1 import prox_l1, lasso_objective

def ista(A, b, lam, step_size, max_iter=1000, tol=1e-6, verbose=False):
    """
    ISTA (Iterative Shrinkage-Thresholding Algorithm) for solving:
        min_x 0.5 * ||Ax - b||^2 + lam * ||x||_1

    This implementation is specialized for the LASSO problem.

    Parameters
    ----------
    A : np.ndarray (m, n)
        Data matrix
    b : np.ndarray (m,)
        Observation vector
    lam : float
        Regularization parameter (lambda ≥ 0)
    step_size : float
        Step size for gradient descent (typically 1 / ||AᵀA||)
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion
    verbose : bool
        If True, print progress per iteration

    Returns
    -------
    x : np.ndarray (n,)
        Final estimated solution
    history : dict
        - 'objective': list of F(x) per iteration
        - 'residual': list of ||x_k+1 - x_k|| per iteration
    """
    m, n = A.shape
    x = np.zeros(n)
    history = {'objective': [], 'residual': []}

    for k in range(max_iter):
        grad = grad_f(A, b, x)
        x_new = prox_l1(x - step_size * grad, lam * step_size)

        obj = lasso_objective(A, b, x_new, lam)
        res = np.linalg.norm(x_new - x)

        history['objective'].append(obj)
        history['residual'].append(res)

        if verbose:
            print(f"[ISTA] Iter {k:4d} | Obj: {obj:.6f} | Residual: {res:.2e}")

        if res < tol:
            break

        x = x_new

    return x, history
