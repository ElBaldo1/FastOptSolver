import numpy as np
from objective_functions import compute_objective
from prox_operators import select_prox_operator

def estimate_lipschitz(A, n_iter=100, tol=1e-6):
    """
    Estimate the Lipschitz constant L of ∇f using power method.
    Approximates the largest eigenvalue of A^T A.
    """
    n = A.shape[1]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    L_old = 0.0
    for _ in range(n_iter):
        w = A.T @ (A @ v)
        L = np.linalg.norm(w)
        v = w / L
        if abs(L - L_old) < tol:
            break
        L_old = L
    return L

def ista(A, b, reg_type="lasso", alpha1=0.1, alpha2=0.1,
         max_iter=1000, tol=1e-6):
    """
    ISTA algorithm for solving:
        min_x ½||Ax − b||² + h(x)
    where h(x) = α1 * ||x||₁ [+ α2 * ||x||² for Elastic-Net]

    Parameters:
        A        : design matrix
        b        : target vector
        reg_type : "lasso" or "elasticnet"
        alpha1   : L1 regularization weight
        alpha2   : L2 regularization weight
        max_iter : maximum number of iterations
        tol      : stopping threshold based on ||x^{k+1} − x^k||

    Returns:
        x        : final iterate
        obj_vals : list of objective values per iteration
    """
    if reg_type == "ridge":
        raise ValueError("ISTA not suitable for smooth-only problems like Ridge. Use L-BFGS instead.")

    if reg_type == "elasticnet" and alpha1 < 1e-6:
        raise ValueError("ISTA/FISTA not suitable for Elastic Net with alpha1 ≈ 0 (smooth-only). Use L-BFGS.")

    m, n = A.shape
    x = np.zeros(n)                  # initial iterate
    L = estimate_lipschitz(A)       # compute Lipschitz constant of ∇f
    obj_vals = []

    for k in range(max_iter):
        grad = A.T @ (A @ x - b)     # gradient of smooth part g(x)
        v = x - grad / L             # gradient step
        x_new = select_prox_operator(v, 1/L,
                                     reg_type=reg_type,
                                     alpha1=alpha1,
                                     alpha2=alpha2)  # prox step

        obj_vals.append(
            compute_objective(x_new, A, b,
                              reg_type=reg_type,
                              alpha1=alpha1,
                              alpha2=alpha2)
        )

        # 🟡 STOPPING CRITERION: early termination if x stabilizes
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new  # update iterate

    return x, obj_vals

def fista(A, b, reg_type="lasso", alpha1=0.1, alpha2=0.1,
          max_iter=1000, tol=1e-6):
    """
    FISTA (accelerated ISTA) for Lasso or Elastic-Net problems.
    """
    if reg_type == "ridge":
        raise ValueError("FISTA not suitable for smooth-only problems like Ridge. Use L-BFGS instead.")

    if reg_type == "elasticnet" and alpha1 < 1e-6:
        raise ValueError("ISTA/FISTA not suitable for Elastic Net with alpha1 ≈ 0 (smooth-only). Use L-BFGS.")

    m, n = A.shape
    x = np.zeros(n)
    y = x.copy()
    t = 1.0
    L = estimate_lipschitz(A)

    obj_vals = []

    for k in range(max_iter):
        grad = A.T @ (A @ y - b)
        v = y - grad / L
        x_new = select_prox_operator(v, 1 / L,
                                     reg_type=reg_type,
                                     alpha1=alpha1,
                                     alpha2=alpha2)

        # 🟡 STOP: arresto se ||x_{k+1} - x_k|| < tol
        if np.linalg.norm(x_new - x) < tol:
            obj_vals.append(
                compute_objective(x_new, A, b,
                                  reg_type=reg_type,
                                  alpha1=alpha1,
                                  alpha2=alpha2)
            )
            break

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

        obj_vals.append(
            compute_objective(x, A, b,
                              reg_type=reg_type,
                              alpha1=alpha1,
                              alpha2=alpha2)
        )

    return x, obj_vals
