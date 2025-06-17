import numpy as np
from objective_functions import compute_objective
from prox_operators import select_prox_operator

def estimate_lipschitz(A, n_iter=100, tol=1e-6):
    """
    Power‐method estimate of L = lambda_max(A^T A)
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
         max_iter=1000):
    """
    ISTA for either Lasso or Elastic‐Net:
      minimize ½||Ax−b||² + h(x)
    where h(x)=alpha1||x||₁ (+ alpha2||x||² for elasticnet).
    """
    m, n = A.shape
    x = np.zeros(n)
    L = estimate_lipschitz(A)

    obj_vals = []
    for k in range(max_iter):
        grad = A.T @ (A @ x - b)
        v = x - grad / L
        x = select_prox_operator(v, 1/L,
                                 reg_type=reg_type,
                                 alpha1=alpha1,
                                 alpha2=alpha2)
        obj_vals.append(
            compute_objective(x, A, b,
                              reg_type=reg_type,
                              alpha1=alpha1,
                              alpha2=alpha2)
        )
    return x, obj_vals

def fista(A, b, reg_type="lasso", alpha1=0.1, alpha2=0.1,
          max_iter=1000):
    """
    FISTA (accelerated ISTA) for Lasso/Elastic‐Net.
    """
    m, n = A.shape
    x = np.zeros(n)
    y = x.copy()
    t = 1.0
    L = estimate_lipschitz(A)

    obj_vals = []
    for k in range(max_iter):
        grad = A.T @ (A @ y - b)
        v = y - grad / L
        x_new = select_prox_operator(v, 1/L,
                                     reg_type=reg_type,
                                     alpha1=alpha1,
                                     alpha2=alpha2)

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t*t))
        y = x_new + ((t - 1)/t_new)*(x_new - x)

        x = x_new
        t = t_new

        obj_vals.append(
            compute_objective(x, A, b,
                              reg_type=reg_type,
                              alpha1=alpha1,
                              alpha2=alpha2)
        )
    return x, obj_vals
