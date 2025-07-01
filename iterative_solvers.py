
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

def ista(A, b,
         reg_type="lasso",
         alpha1=0.1,
         alpha2=0.1,
         max_iter=1000,
         tol=1e-6,
         return_history=False):
    m, n = A.shape
    x = np.zeros(n)
    if return_history:
        x_hist = [x.copy()]
    L = estimate_lipschitz(A)

    obj_vals   = []
    step_norms = []

    for k in range(max_iter):
        grad = A.T @ (A @ x - b)
        v = x - grad / L
        x_new = select_prox_operator(v, 1/L,
                                     reg_type=reg_type,
                                     alpha1=alpha1,
                                     alpha2=alpha2)

        delta = np.linalg.norm(x_new - x)
        step_norms.append(delta)

        x = x_new
        if return_history:
            x_hist.append(x.copy())

        # registro l’oggettivo **dopo** l’update (facoltativo se vogliamo oggetti a *k+1*)
        obj_vals.append(
            compute_objective(x, A, b,
                              reg_type=reg_type,
                              alpha1=alpha1,
                              alpha2=alpha2)
        )

        if delta < tol:
            break

    if return_history:
        return x, obj_vals, step_norms, x_hist
    else:
        return x, obj_vals, step_norms

def fista(A, b,
          reg_type="lasso",
          alpha1=0.1,
          alpha2=0.1,
          max_iter=1000,
          tol=1e-6,
          adaptive_restart=True,
          restart_threshold=1.0,
          return_history=False):
    m, n = A.shape
    x = np.zeros(n)
    if return_history:
        x_hist = [x.copy()]
    y = x.copy()
    t = 1.0
    L = estimate_lipschitz(A)

    obj_vals   = []
    step_norms = []
    ratio_vals = []

    x_prev = x.copy()
    for k in range(max_iter):
        grad = A.T @ (A @ y - b)
        v = y - grad / L
        x_new = select_prox_operator(v, 1/L,
                                     reg_type=reg_type,
                                     alpha1=alpha1,
                                     alpha2=alpha2)

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t*t))

        delta = np.linalg.norm(x_new - x)
        step_norms.append(delta)

        if k > 0:
            prev_delta = np.linalg.norm(x - x_prev)
            ratio = delta / prev_delta if prev_delta>0 else 0.0
        else:
            ratio = 0.0
        ratio_vals.append(ratio)

        if adaptive_restart and ratio > restart_threshold:
            t_new = 1.0
            y = x_new.copy()
        else:
            y = x_new + ((t - 1)/t_new)*(x_new - x)

        x_prev = x.copy()
        x      = x_new
        t      = t_new
        if return_history:
            x_hist.append(x.copy())

        obj_vals.append(
            compute_objective(x, A, b,
                              reg_type=reg_type,
                              alpha1=alpha1,
                              alpha2=alpha2)
        )

        if delta < tol:
            break

    if return_history:
        return x, obj_vals, step_norms, ratio_vals, x_hist
    else:
        return x, obj_vals, step_norms, ratio_vals