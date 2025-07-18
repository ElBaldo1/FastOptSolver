import numpy as np
from objective_functions import compute_objective
from prox_operators import select_prox_operator

def estimate_lipschitz(A, n_iter=100, tol=1e-6):
    """Power method to estimate λ_max(AᵀA)."""
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

# --- ISTA with optional stopping & history support ---
def ista(A, b, reg_type, alpha1, alpha2, max_iter=500, tol=0.0, return_history=False):
    """
    ISTA algorithm:
      min_x ½||Ax - b||² + h(x)
    Parameters:
      tol: early stopping threshold on ||x_{k+1}-x_k|| (use tol=0 for no stopping)
      return_history: if True, returns step_norms and x_history
    Returns:
      x, objs [, step_norms, x_history]
    """
    x = np.zeros(A.shape[1])
    L = estimate_lipschitz(A)
    objs = []
    if return_history:
        step_norms = []
        x_hist = [x.copy()]

    for k in range(max_iter):
        grad = A.T @ (A @ x - b)
        v = x - grad / L
        x_new = select_prox_operator(v, 1.0/L, reg_type, alpha1, alpha2)
        delta = np.linalg.norm(x_new - x)
        x = x_new
        objs.append(compute_objective(x, A, b, reg_type, alpha1, alpha2))

        if return_history:
            step_norms.append(delta)
            x_hist.append(x.copy())
        if tol > 0.0 and delta < tol:
            break

    if return_history:
        return x, objs, step_norms, x_hist
    else:
        return x, objs

# --- FISTA with optional adaptive restart, stopping & history support ---
def fista(A, b, reg_type, alpha1, alpha2, max_iter=500,
          tol=0.0, adaptive_restart=False, restart_threshold=1.0,
          return_history=False):
    """
    FISTA algorithm with:
      - adaptive_restart: if True, resets momentum when ratio > restart_threshold
      - tol: stopping threshold on ||x_{k+1}-x_k||; tol=0 disables
      - return_history: returns step_norms, ratio_vals, x_history if True
    Returns:
      x, objs [, step_norms, ratio_vals, x_history]
    """
    x = np.zeros(A.shape[1])
    y = x.copy()
    t = 1.0
    L = estimate_lipschitz(A)
    objs = []
    if return_history:
        step_norms = []
        ratio_vals = []
        x_hist = [x.copy()]
    x_prev = x.copy()

    for k in range(max_iter):
        grad = A.T @ (A @ y - b)
        v = y - grad / L
        x_new = select_prox_operator(v, 1.0/L, reg_type, alpha1, alpha2)
        delta = np.linalg.norm(x_new - x)
        # ratio for adaptive restart
        if return_history:
            prev_delta = np.linalg.norm(x - x_prev) if k>0 else 0.0
            ratio = delta/prev_delta if prev_delta>0 else 0.0

        # momentum update
        if adaptive_restart and return_history and 'ratio' in locals() and ratio > restart_threshold:
            t_new = 1.0
            y = x_new.copy()
        else:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = x_new + ((t - 1)/t_new) * (x_new - x)

        # record
        x_prev = x.copy()
        x = x_new
        t = t_new
        objs.append(compute_objective(x, A, b, reg_type, alpha1, alpha2))
        if return_history:
            step_norms.append(delta)
            ratio_vals.append(ratio)
            x_hist.append(x.copy())

        # stopping
        if tol > 0.0 and delta < tol:
            break

    if return_history:
        return x, objs, step_norms, ratio_vals, x_hist
    else:
        return x, objs

# --- FISTA-Delta variant with fixed inertial parameter ---
def fista_delta(A, b, reg_type, alpha1, alpha2, delta, max_iter=500, tol=0.0):
    """
    FISTA-Delta: uses fixed momentum coefficient θ_k = k/(k+1+δ)
    Parameters:
      delta: parameter > 2 controlling inertia
      tol: stopping threshold on ||x_{k+1}-x_k||
    Returns:
      x, objs
    """
    x = np.zeros(A.shape[1])
    y = x.copy()
    L = estimate_lipschitz(A)
    objs = []

    for k in range(1, max_iter+1):
        grad = A.T @ (A @ y - b)
        v = y - grad / L
        x_new = select_prox_operator(v, 1.0/L, reg_type, alpha1, alpha2)
        # compute inertial weight
        theta = k / (k + 1 + delta)
        y = x_new + theta * (x_new - x)
        delta_norm = np.linalg.norm(x_new - x)
        x = x_new
        objs.append(compute_objective(x, A, b, reg_type, alpha1, alpha2))
        if tol > 0.0 and delta_norm < tol:
            break
    return x, objs