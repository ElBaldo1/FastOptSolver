import numpy as np
from objective_functions import compute_objective
from prox_operators import select_prox_operator
from prox_operators import prox_l1

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
def fista(A, b, alpha1, alpha2,
          max_iter=500, tol=0.0,
          adaptive_restart=False, restart_threshold=1.0,
          return_history=False):
    """
    FISTA with ridge term inside g(x) and l1 inside h(x):
      g(x) = ½||Ax - b||^2 + ½ alpha2 ||x||^2
      h(x) = alpha1 ||x||_1
    """
    n      = A.shape[1]
    x_k    = np.zeros(n)
    y_k    = x_k.copy()
    t_tilde= 1.0

    # Lipschitz constant of ∇g = A^T A + alpha2 I
    L_A = np.linalg.norm(A, ord=2)**2
    L   = L_A + alpha2
    tau = 1.0 / L

    objs = []
    if return_history:
        step_norms = []
        ratio_vals = []
        x_hist     = [x_k.copy()]

    x_prev = x_k.copy()

    for k in range(max_iter):
        # 1) gradient step (include ridge)
        grad = A.T @ (A @ y_k - b) + alpha2 * y_k
        v    = y_k - tau * grad

        # 2) prox‐step (solo l1)
        x_next = prox_l1(v, tau * alpha1)

        # 3) misura progresso
        delta = np.linalg.norm(x_next - x_k)
        if return_history:
            prev_delta = np.linalg.norm(x_k - x_prev) if k>0 else 0.0
            ratio      = delta / prev_delta if prev_delta>0 else 0.0

        # 4) adaptive restart / momentum
        if adaptive_restart and return_history and ratio > restart_threshold:
            t_tilde_new = 1.0
            y_k         = x_next.copy()
        else:
            t_tilde_new = 0.5 * (1 + np.sqrt(1 + 4 * t_tilde**2))
            beta        = (t_tilde - 1) / t_tilde_new
            y_k         = x_next + beta * (x_next - x_k)

        # 5) update stato
        x_prev   = x_k.copy()
        x_k      = x_next
        t_tilde  = t_tilde_new

        # 6) objective = g + h
        r = A @ x_k - b
        g = 0.5 * r.dot(r) + 0.5 * alpha2 * x_k.dot(x_k)
        h = alpha1 * np.linalg.norm(x_k, 1)
        objs.append(g + h)

        if return_history:
            step_norms.append(delta)
            ratio_vals.append(ratio)
            x_hist.append(x_k.copy())

        # 7) stopping
        if tol > 0.0 and delta < tol:
            break

    if return_history:
        return x_k, objs, step_norms, ratio_vals, x_hist
    return x_k, objs

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