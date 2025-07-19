import numpy as np
from objective_functions import compute_objective
from prox_operators import prox_l1, prox_elastic_net

# Estimate Lipschitz constant L of ∇g via power method on AᵀA
def estimate_lipschitz(A, n_iter=100, tol=1e-6):
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

# --- ISTA: Proximal Gradient Method ---
def ista(
    x0,            # initial guess x₀
    g,             # smooth function g(x)
    grad_g,        # gradient ∇g(x)
    prox_h,        # proximal operator for h: prox_h(v, t)
    L,             # Lipschitz constant of ∇g
    backtracking=False,
    t_init=None,
    eta=0.5,
    max_iter=500,
    tol=0.0,
    return_history=False
):
    """
    ISTA algorithm:
      x_{k+1} = prox_h(x_k - t_k ∇g(x_k), t_k)
    - Fixed step: t_k = 1/L
    - Backtracking: Armijo condition on g
    """
    x = x0.copy()
    # Initialize step size
    t = t_init if t_init is not None else 1.0 / L
    # History storage
    if return_history:
        xs = [x.copy()]
        ts = [t]
        deltas = []

    for k in range(max_iter):
        grad = grad_g(x)
        # 1) Gradient step
        if backtracking:
            t_k = t
            while True:
                v = x - t_k * grad
                x_new = prox_h(v, t_k)
                # Armijo-like condition for g:
                if g(x_new) <= g(x) + grad.dot(x_new - x) + np.linalg.norm(x_new - x)**2 / (2 * t_k):
                    break
                t_k *= eta
            t = t_k
        else:
            v = x - t * grad
            x_new = prox_h(v, t)
        # 2) Shrinkage step via prox_h
        delta = np.linalg.norm(x_new - x)
        x = x_new
        # Record history
        if return_history:
            xs.append(x.copy())
            ts.append(t)
            deltas.append(delta)
        # 3) Stopping criterion
        if tol > 0.0 and delta < tol:
            break

    if return_history:
        return x, {'x': xs, 'step_sizes': deltas, 't_values': ts}
    return x

# --- FISTA: Accelerated ISTA with optional adaptive restart ---
def fista(
    A, b, reg_type, alpha1, alpha2,
    max_iter=500, tol=0.0,
    adaptive_restart=False, restart_threshold=1.0,
    return_history=False
):
    """
    FISTA for Lasso/Elastic-Net:
      - adaptive_restart resets momentum when ratio>threshold
      - records objective history
    """
    m, n = A.shape
    x = np.zeros(n)
    y = x.copy()
    t = 1.0
    L = estimate_lipschitz(A)
    objs = []
    if return_history:
        steps, ratios, xs = [], [], [x.copy()]
    x_prev = x.copy()

    for k in range(max_iter):
        grad = A.T @ (A @ y - b)
        v = y - grad / L
        # prox step for chosen regularizer
        if reg_type == 'lasso':
            x_new = prox_l1(v, alpha1 / L)
        else:
            x_new = prox_elastic_net(v, 1.0 / L, alpha1, alpha2)
        # compute step norm
        delta = np.linalg.norm(x_new - x)
        if return_history:
            prev_delta = np.linalg.norm(x - x_prev) if k > 0 else 0.0
            ratio = delta / prev_delta if prev_delta > 0 else 0.0
        # adaptive restart logic
        if adaptive_restart and return_history and ratio > restart_threshold:
            t_new = 1.0
            y = x_new.copy()
        else:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            y = x_new + ((t - 1) / t_new) * (x_new - x)
        x_prev = x.copy()
        x, t = x_new, t_new
        objs.append(compute_objective(x, A, b, reg_type, alpha1, alpha2))
        if return_history:
            steps.append(delta)
            ratios.append(ratio)
            xs.append(x.copy())
        if tol > 0.0 and delta < tol:
            break

    if return_history:
        return x, objs, steps, ratios, xs
    return x, objs

# --- FISTA-Delta: Variant with fixed inertial parameter ---
def fista_delta(
    A, b, reg_type, alpha1, alpha2,
    delta, max_iter=500, tol=0.0
):
    """
    FISTA-Δ: inertial weight θ_k = k/(k+1+δ)
    """
    m, n = A.shape
    x = np.zeros(n)
    y = x.copy()
    L = estimate_lipschitz(A)
    objs = []

    for k in range(1, max_iter + 1):
        grad = A.T @ (A @ y - b)
        v = y - grad / L
        if reg_type == 'lasso':
            x_new = prox_l1(v, alpha1 / L)
        else:
            x_new = prox_elastic_net(v, 1.0 / L, alpha1, alpha2)
        theta = k / (k + 1 + delta)
        y = x_new + theta * (x_new - x)
        delta_norm = np.linalg.norm(x_new - x)
        x = x_new
        objs.append(compute_objective(x, A, b, reg_type, alpha1, alpha2))
        if tol > 0.0 and delta_norm < tol:
            break

    return x, objs