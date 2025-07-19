from __future__ import annotations

import numpy as np
from objective_functions import compute_objective
from prox_operators      import prox_l1, prox_elastic_net

# ---------------------------------------------------------------------
# Estimate the spectral norm — used as Lipschitz constant of ∇g
# ---------------------------------------------------------------------
def estimate_lipschitz(A, n_iter: int = 100, tol: float = 1e-6) -> float:
    """
    Estimate L = λ_max(AᵀA) via power iteration
    """
    n    = A.shape[1]
    v    = np.random.randn(n)
    v   /= np.linalg.norm(v)
    prev = 0.0
    for _ in range(n_iter):
        w = A.T @ (A @ v)
        L = np.linalg.norm(w)
        v = w / L
        if abs(L - prev) < tol:
            break
        prev = L
    return L

# ---------------------------------------------------------------------
# ISTA — Iterative Shrinkage-Thresholding Algorithm
# ---------------------------------------------------------------------
def ista(
    x0,
    g, grad_g, prox_h, L,
    backtracking: bool = False,
    t_init: float | None = None,
    eta: float = 0.5,
    max_iter: int = 500,
    tol: float = 0.0,
    return_history: bool = False,
):
    """
    Basic proximal gradient:
      x_{k+1} = prox_h(x_k - t ∇g(x_k), t)
    Optional Armijo backtracking. Stop when gradient norm or step size < tol.
    """
    x   = x0.copy()
    t   = t_init if t_init is not None else 1.0 / L
    log = {"x": [x.copy()], "t_val": [t], "delta": []} if return_history else None

    for k in range(max_iter):
        grad = grad_g(x)

        # Stop if gradient is small
        if tol > 0.0 and np.linalg.norm(grad) < tol:
            break

        # Gradient + prox update
        if backtracking:
            t_k = t
            while True:
                v = x - t_k * grad
                x_new = prox_h(v, t_k)
                diff = x_new - x
                if g(x_new) <= g(x) + grad.dot(diff) + np.linalg.norm(diff)**2 / (2 * t_k):
                    break
                t_k *= eta
            t = t_k
        else:
            v = x - t * grad
            x_new = prox_h(v, t)

        # Stop if update step is small
        delta = np.linalg.norm(x_new - x)
        if tol > 0.0 and delta < tol:
            x = x_new
            if log:
                log["delta"].append(delta)
            break

        x = x_new
        if log:
            log["x"].append(x.copy())
            log["t_val"].append(t)
            log["delta"].append(delta)

    return (x, log) if return_history else x

# ---------------------------------------------------------------------
# FISTA — Accelerated Proximal Gradient with optional restart
# ---------------------------------------------------------------------
def fista(
    A, b,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    max_iter: int = 500,
    tol: float = 0.0,
    adaptive_restart: bool = False,
    restart_threshold: float = 1.0,
    return_history: bool = False,
):
    """
    FISTA for lasso, ridge, elastic-net:
    - gradient of g includes ridge
    - proximal step uses soft-thresholding
    - optional restart on momentum
    """
    # choose effective weights, treat tiny alpha as zero
    eps = 1e-8
    if reg_type == "lasso":
        α1, α2 = alpha1, 0.0
    elif reg_type == "ridge":
        α1, α2 = 0.0, alpha2
    elif reg_type == "elasticnet":
        if alpha1 < eps:
            # effectively no L1 => ridge
            α1, α2 = 0.0, alpha2
        elif alpha2 < eps:
            # effectively no L2 => lasso
            α1, α2 = alpha1, 0.0
        else:
            α1, α2 = alpha1, alpha2
    else:
        raise ValueError(f"Unsupported reg_type='{reg_type}'")

    n       = A.shape[1]
    x_k     = np.zeros(n)
    y_k     = x_k.copy()
    t_tilde = 1.0

    # step size from Lipschitz constant
    L = np.linalg.norm(A, ord=2)**2 + α2
    τ = 1.0 / L

    log = {"x": [x_k.copy()], "delta": [], "ratio": [], "obj": []} if return_history else None
    x_prev = x_k.copy()

    for k in range(max_iter):
        # gradient step including ridge penalty
        grad = A.T @ (A @ y_k - b) + α2 * y_k
        if tol > 0.0 and np.linalg.norm(grad) < tol:
            break

        # proximal step (soft-threshold)
        v      = y_k - τ * grad
        x_next = prox_l1(v, τ * α1)

        # measure progress
        delta      = np.linalg.norm(x_next - x_k)
        prev_delta = np.linalg.norm(x_k - x_prev) if k > 0 else 0.0
        ratio      = delta / prev_delta if prev_delta > 0 else 0.0
        ip_test    = (y_k - x_next).dot(x_next - x_k)

        # adaptive restart if momentum bad
        restart = adaptive_restart and ((ratio > restart_threshold) or (ip_test > 0.0))
        if restart:
            t_tilde_new = 1.0
            y_next      = x_next.copy()
        else:
            t_tilde_new = 0.5 * (1 + np.sqrt(1 + 4 * t_tilde**2))
            beta        = (t_tilde - 1) / t_tilde_new
            y_next      = x_next + beta * (x_next - x_k)

        # objective for relative drop stopping
        r     = A @ x_next - b
        g_val = 0.5 * r.dot(r) + 0.5 * α2 * x_next.dot(x_next)
        h_val = α1 * np.linalg.norm(x_next, 1)
        obj   = g_val + h_val
        if k > 0 and tol > 0.0:
            rel_drop = abs(obj - log["obj"][-1]) / max(1.0, abs(log["obj"][-1]))
            if rel_drop < tol:
                if log:
                    log["obj"].append(obj)
                    log["delta"].append(delta)
                    log["ratio"].append(ratio)
                    log["x"].append(x_next.copy())
                break

        if log:
            log["obj"].append(obj)
            log["delta"].append(delta)
            log["ratio"].append(ratio)
            log["x"].append(x_next.copy())

        # update for next iteration
        x_prev, x_k, y_k, t_tilde = x_k, x_next, y_next, t_tilde_new
        if tol > 0.0 and delta < tol:
            break

    return (x_k, log) if return_history else x_k

# ---------------------------------------------------------------------
# FISTA-Δ — fixed momentum θ_k = k/(k+1+δ)
# ---------------------------------------------------------------------
def fista_delta(
    A, b,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    delta: float,
    max_iter: int = 500,
    tol: float = 0.0,
):
    """
    FISTA-Δ with θ_k = k / (k + 1 + δ)
    """
    m, n = A.shape
    x_k  = np.zeros(n)
    y_k  = x_k.copy()
    L    = estimate_lipschitz(A)
    τ    = 1.0 / L
    objs = []

    for k in range(1, max_iter + 1):
        grad = A.T @ (A @ y_k - b)
        v    = y_k - τ * grad
        if reg_type == "lasso":
            x_next = prox_l1(v, τ * alpha1)
        else:
            x_next = prox_elastic_net(v, τ, alpha1, alpha2)

        θ_k = k / (k + 1 + delta)
        y_k = x_next + θ_k * (x_next - x_k)

        delta_norm = np.linalg.norm(x_next - x_k)
        x_k = x_next
        objs.append(compute_objective(x_k, A, b, reg_type, alpha1, alpha2))
        if tol > 0.0 and delta_norm < tol:
            break

    return x_k, objs
