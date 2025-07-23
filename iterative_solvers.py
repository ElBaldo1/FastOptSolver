from __future__ import annotations
import numpy as np
from objective_functions import compute_objective
from prox_operators import prox_l1, prox_elastic_net
from typing import Callable

# ---------------------------------------------------------------------
# Estimate Lipschitz constant L = λ_max(AᵀA)
# ---------------------------------------------------------------------
def estimate_lipschitz(A: np.ndarray, n_iter: int = 100, tol: float = 1e-6) -> float:
    """
    Estimate L = λ_max(AᵀA) via power iteration.
    """
    n = A.shape[1]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
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
# ISTA: Proximal Gradient Method
# ---------------------------------------------------------------------
def ista(
    x0: np.ndarray,
    g: Callable[[np.ndarray], float],
    grad_g: Callable[[np.ndarray], np.ndarray],
    prox_h: Callable[[np.ndarray, float], np.ndarray],
    L: float,
    backtracking: bool = False,
    t_init: float | None = None,
    eta: float = 0.5,
    max_iter: int = 500,
    tol: float = 0.0,
    return_history: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    x_{k+1} = prox_h(x_k - t ∇g(x_k), t)
    Optional Armijo backtracking. Stop when ∥grad∥ or ∥step∥ < tol.
    """
    x = x0.copy()
    t = t_init if t_init is not None else 1.0 / L
    log = {"x": [x.copy()], "t": [t], "delta": []} if return_history else None

    for _ in range(max_iter):
        grad = grad_g(x)
        if tol > 0.0 and np.linalg.norm(grad) < tol:
            break
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
        delta = np.linalg.norm(x_new - x)
        if tol > 0.0 and delta < tol:
            x = x_new
            if log:
                log["delta"].append(delta)
            break
        x = x_new
        if log:
            log["x"].append(x.copy())
            log["t"].append(t)
            log["delta"].append(delta)

    return (x, log) if return_history else x

# ---------------------------------------------------------------------
# FISTA: Accelerated Proximal Gradient with optional restart
# ---------------------------------------------------------------------
def fista(
    A: np.ndarray,
    b: np.ndarray,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    max_iter: int = 500,
    tol: float = 0.0,
    adaptive_restart: bool = False,
    restart_threshold: float = 1.0,
    return_history: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    FISTA for lasso, ridge, elastic-net:
      - gradient step includes ridge term if alpha2>0
      - prox step applies soft-thresholding with alpha1
      - optional adaptive restart
    Note: alpha1 and alpha2 should be pre-adjusted (small values set to zero) externally.
    """
    n = A.shape[1]
    x_k = np.zeros(n)
    y_k = x_k.copy()
    t_tilde = 1.0
    L_val = np.linalg.norm(A, ord=2)**2 + alpha2
    τ = 1.0 / L_val

    log = {"x": [x_k.copy()], "delta": [], "ratio": [], "obj": []} if return_history else None
    x_prev = x_k.copy()

    for k in range(max_iter):
        grad = A.T @ (A @ y_k - b)
        if alpha2 > 0:
            grad += alpha2 * y_k
        if tol > 0.0 and np.linalg.norm(grad) < tol:
            break

        v = y_k - τ * grad
        x_next = prox_l1(v, τ * alpha1) if alpha1 > 0 else v

        delta = np.linalg.norm(x_next - x_k)
        prev_delta = np.linalg.norm(x_k - x_prev) if k > 0 else 0.0
        ratio = delta / prev_delta if prev_delta > 0 else 0.0
        ip = (y_k - x_next).dot(x_next - x_k)

        restart = adaptive_restart and ((ratio > restart_threshold) or (ip > 0))
        if restart:
            t_next = 1.0
            y_next = x_next.copy()
        else:
            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_tilde**2))
            beta = (t_tilde - 1) / t_next
            y_next = x_next + beta * (x_next - x_k)

        r = A @ x_next - b
        obj = 0.5 * r.dot(r)
        if alpha2 > 0:
            obj += 0.5 * alpha2 * x_next.dot(x_next)
        if alpha1 > 0:
            obj += alpha1 * np.linalg.norm(x_next, 1)

        if k > 0 and tol > 0.0:
            rel = abs(obj - log["obj"][-1]) / max(1.0, abs(log["obj"][-1]))
            if rel < tol:
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

        x_prev, x_k, y_k, t_tilde = x_k, x_next, y_next, t_next
        if tol > 0.0 and delta < tol:
            break

    return (x_k, log) if return_history else x_k

# ---------------------------------------------------------------------
# FISTA-Δ: fixed momentum θ_k = k/(k+1+δ)
# ---------------------------------------------------------------------
def fista_delta(
    A: np.ndarray,
    b: np.ndarray,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    delta: float,
    max_iter: int = 500,
    tol: float = 0.0,
) -> tuple[np.ndarray, list]:
    """
    FISTA-Δ with θ_k = k / (k + 1 + δ).
    alpha1, alpha2 assumed pre-adjusted.
    """
    m, n = A.shape
    x_k = np.zeros(n)
    y_k = x_k.copy()
    L_val = estimate_lipschitz(A)
    τ = 1.0 / L_val
    objs = []

    for k in range(1, max_iter + 1):
        grad = A.T @ (A @ y_k - b)
        if alpha2 > 0:
            grad += alpha2 * y_k
        v = y_k - τ * grad
        x_next = prox_l1(v, τ * alpha1) if alpha1 > 0 else v
        θk = k / (k + 1 + delta)
        y_k = x_next + θk * (x_next - x_k)
        diff = np.linalg.norm(x_next - x_k)
        x_k = x_next
        objs.append(compute_objective(x_k, A, b, reg_type, alpha1, alpha2))
        if tol > 0.0 and diff < tol:
            break

    return x_k, objs
