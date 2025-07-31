from __future__ import annotations
import time
import numpy as np
from objective_functions import compute_objective
from prox_operators import prox_l1, prox_elastic_net
from typing import Callable

# ---------------------------------------------------------------------
# METRICS INFRASTRUCTURE FOR ISTA/FISTA TIMING
# ---------------------------------------------------------------------
grad_call_times: list[float] = []
ls_call_times: list[float] = []
ls_call_iters: list[int]   = []

def get_metrics():
    """
    Return timing statistics for ISTA/FISTA:
      - total/mean gradient evaluation time
      - total/mean line-search time and total line-search iterations
    """
    return {
        'grad_num_calls':   len(grad_call_times),
        'grad_time_total':  sum(grad_call_times),
        'grad_time_mean':   np.mean(grad_call_times) if grad_call_times else 0.0,
        'ls_num_calls':     len(ls_call_times),
        'ls_time_total':    sum(ls_call_times),
        'ls_time_mean':     np.mean(ls_call_times) if ls_call_times else 0.0,
        'ls_iters_total':   sum(ls_call_iters),
    }

# ---------------------------------------------------------------------
# Default line-search parameters
# ---------------------------------------------------------------------
DEFAULT_C1 = 1e-4    # Armijo sufficient-decrease
DEFAULT_C2 = 0.9     # Wolfe curvature
DEFAULT_ALPHA = 0.5  # Backtracking contraction

# ---------------------------------------------------------------------
# Armijo backtracking line search
# ---------------------------------------------------------------------
def armijo_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    d: np.ndarray,
    t_init: float = 1.0
) -> tuple[float, int]:
    """
    Armijo search: returns step t and iteration count
    """
    t = t_init
    iters = 0
    fx = f(x)
    g0 = grad_f(x).dot(d)
    while f(x + t*d) > fx + DEFAULT_C1 * t * g0:
        t *= DEFAULT_ALPHA
        iters += 1
    return t, iters

# ---------------------------------------------------------------------
# Strong Wolfe backtracking line search
# ---------------------------------------------------------------------
def wolfe_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    d: np.ndarray,
    t_init: float = 1.0
) -> tuple[float, int]:
    """
    Strong Wolfe search: returns step t and iteration count
    """
    t = t_init
    iters = 0
    fx = f(x)
    g0 = grad_f(x).dot(d)
    while True:
        x_new = x + t*d
        if (f(x_new) <= fx + DEFAULT_C1 * t * g0 and
            abs(grad_f(x_new).dot(d)) <= DEFAULT_C2 * abs(g0)):
            break
        t *= DEFAULT_ALPHA
        iters += 1
    return t, iters

# ---------------------------------------------------------------------
# Estimate Lipschitz constant L = λ_max(AᵀA)
# ---------------------------------------------------------------------
def estimate_lipschitz(A: np.ndarray, n_iter: int = 100, tol: float = 1e-6) -> float:
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
# ISTA: Proximal Gradient with selectable step-size
# ---------------------------------------------------------------------
def ista(
    x0: np.ndarray,
    g: Callable[[np.ndarray], float],
    grad_g: Callable[[np.ndarray], np.ndarray],
    prox_h: Callable[[np.ndarray, float], np.ndarray],
    L: float,
    step_method: str = "fixed",    # 'fixed', 'armijo', or 'wolfe'
    max_iter: int = 500,
    tol: float = 0.0,
    return_history: bool = False
) -> tuple[np.ndarray, dict] | np.ndarray:
    x = x0.copy()
    t_fixed = 1.0 / L
    history = {"x": [x.copy()], "t": [], "delta": []} if return_history else None

    for _ in range(max_iter):
        # gradient
        t0 = time.perf_counter()
        grad = grad_g(x)
        grad_call_times.append(time.perf_counter() - t0)
        if tol > 0 and np.linalg.norm(grad) < tol:
            break
        # choose step-size
        d = -grad
        if step_method == 'armijo':
            ls_t0 = time.perf_counter()
            t, iters = armijo_line_search(g, grad_g, x, d, t_fixed)
            ls_call_times.append(time.perf_counter() - ls_t0)
            ls_call_iters.append(iters)
        elif step_method == 'wolfe':
            ls_t0 = time.perf_counter()
            t, iters = wolfe_line_search(g, grad_g, x, d, t_fixed)
            ls_call_times.append(time.perf_counter() - ls_t0)
            ls_call_iters.append(iters)
        else:
            t = t_fixed
        # update
        x_new = prox_h(x + t*d, t)
        delta = np.linalg.norm(x_new - x)
        x = x_new
        if return_history:
            history["x"].append(x.copy())
            history["t"].append(t)
            history["delta"].append(delta)
        if tol > 0 and delta < tol:
            break

    return (x, history) if return_history else x

# ---------------------------------------------------------------------
# FISTA: Accelerated Proximal with selectable step-size
# ---------------------------------------------------------------------
def fista(
    A: np.ndarray,
    b: np.ndarray,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    step_method: str = "fixed",   # 'fixed' or 'armijo'
    max_iter: int = 500,
    tol: float = 0.0,
    adaptive_restart: bool = False,
    restart_threshold: float = 1.0,
    return_history: bool = False
) -> tuple[np.ndarray, dict] | np.ndarray:
    n = A.shape[1]
    x_k = np.zeros(n)
    y_k = x_k.copy()
    t_prev = 1.0
    L_val = np.linalg.norm(A, 2)**2 + alpha2
    tau_fixed = 1.0 / L_val
    history = {"x": [x_k.copy()], "obj": [], "delta": []} if return_history else None

    def g_s(z: np.ndarray) -> float:
        r = A @ z - b
        val = 0.5 * r.dot(r)
        if alpha2 > 0:
            val += 0.5 * alpha2 * z.dot(z)
        return val

    def grad_s(z: np.ndarray) -> np.ndarray:
        g = A.T @ (A @ z - b)
        return g + (alpha2 * z if alpha2 > 0 else 0)

    for k in range(max_iter):
        # gradient
        t0 = time.perf_counter()
        grad = grad_s(y_k)
        grad_call_times.append(time.perf_counter() - t0)
        if tol > 0 and np.linalg.norm(grad) < tol:
            break
        # select tau
        d = -grad
        if step_method == 'armijo':
            ls_t0 = time.perf_counter()
            tau, iters = armijo_line_search(g_s, grad_s, y_k, d, tau_fixed)
            ls_call_times.append(time.perf_counter() - ls_t0)
            ls_call_iters.append(iters)
        else:
            tau = tau_fixed
        # prox
        v = y_k + tau*d
        x_next = prox_l1(v, tau*alpha1) if alpha1>0 else v
        delta = np.linalg.norm(x_next - x_k)
        # restart/momentum
        if adaptive_restart and return_history and k>0 and delta > history['delta'][-1]:
            t_curr = 1.0
            y_k = x_next.copy()
        else:
            t_curr = 0.5*(1+np.sqrt(1+4*t_prev**2))
            beta   = (t_prev-1)/t_curr
            y_k    = x_next + beta*(x_next-x_k)
        # record
        if return_history:
            history['x'].append(x_next.copy())
            history['obj'].append(compute_objective(x_next,A,b,reg_type,alpha1,alpha2))
            history['delta'].append(delta)
        # prepare next
        x_k, t_prev = x_next, t_curr
        if tol>0 and delta<tol:
            break

    return (x_k, history) if return_history else x_k

# ---------------------------------------------------------------------
# FISTA-Δ: fixed momentum θ_k = k/(k+1+δ) with selectable step-size
# ---------------------------------------------------------------------
def fista_delta(
    A: np.ndarray,
    b: np.ndarray,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    delta: float,
    step_method: str = "fixed",   # 'fixed' or 'armijo'
    max_iter: int = 500,
    tol: float = 0.0,
    return_history: bool = False
) -> tuple[np.ndarray, dict] | np.ndarray:
    m,n = A.shape
    x_k = np.zeros(n)
    y_k = x_k.copy()
    L_val = estimate_lipschitz(A) + (2*alpha2 if alpha2>0 else 0)
    tau_fixed = 1.0 / L_val
    history = {"x": [], "obj": []} if return_history else None

    def g_s(z: np.ndarray) -> float:
        r = A @ z - b
        val = 0.5*r.dot(r)
        if alpha2>0: val += 0.5*alpha2*z.dot(z)
        return val

    for k in range(1, max_iter+1):
        t0 = time.perf_counter()
        grad = A.T@(A@y_k - b) + (2*alpha2*y_k if alpha2>0 else 0)
        grad_call_times.append(time.perf_counter() - t0)
        # step-size
        d = -grad
        if step_method=='armijo':
            ls_t0 = time.perf_counter()
            tau, iters = armijo_line_search(g_s, lambda z: A.T@(A@z-b)+(2*alpha2*z if alpha2>0 else 0), y_k, d, tau_fixed)
            ls_call_times.append(time.perf_counter()-ls_t0)
            ls_call_iters.append(iters)
        else:
            tau = tau_fixed
        # prox & momentum
        v = y_k + tau*d
        x_next = prox_l1(v, tau*alpha1) if alpha1>0 else v
        if return_history:
            history['x'].append(x_next.copy())
            history['obj'].append(compute_objective(x_next, A, b, reg_type, alpha1, alpha2))
        theta = k/(k+1+delta)
        y_k = x_next + theta*(x_next - x_k)
        step_norm = np.linalg.norm(x_next - x_k)
        x_k = x_next
        if tol>0 and step_norm<tol:
            break

    return (x_k, history) if return_history else x_k
