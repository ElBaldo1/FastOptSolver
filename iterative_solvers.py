from __future__ import annotations
import time
import numpy as np
from objective_functions import compute_objective
from prox_operators import prox_l1, prox_elastic_net
from typing import Callable

# ---------------------------------------------------------------------
# METRICS INFRASTRUCTURE FOR ISTA TIMING
# ---------------------------------------------------------------------
grad_call_times = []
ls_call_times   = []
ls_call_iters   = []

def get_metrics():
    """
    Return timing statistics for ISTA:
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
# ISTA: Proximal Gradient Method with optional backtracking & timing
# ---------------------------------------------------------------------
def ista(
    x0: np.ndarray,
    g: Callable[[np.ndarray], float],
    grad_g: Callable[[np.ndarray], np.ndarray],
    prox_h: Callable[[np.ndarray, float], np.ndarray],
    L: float,
    backtracking: bool = False,
    eta: float = 0.5,
    max_iter: int = 500,
    tol: float = 0.0,
    return_history: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    ISTA (Iterative Soft-Thresholding / Proximal Gradient):
      x_{k+1} = prox_h(x_k - t ∇g(x_k), t)
    Parameters:
      x0           - initial iterate
      g            - smooth part of objective
      grad_g       - gradient of g
      prox_h       - proximal map of non-smooth part
      L            - Lipschitz constant of ∇g
      backtracking - whether to perform Armijo backtracking
      eta          - backtracking shrinkage factor
      max_iter     - maximum number of iterations
      tol          - tolerance on step norm for stopping
      return_history - if True, return dict with iterate history
    Timing:
      Records time for each gradient evaluation in grad_call_times,
      and time + iteration count for each backtracking line-search
      in ls_call_times and ls_call_iters.
    """
    x = x0.copy()
    # initial step-size
    t = 1.0 / L
    # prepare history log if requested
    log = {"x": [x.copy()], "t": [t], "delta": []} if return_history else None

    for _ in range(max_iter):
        # 1) timed gradient evaluation
        t0 = time.perf_counter()
        grad = grad_g(x)
        grad_call_times.append(time.perf_counter() - t0)

        # 2) optionally backtrack + record line-search timing & iterations
        if backtracking:
            bt_steps = 0
            ls_t0    = time.perf_counter()
            t_k      = t
            while True:
                v      = x - t_k * grad
                x_new  = prox_h(v, t_k)
                diff   = x_new - x
                # Armijo condition
                if g(x_new) <= g(x) + grad.dot(diff) + np.linalg.norm(diff)**2 / (2*t_k):
                    break
                t_k   *= eta
                bt_steps += 1
            # record line-search stats
            ls_call_times.append(time.perf_counter() - ls_t0)
            ls_call_iters.append(bt_steps)
            t = t_k
        else:
            v     = x - t * grad
            x_new = prox_h(v, t)

        # 3) update and check convergence
        delta = np.linalg.norm(x_new - x)
        x     = x_new

        if return_history:
            log["x"].append(x.copy())
            log["t"].append(t)
            log["delta"].append(delta)

        if tol > 0.0 and delta < tol:
            break

    return (x, log) if return_history else x

# ---------------------------------------------------------------------
# FISTA: Accelerated Proximal Gradient with optional backtracking,
#        adaptive restart, and stopping on gradient norm, step norm,
#        or step‐ratio norm.
# ---------------------------------------------------------------------
def fista(
    A: np.ndarray,
    b: np.ndarray,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    backtracking: bool        = False,
    eta: float                = 0.5,
    max_iter: int             = 500,
    tol: float                = 0.0,
    tol_ratio: float          = 0.0,
    adaptive_restart: bool    = False,
    restart_threshold: float  = 1.0,
    return_history: bool      = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    FISTA for lasso, ridge, elastic-net:
      1) Compute gradient ∇g(y_k) = Aᵀ(A y_k – b) + α₂ y_k
      2) Optionally backtracking line-search on g_smooth
      3) Proximal step: x_next = prox_l1(y_k – τ∇g, τ*α₁)
      4) Nesterov momentum update with optional adaptive restart
      5) Stop when any of:
           - ∥grad∥ < tol
           - ∥x^{k+1}-x^k∥ < tol
           - ∥x^{k+1}-x^k∥/∥x^k-x^{k-1}∥ < tol_ratio
    Timing:
      - Records gradient durations in grad_call_times
      - Records line-search durations in ls_call_times and iterations in ls_call_iters
    """
    n      = A.shape[1]
    x_k    = np.zeros(n)
    y_k    = x_k.copy()
    t_prev = 1.0

    # Lipschitz L = ∥A∥² + α₂
    L_val = np.linalg.norm(A, 2)**2 + alpha2
    tau   = 1.0 / L_val

    history = {"x": [x_k.copy()], "obj": []} if return_history else None
    x_prev  = x_k.copy()

    def g_smooth(z: np.ndarray) -> float:
        """Smooth part g(z) = ½∥Az - b∥² + (α₂/2)∥z∥²."""
        r = A @ z - b
        val = 0.5 * r.dot(r)
        if alpha2 > 0:
            val += 0.5 * alpha2 * z.dot(z)
        return val

    for k in range(max_iter):
        # 1) timed gradient evaluation
        t0 = time.perf_counter()
        grad = A.T @ (A @ y_k - b)
        if alpha2 > 0:
            grad += alpha2 * y_k
        grad_call_times.append(time.perf_counter() - t0)

        # 2) stopping on gradient norm
        if tol > 0.0 and np.linalg.norm(grad) < tol:
            break

        # 3) optional backtracking line-search
        if backtracking:
            bt_steps = 0
            ls_t0    = time.perf_counter()
            t_k      = tau
            while True:
                v_tmp = y_k - t_k * grad
                x_tmp = prox_l1(v_tmp, t_k * alpha1) if alpha1 > 0 else v_tmp
                diff  = x_tmp - y_k
                if g_smooth(x_tmp) <= g_smooth(y_k) + grad.dot(diff) + np.linalg.norm(diff)**2/(2*t_k):
                    break
                t_k *= eta
                bt_steps += 1
            ls_call_times.append(time.perf_counter() - ls_t0)
            ls_call_iters.append(bt_steps)
            tau = t_k

        # 4) proximal update
        v      = y_k - tau * grad
        x_next = prox_l1(v, tau * alpha1) if alpha1 > 0 else v

        # 5) compute step‐norm and ratio
        this_step = np.linalg.norm(x_next - x_k)
        prev_step = np.linalg.norm(x_k     - x_prev)
        ratio     = this_step / prev_step if prev_step > 0 else np.inf

        # 6) adaptive restart if requested
        if adaptive_restart:
            if ratio > restart_threshold:
                # reset momentum
                t_curr = 1.0
                y_next = x_next.copy()
            else:
                t_curr = 0.5 * (1 + np.sqrt(1 + 4*t_prev**2))
                beta   = (t_prev - 1) / t_curr
                y_next = x_next + beta * (x_next - x_k)
        else:
            t_curr = 0.5 * (1 + np.sqrt(1 + 4*t_prev**2))
            beta   = (t_prev - 1) / t_curr
            y_next = x_next + beta * (x_next - x_k)

        # 7) record objective if requested
        if return_history:
            r   = A @ x_next - b
            obj = 0.5 * r.dot(r)
            if alpha2 > 0:
                obj += 0.5 * alpha2 * x_next.dot(x_next)
            if alpha1 > 0:
                obj += alpha1 * np.linalg.norm(x_next, 1)
            history["obj"].append(obj)
            history["x"].append(x_next.copy())

        # 8) update iterate history
        x_prev, x_k, y_k, t_prev = x_k, x_next, y_next, t_curr

        # 9) stopping on step‐norm
        if tol > 0.0 and this_step < tol:
            break

        # 10) stopping on step‐ratio
        if tol_ratio > 0.0 and ratio < tol_ratio:
            break

    return (x_k, history) if return_history else x_k


# ---------------------------------------------------------------------
# FISTA-Δ: fixed momentum θ_k = k/(k+1+δ), same stopping options
# ---------------------------------------------------------------------
def fista_delta(
    A: np.ndarray,
    b: np.ndarray,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    delta: float,
    backtracking: bool     = False,
    eta: float             = 0.5,
    max_iter: int          = 500,
    tol: float             = 0.0,
    tol_ratio: float       = 0.0,
    return_history: bool   = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    FISTA-Δ solver with fixed θ_k = k/(k+1+δ):
      1) ∇g(y_k) = Aᵀ(A y_k – b) + 2α₂ y_k (if elastic-net)
      2) Optional backtracking on g_smooth
      3) Proximal x_next = prox_l1(y_k – τ∇g, τ*α₁)
      4) Momentum y_{k+1} = x_next + θ_k (x_next – x_k)
      5) Stop when any of:
           - ∥x_next – x_k∥ < tol
           - ∥x_next – x_k∥/∥x_k – x_{k-1}∥ < tol_ratio
    Timing stats as above.
    """
    m, n = A.shape
    x_k  = np.zeros(n)
    y_k  = x_k.copy()

    # Lipschitz estimate λ_max(AᵀA) + 2α₂
    L_val = estimate_lipschitz(A)
    if reg_type == 'elasticnet' and alpha2 > 0:
        L_val += 2*alpha2
    tau = 1.0 / L_val

    history = {"x": [], "obj": []} if return_history else None
    x_prev = x_k.copy()

    def g_smooth(z: np.ndarray) -> float:
        """Smooth part for FISTA-Δ."""
        r = A @ z - b
        val = 0.5 * r.dot(r)
        if alpha2 > 0:
            val += 0.5 * alpha2 * z.dot(z)
        return val

    for k in range(1, max_iter+1):
        # 1) timed gradient
        t0 = time.perf_counter()
        grad = A.T @ (A @ y_k - b)
        if reg_type == 'elasticnet' and alpha2 > 0:
            grad += 2*alpha2 * y_k
        grad_call_times.append(time.perf_counter() - t0)

        # 2) backtracking if desired
        if backtracking:
            bt_steps = 0
            ls_t0    = time.perf_counter()
            tau_k    = tau
            while True:
                v_tmp = y_k - tau_k * grad
                x_tmp = prox_l1(v_tmp, tau_k * alpha1) if alpha1>0 else v_tmp
                diff  = x_tmp - y_k
                if g_smooth(x_tmp) <= g_smooth(y_k) + grad.dot(diff) + np.linalg.norm(diff)**2/(2*tau_k):
                    break
                tau_k *= eta
                bt_steps += 1
            ls_call_times.append(time.perf_counter() - ls_t0)
            ls_call_iters.append(bt_steps)
            tau = tau_k

        # 3) proximal update
        v      = y_k - tau * grad
        x_next = prox_l1(v, tau * alpha1) if alpha1 > 0 else v

        # 4) record history
        if return_history:
            history["x"].append(x_next.copy())
            obj = compute_objective(x_next, A, b, reg_type, alpha1, alpha2)
            history["obj"].append(obj)

        # 5) compute step & ratio
        this_step = np.linalg.norm(x_next - x_k)
        prev_step = np.linalg.norm(x_k     - x_prev)
        ratio     = this_step / prev_step if prev_step > 0 else np.inf

        # 6) momentum update θ_k = k/(k+1+δ)
        theta = k / (k + 1 + delta)
        y_k   = x_next + theta * (x_next - x_k)

        # 7) prepare next
        x_prev, x_k = x_k, x_next

        # 8) stopping on step norm
        if tol > 0.0 and this_step < tol:
            break

        # 9) stopping on step ratio
        if tol_ratio > 0.0 and ratio < tol_ratio:
            break

    return (x_k, history) if return_history else x_k