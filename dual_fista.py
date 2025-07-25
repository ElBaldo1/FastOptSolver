import time
import numpy as np
import prox_operators
import iterative_solvers

# --- METRICS INFRASTRUCTURE ----------------------------------

# Buffers for timing different operations
prox_call_times   = []
grad_call_times   = []
ls_call_times     = []
ls_call_iters     = []
dual_solve_times  = []
dual_clip_times   = []

# Wrap and time the L1 proximal operator
_orig_prox_l1 = prox_operators.prox_l1
def timed_prox_l1(v, tau):
    """Timed L1 proximal operator."""
    start = time.perf_counter()
    out = _orig_prox_l1(v, tau)
    prox_call_times.append(time.perf_counter() - start)
    return out

# Wrap and time the elastic-net proximal operator
_orig_prox_en = prox_operators.prox_elastic_net
def timed_prox_elastic_net(v, tau, a1, a2):
    """Timed Elastic-Net proximal operator."""
    start = time.perf_counter()
    out = _orig_prox_en(v, tau, a1, a2)
    prox_call_times.append(time.perf_counter() - start)
    return out

# Patch both prox_operators and iterative_solvers modules
prox_operators.prox_l1           = timed_prox_l1
prox_operators.prox_elastic_net  = timed_prox_elastic_net
iterative_solvers.prox_l1        = timed_prox_l1
iterative_solvers.prox_elastic_net = timed_prox_elastic_net

# Optional: wrap any gradient function to time it
def wrap_grad(fn):
    """Wrap a gradient function to measure its execution time."""
    def _wrapped(x):
        start = time.perf_counter()
        out = fn(x)
        grad_call_times.append(time.perf_counter() - start)
        return out
    return _wrapped

# Helpers to extract statistics
def get_prox_stats():
    """Return number of prox calls and timing statistics."""
    return {
        'prox_num_calls': len(prox_call_times),
        'prox_time_total': sum(prox_call_times),
        'prox_time_mean':  np.mean(prox_call_times) if prox_call_times else 0.0,
    }

def get_grad_stats():
    """Return number of gradient calls and timing statistics."""
    return {
        'grad_num_calls': len(grad_call_times),
        'grad_time_total': sum(grad_call_times),
        'grad_time_mean':  np.mean(grad_call_times) if grad_call_times else 0.0,
    }

def get_ls_stats():
    """Return line‐search iteration counts and timing statistics."""
    return {
        'ls_num_iters':   sum(ls_call_iters),
        'ls_time_total':  sum(ls_call_times),
        'ls_time_mean':   np.mean(ls_call_times) if ls_call_times else 0.0,
    }

def get_dual_stats():
    """
    Return count and timing statistics for dual‐FISTA subroutines:
      - dual_solve_count
      - dual_solve_time_total, dual_solve_time_mean
      - dual_clip_count
      - dual_clip_time_total, dual_clip_time_mean
    """
    return {
        'dual_solve_count':      len(dual_solve_times),
        'dual_solve_time_total':  sum(dual_solve_times),
        'dual_solve_time_mean':   np.mean(dual_solve_times)  if dual_solve_times  else 0.0,
        'dual_clip_count':       len(dual_clip_times),
        'dual_clip_time_total':   sum(dual_clip_times),
        'dual_clip_time_mean':    np.mean(dual_clip_times)   if dual_clip_times   else 0.0,
    }

# --- DUAL FISTA IMPLEMENTATION WITH TIMING -------------------

def dual_fista(
        A: np.ndarray,
        b: np.ndarray,
        alpha1: float,
        alpha2: float,
        max_iter: int = 500,
        tol: float = 0.0,
        return_history: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    Dual FISTA for Lasso/Elastic-Net (requires alpha1>0):
      1) Reconstruct primal x̂_k = (AᵀA + α₂ I)⁻¹ Aᵀ (b + ν_k)
      2) Evaluate P(x̂_k)
      3) Stop if |P(x̂_k)−P(x̂ₖ₋₁)|/max(1,|P(x̂ₖ₋₁)|) < tol
      4) Dual prox: λₖ₊₁ = clip(ν_k − τ A x̂_k, −α₁, α₁)
      5) Nesterov inertial update on ν
    """
    m, n = A.shape
    Q     = A.T @ A + alpha2 * np.eye(n)
    L     = np.linalg.norm(A, ord=2)**2 + alpha2
    tau   = 1.0 / L

    lam_k = np.zeros(m)
    nu_k  = lam_k.copy()
    t_k   = 1.0

    history = {"lam": [], "obj_primal": []} if return_history else None
    P_last: float | None = None

    for k in range(max_iter):
        # (1) timed solve of the linear system for primal reconstruction
        t0 = time.perf_counter()
        rhs = A.T @ (b + nu_k)
        x_hat = np.linalg.solve(Q, rhs)
        dual_solve_times.append(time.perf_counter() - t0)

        # (2) primal objective evaluation
        r      = A @ x_hat - b
        P_next = 0.5 * r.dot(r) \
               + 0.5 * alpha2 * x_hat.dot(x_hat) \
               + alpha1 * np.linalg.norm(x_hat, 1)

        # (3) relative‐gap stopping criterion
        if tol > 0 and P_last is not None:
            rel = abs(P_next - P_last) / max(1.0, abs(P_last))
            if rel < tol:
                break
        P_last = P_next

        # (4) record history if requested
        if return_history:
            history["lam"].append(lam_k.copy())
            history["obj_primal"].append(P_next)

        # (5) timed dual proximal clipping step
        t1 = time.perf_counter()
        v        = nu_k - tau * (A @ x_hat)
        lam_next = np.clip(v, -alpha1, alpha1)
        dual_clip_times.append(time.perf_counter() - t1)

        # (6) Nesterov inertial update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
        beta   = (t_k - 1) / t_next
        nu_next = lam_next + beta * (lam_next - lam_k)

        # (7) prepare for next iteration
        lam_k, nu_k, t_k = lam_next, nu_next, t_next

    return (lam_k, history) if return_history else lam_k