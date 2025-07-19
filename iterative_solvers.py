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

# --- FISTA with optional adaptive restart, stopping & history support ---
def fista(A, b, reg_type, alpha1, alpha2,
          max_iter=500, tol=0.0,
          adaptive_restart=False, restart_threshold=1.0,
          return_history=False):
    """
    FISTA for three cases:
      - lasso:       g(x)=½||Ax-b||²,                   h(x)=alpha1||x||₁
      - ridge:       g(x)=½||Ax-b||² + ½ alpha2||x||²,   h(x)=0
      - elasticnet:  g(x)=½||Ax-b||² + ½ alpha2||x||²,   h(x)=alpha1||x||₁
 
    reg_type: one of "lasso", "ridge", "elasticnet"
    alpha1, alpha2: non-negative regularization weights
    """
    # ————————————————————————————————
    # 1) Scegli i coefficienti effettivi
    if reg_type == "lasso":
        α1, α2 = alpha1,    0.0
    elif reg_type == "ridge":
        α1, α2 = 0.0,       alpha2
    elif reg_type == "elasticnet":
        α1, α2 = alpha1,    alpha2
    else:
        raise ValueError(f"Unsupported reg_type='{reg_type}'")
 
    # ————————————————————————————————
    # 2) Inizializzazione
    n       = A.shape[1]
    x_k     = np.zeros(n)
    y_k     = x_k.copy()
    t_tilde = 1.0
 
    # Lipschitz of ∇g = AᵀA + α2 I
    L_A     = np.linalg.norm(A, ord=2)**2
    L       = L_A + α2
    τ       = 1.0 / L
 
    objs = []
    if return_history:
        step_norms = []
        ratio_vals = []
        x_hist     = [x_k.copy()]
 
    x_prev = x_k.copy()
 
    # ————————————————————————————————
    # 3) Ciclo FISTA
    for k in range(max_iter):
        # 3.1) grad step su g: ∇g(y_k) = Aᵀ(Ay_k - b) + α2 y_k
        grad = A.T @ (A @ y_k - b) + α2 * y_k
        v    = y_k - τ * grad
 
        # 3.2) prox‐step su h = α1||·||₁
        x_next = prox_l1(v, τ * α1)
 
        # 3.3) misura progresso
        delta = np.linalg.norm(x_next - x_k)
        if return_history:
            prev_delta = np.linalg.norm(x_k - x_prev) if k>0 else 0.0
            ratio      = delta / prev_delta if prev_delta>0 else 0.0
 
        # 3.4) adaptive‐restart o update del fattore t̃
        if adaptive_restart and return_history and ratio > restart_threshold:
            t_tilde_new = 1.0
            y_k         = x_next.copy()
        else:
            t_tilde_new = 0.5 * (1 + np.sqrt(1 + 4 * t_tilde**2))
            beta        = (t_tilde - 1) / t_tilde_new
            y_k         = x_next + beta * (x_next - x_k)
 
        # 3.5) aggiorna stato
        x_prev, x_k, t_tilde = x_k.copy(), x_next, t_tilde_new
 
        # 3.6) calcola f(x_k) = g + h
        r = A @ x_k - b
        g = 0.5 * r.dot(r) + 0.5 * α2 * x_k.dot(x_k)
        h = α1 * np.linalg.norm(x_k, 1)
        objs.append(g + h)
 
        if return_history:
            step_norms.append(delta)
            ratio_vals.append(ratio)
            x_hist.append(x_k.copy())
 
        # 3.7) stopping
        if tol > 0.0 and delta < tol:
            break
 
    # ————————————————————————————————
    # 4) return
    if return_history:
        return x_k, objs, step_norms, ratio_vals, x_hist
    return x_k, objs

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