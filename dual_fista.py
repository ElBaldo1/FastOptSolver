import numpy as np
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

    history = {"lam": [], "x_hat": [], "obj_primal": []} if return_history else None
    P_last: float | None = None

    for k in range(max_iter):
        # (1) primal reconstruction
        rhs   = A.T @ (b + nu_k)
        x_hat = np.linalg.solve(Q, rhs)

        # (2) primal objective
        r      = A @ x_hat - b
        P_next = 0.5 * r.dot(r) \
               + 0.5 * alpha2 * x_hat.dot(x_hat) \
               + alpha1 * np.linalg.norm(x_hat, 1)

        # (3) relative‐gap stopping
        if tol > 0 and P_last is not None:
            rel = abs(P_next - P_last) / max(1.0, abs(P_last))
            if rel < tol:
                break
        P_last = P_next

        # (4) record history
        if return_history:
            history["lam"].append(lam_k.copy())
            history["x_hat"].append(x_hat.copy())
            history["obj_primal"].append(P_next)

        # (5) dual proximal step
        grad_dual = A @ x_hat
        v         = nu_k - tau * grad_dual
        lam_next  = np.clip(v, -alpha1, alpha1)

        # (6) Nesterov inertial update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
        beta   = (t_k - 1) / t_next
        nu_next = lam_next + beta * (lam_next - lam_k)

        # (7) prepare next iteration
        lam_k, nu_k, t_k = lam_next, nu_next, t_next

    return (lam_k, history) if return_history else lam_k

