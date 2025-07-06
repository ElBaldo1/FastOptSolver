import numpy as np


# -------- utility -------------------------------------------------
def spectral_norm_squared(A, n_iter=100):
    """Estimates ||A||² using power iteration."""
    u = np.random.randn(A.shape[1])
    for _ in range(n_iter):
        u = A.T @ (A @ u)
        u /= np.linalg.norm(u)
    return np.linalg.norm(A @ u) ** 2


# ------------------------------------------------------------------

def prox_dual_linf_ball(v, alpha1):
    """Projection onto the L-infinity ball {u | ||u||_∞ ≤ α₁}."""
    return np.clip(v, -alpha1, alpha1)


def dual_objective(λ, A, b):
    """Objective function for the 'wrong' dual problem."""
    return 0.5 * np.linalg.norm(A @ λ - b) ** 2


def dual_fista(A, b, alpha1, max_iter=1000, tol=1e-6):
    """
    This function solves the problem:
        min_λ ½‖Aλ - b‖²   s.t.  ‖λ‖_∞ ≤ α₁

    NOTE: This is NOT the dual of the LASSO problem.
    """
    if alpha1 < 1e-6:
        raise ValueError("This problem requires α₁ > 0.")

    # L is the Lipschitz constant of the gradient of ½‖Aλ - b‖²
    L = spectral_norm_squared(A)
    τ = 1.0 / L

    λ = np.zeros(A.shape[1])
    ν = λ.copy()
    t = 1.0

    obj_vals = []

    for k in range(max_iter):
        λ_prev = λ.copy()

        # Gradient of the smooth part F(ν) = ½‖Aν - b‖²
        grad = A.T @ (A @ ν - b)

        # Proximal gradient step
        λ = prox_dual_linf_ball(ν - τ * grad, alpha1)

        obj_vals.append(dual_objective(λ, A, b))

        # Check for convergence
        if np.linalg.norm(λ - λ_prev) < tol:
            break

        # FISTA acceleration step
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        ν = λ + ((t - 1) / t_next) * (λ - λ_prev)
        t = t_next

    return λ, obj_vals, obj_vals  # Return obj_vals for gaps for compatibility