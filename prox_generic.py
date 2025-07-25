import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from objective_functions import compute_reg_and_grad

def prox_via_lbfgs(
    v: np.ndarray,
    tau: float,
    reg_type: str,
    alpha1: float,
    alpha2: float,
    max_iter: int = 50,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Approximate the proximal mapping of g at v:
        prox_{tau·g}(v) = argmin_z { g(z) + (1/(2*tau)) * ||z - v||^2 }
    using a limited-memory quasi-Newton solver (L-BFGS).

    Parameters
    ----------
    v : np.ndarray
        The input point to apply the prox to.
    tau : float
        The step-size parameter.
    reg_type : str
        One of "lasso", "ridge", "elasticnet", etc.
    alpha1 : float
        The ℓ₁ regularization weight.
    alpha2 : float
        The ℓ₂ regularization weight.
    max_iter : int
        Maximum iterations for the inner L-BFGS solver.
    tol : float
        Tolerance for the inner solver (pgtol).

    Returns
    -------
    x_star : np.ndarray
        The approximate prox output.
    """
    def loss_and_grad(z: np.ndarray):
        # quadratic term: (1/(2*tau)) * ||z - v||^2
        diff = z - v
        loss_q = 0.5 * (diff @ diff) / tau
        grad_q = diff / tau

        # regularizer term and its gradient
        reg_loss, reg_grad = compute_reg_and_grad(z, reg_type, alpha1, alpha2)

        # total loss and gradient
        return loss_q + reg_loss, grad_q + reg_grad

    # warm start from v
    x0 = v.copy()
    x_star, _, _ = fmin_l_bfgs_b(
        func=loss_and_grad,
        x0=x0,
        maxiter=max_iter,
        pgtol=tol
    )
    return x_star


def prox_is_closed_form(
    reg_type: str,
    alpha1: float,
    alpha2: float
) -> bool:
    """
    Return True if we already have a closed‐form prox for (reg_type,α1,α2),
    e.g. Lasso (α2=0), Ridge (α1=0), or pure Elastic‐Net with both nonzero
    but separable (we still do soft‐threshold + shrink).
    """
    # L1 only (soft‐threshold)
    if reg_type == "lasso" and alpha2 == 0.0:
        return True
    # L2 only (shrinkage)
    if reg_type == "ridge" and alpha1 == 0.0:
        return True
    # Elastic‐Net: prox factorizes into soft‐threshold + shrink
    if reg_type == "elasticnet" and alpha1 > 0.0 and alpha2 > 0.0:
        return True
    return False
