# solvers/lbfgs.py

import numpy as np
from scipy.optimize import minimize

def lbfgs_solver(A, b, lam, model="ridge"):
    """
    Solves a smooth regularized least squares problem using L-BFGS.
    
    Assumptions:
    - Only supports smooth regularizers (ℓ2) — no ℓ1 or nonsmooth terms.
    - For 'ridge', this solves the correct objective:
        0.5 * ||Ax - b||^2 + λ * ||x||^2
    - For 'elasticnet', we ignore the ℓ1 part (α * ||x||_1),
      and optimize only the ℓ2 term. This is an approximation and should be clearly
      stated in the technical report.
    - Not suitable for LASSO (ℓ1 only) — do not use L-BFGS in that case.
    """

    m, n = A.shape

    def loss(x):
        res = A @ x - b
        if model == "ridge":
            return 0.5 * np.dot(res, res) + lam * np.dot(x, x)
        elif model == "elasticnet":
            return 0.5 * np.dot(res, res) + lam * 0.5 * np.dot(x, x)  # ℓ1 ignored
        else:
            raise ValueError("L-BFGS only supports 'ridge' or approximate 'elasticnet'")

    def grad(x):
        res = A @ x - b
        if model == "ridge":
            return A.T @ res + 2 * lam * x
        elif model == "elasticnet":
            return A.T @ res + lam * x

    x0 = np.zeros(n)
    result = minimize(loss, x0, method="L-BFGS-B", jac=grad, options={"maxiter": 1000})

    return {
        "x": result.x,
        "history": [],  # No tracking of per-iteration values
        "converged": result.success
    }
