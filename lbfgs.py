
# lbfgs.py

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from objective_functions import compute_objective

class LBFGSSolver:
    """
    Limited-memory BFGS solver for smooth losses (Ridge or ElasticNet with alpha1 ≈ 0).
    """
    def __init__(self, reg_type="ridge", alpha1=0.0, alpha2=0.1, max_iter=1000, tol=1e-6):
        if reg_type == "elasticnet" and alpha1 > 1e-6:
            raise ValueError("L-BFGS only supports smooth problems: alpha1 must be near 0.")
        if reg_type not in ("ridge", "elasticnet"):
            raise ValueError("L-BFGS only supports 'ridge' or 'elasticnet' (with alpha1 ≈ 0).")
        self.reg_type = reg_type
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.tol = tol
        self.history_ = []

    def fit(self, A, b):
        n = A.shape[1]
        x0 = np.zeros(n)

        def loss_and_grad(x):
            residual = A @ x - b
            loss = 0.5 * np.dot(residual, residual)
            grad = A.T @ residual

            if self.reg_type == "ridge":
                loss += self.alpha2 * np.dot(x, x)
                grad += 2 * self.alpha2 * x
            elif self.reg_type == "elasticnet":
                loss += self.alpha2 * np.dot(x, x)  # alpha1 ignored
                grad += 2 * self.alpha2 * x

            return loss, grad

        def callback(xk):
            val = compute_objective(xk, A, b, reg_type=self.reg_type,
                                    alpha1=self.alpha1, alpha2=self.alpha2)
            self.history_.append(val)

        result = fmin_l_bfgs_b(func=loss_and_grad,
                               x0=x0,
                               maxiter=self.max_iter,
                               pgtol=self.tol,
                               callback=callback)

        self.x_ = result[0]
        self.final_obj_ = result[1]
        return self
