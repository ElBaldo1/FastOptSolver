import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from objective_functions import compute_objective

class LBFGSSolver:
    """L-BFGS for Ridge and smooth Elastic-Net."""
    def __init__(self, reg_type, alpha1, alpha2, max_iter=500, tol=1e-6):
        self.reg_type = reg_type
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.tol = tol
        self.history_ = []

    def fit(self, A, b):
        def fg(x):
            r = A @ x - b
            loss = 0.5 * r.dot(r)
            grad = A.T @ r
            if self.reg_type in ("ridge", "elasticnet"):
                loss += self.alpha2 * x.dot(x)
                grad += 2 * self.alpha2 * x
            return loss, grad

        def cb(xk, *args):
            val = compute_objective(xk, A, b, self.reg_type, self.alpha1, self.alpha2)
            self.history_.append(val)

        x0 = np.zeros(A.shape[1])
        res = fmin_l_bfgs_b(func=fg, x0=x0, maxiter=self.max_iter, pgtol=self.tol, callback=cb)
        self.x_ = res[0]
        self.final_obj_ = res[1]
        return self