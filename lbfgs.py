import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from objective_functions import compute_objective
import time
from iterative_solvers import grad_call_times

class LBFGSSolver:
    """L-BFGS for Ridge and smooth Elastic-Net, with tiny-α shortcut."""
    def __init__(self, reg_type, alpha1, alpha2, max_iter=500, tol=1e-6, eps=1e-8):
        # apply tiny-α → 0 logic
        if reg_type == "lasso":
            self.reg_type = "lasso"
            self.alpha1 = alpha1
            self.alpha2 = 0.0
        elif reg_type == "ridge":
            self.reg_type = "ridge"
            self.alpha1 = 0.0
            self.alpha2 = alpha2
        elif reg_type == "elasticnet":
            if alpha1 < eps:
                # effectively pure ridge
                self.reg_type = "ridge"
                self.alpha1 = 0.0
                self.alpha2 = alpha2
            elif alpha2 < eps:
                # effectively pure lasso
                self.reg_type = "lasso"
                self.alpha1 = alpha1
                self.alpha2 = 0.0
            else:
                self.reg_type = "elasticnet"
                self.alpha1 = alpha1
                self.alpha2 = alpha2
        else:
            raise ValueError(f"Unsupported reg_type='{reg_type}'")

        self.max_iter = max_iter
        self.tol      = tol
        self.history_ = []

    def fit(self, A, b):
        def fg(x):
            # 1) time the smooth-part gradient exactly like in ISTA/FISTA
            t0 = time.perf_counter()
            r    = A @ x - b
            loss = 0.5 * r.dot(r)
            grad = A.T @ r
            if self.reg_type in ("ridge", "elasticnet"):
                loss += 0.5 * self.alpha2 * x.dot(x)
                grad += self.alpha2 * x
            # 2) record the gradient timing
            grad_call_times.append(time.perf_counter() - t0)
            return loss, grad

        def callback(xk):
            val = compute_objective(xk, A, b,
                                    self.reg_type,
                                    self.alpha1,
                                    self.alpha2)
            self.history_.append(val)

        x0 = np.zeros(A.shape[1])
        res = fmin_l_bfgs_b(
            func=fg,
            x0=x0,
            maxiter=self.max_iter,
            pgtol=self.tol,
            callback=callback
        )
        self.x_         = res[0]
        self.final_obj_ = res[1]
        return self
