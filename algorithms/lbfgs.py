# algorithms/lbfgs.py

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from typing import Dict, Union
from algorithms.base_solver import BaseSolver
from losses.base_loss import BaseLoss

class LBFGSSolver(BaseSolver):
    """
    Limited-memory BFGS solver using scipy.optimize.fmin_l_bfgs_b.

    Only supports fully differentiable losses (e.g. Ridge, ElasticNet).
    """

    def __init__(self, loss_obj: BaseLoss, max_iter=1000, tol=1e-4):
        # step_size not used by L-BFGS
        super().__init__(loss_obj, step_size=None, max_iter=max_iter, tol=tol)

    def _step(self, X: np.ndarray, y: np.ndarray, iteration: int) -> float:
        """
        No per-step update – L-BFGS is run in full during fit().
        """
        raise RuntimeError("LBFGSSolver does not support step-by-step updates. Use fit().")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LBFGSSolver":
        """
        Runs full L-BFGS optimization using fmin_l_bfgs_b, logging objective values.
        """
        n_features = X.shape[1]
        w0 = np.zeros(n_features)
        self.history_["loss"] = []  # Reset history

        def callback(wk):
            # Save current objective during optimization
            loss_val = self.loss_obj.compute(X, y, wk)
            self.history_["loss"].append(loss_val)

        def obj_fun(w):
            loss = self.loss_obj.compute(X, y, w)
            grad = self.loss_obj.grad(X, y, w)
            return loss, grad

        # Optimize
        w_opt, f_opt, _ = fmin_l_bfgs_b(
            func=obj_fun,
            x0=w0,
            maxiter=self.max_iter,
            pgtol=self.tol,
            callback=callback
        )

        self.w_ = w_opt

        # Ensure final value is logged
        if len(self.history_["loss"]) == 0 or self.history_["loss"][-1] != f_opt:
            self.history_["loss"].append(f_opt)

        if self._verbose:
            print(f"[LBFGS] Completed in {len(self.history_['loss'])} iterations")

        return self