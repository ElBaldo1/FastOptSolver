# algorithms/dual_fista.py

import numpy as np
from algorithms.base_solver import BaseSolver
from losses.base_loss import BaseLoss

class DualFISTA(BaseSolver):
    """
    Dual-FISTA solver for Lasso: solves the dual problem and recovers primal w.
    """

    def __init__(self, loss_obj: BaseLoss, step_size: float, max_iter=100, tol=1e-6):
        super().__init__(loss_obj, step_size, max_iter, tol)
        self.verbose = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DualFISTA":
        n, d = X.shape
        alpha = self.loss_obj.alpha  # assume loss is LassoLoss
        Lipschitz = np.linalg.norm(X, ord=2) ** 2

        u = np.random.randn(n) * 0.01
        z = np.zeros(n)
        t = 1.0

        self.history_["loss"] = []

        for k in range(self.max_iter):
            u_old = u.copy()
            grad = z - y  # gradient of f(z) = 0.5 * ||z - y||^2

            grad_norm = np.linalg.norm(grad)
            if "grad_norms" not in self.profile_:
                self.profile_["grad_norms"] = []
            self.profile_["grad_norms"].append(grad_norm)

            u = z - self.step_size * grad

            u = z - self.step_size * grad

            # Project u onto the feasible set ||X^T u||_∞ ≤ alpha
            XTu = X.T @ u
            norm_XTu = np.max(np.abs(XTu))

            if norm_XTu > alpha:
                scaling = alpha / norm_XTu
                u = u * scaling

            t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = u + ((t - 1) / t_next) * (u - u_old)
            t = t_next

            # Recover primal variable from dual iterate
            w = self._soft_thresholding(X.T @ u, alpha)
            loss_val = self.loss_obj.compute(X, y, w)

            if self.verbose:
                print(f"[DualFISTA] Iter {k:3d} | loss = {loss_val:.6f} | ||u|| = {np.linalg.norm(u):.4f} | max|X^T u| = {np.max(np.abs(X.T @ u)):.4f}")

            self.history_["loss"].append(loss_val)

            if self.verbose and k % 10 == 0:
                print(f"[Iter {k}] loss = {loss_val:.6f}")

            if k > 1 and abs(self.history_["loss"][-1] - self.history_["loss"][-2]) < self.tol:
                break

        self.w_ = w

        if self.verbose:
            print(f"[DualFISTA] Final loss = {self.history_['loss'][-1]:.6f} | iters: {len(self.history_['loss'])}")

        return self
    
    def _step(self, X: np.ndarray, y: np.ndarray, iteration: int) -> float:
        raise NotImplementedError("DualFISTA does not support step-wise updates. Use fit() instead.")

    @staticmethod
    def _soft_thresholding(z: np.ndarray, alpha: float) -> np.ndarray:
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)