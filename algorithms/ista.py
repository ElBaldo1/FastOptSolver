"""
ISTA – Iterative Shrinkage-Thresholding Algorithm
=================================================

Performs the update:

    w  ←  soft_threshold( w − η ∇L(w),  η · α )

where α is the L1 regularisation strength stored in `loss_obj.alpha`.
"""

from __future__ import annotations

import numpy as np

from .base_solver import BaseSolver, _check_gradient
from losses.base_loss import BaseLoss

import time


def _soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft-thresholding operator."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

"""
ISTA solver (proximal gradient)

Convergence rate: O(1/k)  
Time complexity: O(n·d)  
Memory complexity: O(d)  
Includes profiling of gradient time and step time
"""
class ISTA(BaseSolver):
    """Concrete implementation of ISTA (single-step logic in `_step`)."""

    def __init__(self, loss_obj: BaseLoss, step_size=1e-2, max_iter=1000, tol=1e-4):
        super().__init__(loss_obj, step_size, max_iter, tol)

    # ------------------------------------------------------------------ core
    def _step(self, X: np.ndarray, y: np.ndarray, iteration: int) -> float:
        """
        One ISTA iteration: gradient descent step + soft thresholding.

        Returns
        -------
        float
            The loss value **after** updating `self.w_`.
        """
        # Compute gradient of the smooth part
        t_grad0 = time.perf_counter()
        grad = self.loss_obj.gradient(X, y, self.w_)
        t_grad1 = time.perf_counter()
        _check_gradient(grad)

        grad_norm = np.linalg.norm(grad)
        if "grad_norms" not in self.profile_:
            self.profile_["grad_norms"] = []
        self.profile_["grad_norms"].append(grad_norm)

        # Gradient step
        w_temp = self.w_ - self.step_size * grad

        # L1 shrinkage
        if not hasattr(self.loss_obj, "alpha"):
            raise AttributeError("loss_obj must expose an `alpha` attribute.")
        lam = self.loss_obj.alpha * self.step_size
        self.w_ = _soft_threshold(w_temp, lam)

        if not hasattr(self.profile_, "grad_time"):
            self.profile_["grad_time"] = []
        self.profile_["grad_time"].append(t_grad1 - t_grad0)

        # Return current loss for logging
        return self.loss_obj.compute(X, y, self.w_)
