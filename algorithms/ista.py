"""
ISTA – Iterative Shrinkage-Thresholding Algorithm
=================================================

Performs the update:

    w  ←  soft_threshold( w − η ∇L(w),  η · α )

where α is the L1 regularisation strength stored in `loss_obj.alpha`.
"""

from __future__ import annotations

import numpy as np

from .base_solver import BaseSolver
from losses.base_loss import BaseLoss


def _soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft-thresholding operator."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


class ISTA(BaseSolver):
    """Concrete implementation of ISTA (single-step logic in `_step`)."""

    def __init__(
        self,
        loss_obj: BaseLoss,
        step_size: float = 1e-2,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(loss_obj, step_size, max_iter, tol)

    # ------------------------------------------------------------------ core
    def _step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        One ISTA iteration: gradient descent step + soft thresholding.

        Returns
        -------
        float
            The loss value **after** updating `self.w_`.
        """
        # Compute gradient of the smooth part
        grad = self.loss_obj.gradient(X, y, self.w_)

        # Gradient step
        w_temp = self.w_ - self.step_size * grad

        # L1 shrinkage
        if not hasattr(self.loss_obj, "alpha"):
            raise AttributeError("loss_obj must expose an `alpha` attribute.")
        lam = self.loss_obj.alpha * self.step_size
        self.w_ = _soft_threshold(w_temp, lam)

        # Return current loss for logging
        return self.loss_obj.compute(X, y, self.w_)
