"""
Lasso loss (L1-regularised squared error):

    f(w) = (1/2n)‖Xw − y‖²₂ + α‖w‖₁
"""

from __future__ import annotations

import numpy as np
from .base_loss import BaseLoss


class LassoLoss(BaseLoss):
    """Lasso / L1 regularised least-squares."""

    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)

    # ------------------------------------------------------------------ API
    def compute(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        n = X.shape[0]
        residual = X @ w - y
        mse_part = 0.5 * np.dot(residual, residual) / n
        l1_part = self.alpha * np.sum(np.abs(w))
        return mse_part + l1_part

    def grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        grad_mse = X.T @ (X @ w - y) / n
        grad_l1 = self.alpha * np.sign(w)
        return grad_mse + grad_l1
