"""
Elastic Net loss:

    f(w) = (1/2n)‖Xw − y‖²₂ + α1‖w‖₁ + α2‖w‖²₂
"""

from __future__ import annotations

import numpy as np
from .base_loss import BaseLoss


class ElasticNetLoss(BaseLoss):
    """Combination of L1 and L2 penalties."""

    def __init__(self, alpha1: float, alpha2: float) -> None:
        if alpha1 <= 0 or alpha2 <= 0:
            raise ValueError("alpha1 and alpha2 must be positive.")
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.alpha = self.alpha1  # Alias for compatibility with ISTA/FISTA

    # ------------------------------------------------------------------ API
    def compute(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        n = X.shape[0]
        residual = X @ w - y
        mse_part = 0.5 * np.dot(residual, residual) / n
        l1_part = self.alpha1 * np.sum(np.abs(w))
        l2_part = self.alpha2 * np.dot(w, w)
        return mse_part + l1_part + l2_part

    def grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        grad_mse = X.T @ (X @ w - y) / n
        grad_l1 = self.alpha1 * np.sign(w)
        grad_l2 = 2.0 * self.alpha2 * w
        return grad_mse + grad_l1 + grad_l2
