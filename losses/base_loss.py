"""
BaseLoss
========
Abstract interface for optimisation losses.

Required core methods
---------------------
* compute(X, y, w) -> float
* grad(X, y, w) -> np.ndarray

Compatibility aliases
---------------------
* gradient = grad   (some solvers expect .gradient)
* loss      = compute  (older runner expects .loss)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseLoss(ABC):
    # ----------------------------------------------------- API to implement
    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Return objective value f(w)."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Return gradient ∇f(w)."""
        raise NotImplementedError

    # --------------------------------------------------- backward aliases
    def gradient(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> np.ndarray:  # noqa: D401
        """Alias for :meth:`grad`."""
        return self.grad(X, y, w)

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:  # noqa: D401
        """Alias for :meth:`compute` (kept for legacy code)."""
        return self.compute(X, y, w)
