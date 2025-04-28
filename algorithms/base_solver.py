"""
BaseSolver
==========

Abstract base class for iterative optimisation algorithms.  
Key additions:

* **step(X, y)** – one public optimisation step that wraps the protected
  `_step` implemented by concrete solvers.
* **w property** – read-only alias exposing the learned weight vector
  (`self.w_`) so external code can access `solver.w`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from losses.base_loss import BaseLoss


class BaseSolver(ABC):
    """Parent class for ISTA, FISTA, etc."""

    def __init__(
        self,
        loss_obj: BaseLoss,
        step_size: float = 1e-2,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.loss_obj = loss_obj
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

        self.w_: Optional[np.ndarray] = None  # lazy initialisation
        self.history_: Dict[str, List[float]] = {"loss": []}

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    @property
    def w(self) -> np.ndarray:
        """Read-only view of the current parameter vector."""
        if self.w_ is None:
            raise RuntimeError("Weights not initialised: call step() or fit() first.")
        return self.w_

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform **one** optimisation step.

        This method is what external helpers (e.g. `run_experiment`) should
        call in their per-iteration loop.

        Returns
        -------
        float
            Current loss value after the update (convenience).
        """
        # Lazy weight initialisation
        if self.w_ is None:
            self._init_params(X.shape[1])

        loss_val = self._step(X, y)
        self.history_["loss"].append(loss_val)
        return loss_val

    # --------------------------------------------------------------------- #
    # API to be implemented by subclasses
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _step(self, X: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover
        """One algorithm-specific update returning the new loss value."""
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Fit / predict utilities (unchanged)
    # --------------------------------------------------------------------- #
    def _init_params(self, n_features: int) -> None:
        """Initialise parameter vector with zeros."""
        self.w_ = np.zeros(n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSolver":
        """
        Full training loop – included for completeness, but helpers may still
        prefer to call `step` manually for fine-grained control.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self._init_params(X.shape[1])

        for _ in range(self.max_iter):
            loss = self.step(X, y)  # re-use public step
            if len(self.history_["loss"]) > 1:
                if abs(self.history_["loss"][-2] - loss) < self.tol:
                    break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model not fitted yet.")
        if X.shape[1] != self.w_.shape[0]:
            raise ValueError("Feature dimension mismatch.")
        return X @ self.w_
