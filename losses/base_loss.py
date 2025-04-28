"""Base loss module defining the abstract interface for loss functions.

All concrete loss implementations should inherit from BaseLoss and implement:
- compute(): Computes the loss value given predictions and targets
- grad(): Computes the gradient of the loss w.r.t. predictions

Expected shapes:
- predictions: (n_samples, n_outputs) array-like
- targets: (n_samples, n_outputs) array-like
- Return values:
  - compute(): scalar
  - grad(): same shape as predictions
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseLoss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the loss value.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Computed loss value (scalar)
        """
        pass
    
    @abstractmethod
    def grad(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Gradient array with same shape as predictions
        """
        pass