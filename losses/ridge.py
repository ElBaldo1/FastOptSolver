"""Ridge (L2 regularized) loss implementation."""

import numpy as np
from .base_loss import BaseLoss


class RidgeLoss(BaseLoss):
    """Ridge loss with L2 regularization.
    
    Loss function: (1/2n)||Xw - y||² + α||w||₂²
    Gradient: (1/n)Xᵀ(Xw - y) + 2αw
    """
    
    def __init__(self, alpha: float):
        """Initialize Ridge loss with regularization parameter.
        
        Args:
            alpha: L2 regularization coefficient (must be > 0)
            
        Raises:
            ValueError: If alpha is not positive
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        
    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Ridge loss value.
        
        Args:
            predictions: Model predictions (n_samples,)
            targets: Ground truth values (n_samples,)
            
        Returns:
            Computed loss value: (1/2n)||predictions - targets||² + α||w||₂²
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        n = len(predictions)
        residual = predictions - targets
        mse = 0.5 * np.sum(residual ** 2) / n
        l2_penalty = self.alpha * np.sum(predictions ** 2)
        return mse + l2_penalty
        
    def grad(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of Ridge loss w.r.t. predictions.
        
        Args:
            predictions: Model predictions (n_samples,)
            targets: Ground truth values (n_samples,)
            
        Returns:
            Gradient array: (1/n)(predictions - targets) + 2αw
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        n = len(predictions)
        grad_mse = (predictions - targets) / n
        grad_l2 = 2 * self.alpha * predictions
        return grad_mse + grad_l2