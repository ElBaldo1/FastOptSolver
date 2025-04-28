"""Lasso (L1 regularized) loss implementation."""

import numpy as np
from .base_loss import BaseLoss


class LassoLoss(BaseLoss):
    """Lasso loss with L1 regularization.
    
    Loss function: (1/2n)||Xw - y||² + α||w||₁
    Gradient: (1/n)Xᵀ(Xw - y) + α sign(w)
    """
    
    def __init__(self, alpha: float):
        """Initialize Lasso loss with regularization parameter.
        
        Args:
            alpha: L1 regularization coefficient (must be > 0)
            
        Raises:
            ValueError: If alpha is not positive
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        
    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Lasso loss value.
        
        Args:
            predictions: Model predictions (n_samples,)
            targets: Ground truth values (n_samples,)
            
        Returns:
            Computed loss value: (1/2n)||predictions - targets||² + α||w||₁
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        n = len(predictions)
        residual = predictions - targets
        mse = 0.5 * np.sum(residual ** 2) / n
        l1_penalty = self.alpha * np.sum(np.abs(predictions))
        return mse + l1_penalty
        
    def grad(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of Lasso loss w.r.t. predictions.
        
        Args:
            predictions: Model predictions (n_samples,)
            targets: Ground truth values (n_samples,)
            
        Returns:
            Gradient array: (1/n)(predictions - targets) + α sign(predictions)
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        n = len(predictions)
        grad_mse = (predictions - targets) / n
        grad_l1 = self.alpha * np.sign(predictions)
        return grad_mse + grad_l1