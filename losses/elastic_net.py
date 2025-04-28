"""Elastic Net (L1 + L2 regularized) loss implementation."""

import numpy as np
from .base_loss import BaseLoss


class ElasticNetLoss(BaseLoss):
    """Elastic Net loss combining L1 and L2 regularization.
    
    Loss function: (1/2n)||Xw - y||² + α₁||w||₁ + α₂||w||₂²
    Gradient: (1/n)Xᵀ(Xw - y) + α₁ sign(w) + 2α₂w
    """
    
    def __init__(self, alpha1: float, alpha2: float):
        """Initialize Elastic Net loss with regularization parameters.
        
        Args:
            alpha1: L1 regularization coefficient (must be > 0)
            alpha2: L2 regularization coefficient (must be > 0)
            
        Raises:
            ValueError: If either alpha1 or alpha2 is not positive
        """
        if alpha1 <= 0 or alpha2 <= 0:
            raise ValueError("alpha1 and alpha2 must be positive")
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Elastic Net loss value.
        
        Args:
            predictions: Model predictions (n_samples,)
            targets: Ground truth values (n_samples,)
            
        Returns:
            Computed loss value: (1/2n)||predictions - targets||² + α₁||w||₁ + α₂||w||₂²
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        n = len(predictions)
        residual = predictions - targets
        mse = 0.5 * np.sum(residual ** 2) / n
        l1_penalty = self.alpha1 * np.sum(np.abs(predictions))
        l2_penalty = self.alpha2 * np.sum(predictions ** 2)
        return mse + l1_penalty + l2_penalty
        
    def grad(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of Elastic Net loss w.r.t. predictions.
        
        Args:
            predictions: Model predictions (n_samples,)
            targets: Ground truth values (n_samples,)
            
        Returns:
            Gradient array: (1/n)(predictions - targets) + α₁ sign(w) + 2α₂w
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        n = len(predictions)
        grad_mse = (predictions - targets) / n
        grad_l1 = self.alpha1 * np.sign(predictions)
        grad_l2 = 2 * self.alpha2 * predictions
        return grad_mse + grad_l1 + grad_l2