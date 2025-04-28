import numpy as np
from typing import Dict, Union
from .base_solver import BaseSolver
from losses.base_loss import BaseLoss

class ISTA(BaseSolver):
    """Iterative Shrinkage-Thresholding Algorithm (ISTA) solver.
    
    Parameters
    ----------
    loss_obj : BaseLoss
        Loss function object implementing compute() and gradient() methods
    step_size : float
        Learning rate/step size for gradient update
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence checking
    """
    
    def __init__(self, loss_obj: BaseLoss, step_size: float = 0.01, 
                 max_iter: int = 1000, tol: float = 1e-4):
        super().__init__(loss_obj, step_size, max_iter, tol)
    
    def _step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform single ISTA optimization step.
        
        1. Compute gradient step
        2. Apply soft-thresholding with parameter loss_obj.alpha * step_size
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        float
            Current loss value
        """
        # Compute gradient
        grad = self.loss_obj.gradient(X, y, self.w_)
        
        # Gradient step
        w_temp = self.w_ - self.step_size * grad
        
        # Soft-thresholding
        alpha = self.loss_obj.alpha * self.step_size
        self.w_ = np.sign(w_temp) * np.maximum(np.abs(w_temp) - alpha, 0)
        
        # Compute and return current loss
        return self.loss_obj.compute(X, y, self.w_)