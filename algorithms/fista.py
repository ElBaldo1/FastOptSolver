import numpy as np
from typing import Dict, Union
from .base_solver import BaseSolver, _check_gradient
from losses.base_loss import BaseLoss

import time

"""
FISTA solver (accelerated proximal gradient)

Convergence rate: O(1/k^2)  
Time complexity: O(n·d)  
Memory: O(d)  
Tracks gradient and step times per iteration
"""
class FISTA(BaseSolver):
    """Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) solver.
    
    Implementation of the FISTA algorithm from:
    Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm 
    for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1), 183-202.
    
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
    
    def __init__(self, loss_obj: BaseLoss, step_size=0.01, max_iter=1000, tol=1e-4):
        super().__init__(loss_obj, step_size, max_iter, tol)
        self.t = 1.0  # Momentum variable
        self.y = None  # Extrapolation point
    
    def _step(self, X: np.ndarray, y: np.ndarray, iteration: int) -> float:
        """Perform single FISTA optimization step.
        
        1. Compute gradient step at extrapolation point y
        2. Update weights with momentum
        3. Update momentum variable t
        4. Compute new extrapolation point y
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        if self.y is None:
            self.y = np.zeros(X.shape[1])
        
        # Compute gradient at extrapolation point
        t_grad0 = time.perf_counter()
        grad = self.loss_obj.gradient(X, y, self.y)
        t_grad1 = time.perf_counter()
        _check_gradient(grad)

        if not np.isfinite(self.y).all():
            raise ValueError("Extrapolation point y contains NaN or Inf.")
        
        grad_norm = np.linalg.norm(grad)
        if "grad_norms" not in self.profile_:
            self.profile_["grad_norms"] = []
        self.profile_["grad_norms"].append(grad_norm)

        if iteration > 0 and iteration % 10 == 0 and self._verbose:
            print(f"[Iter {iteration}] ||grad|| = {grad_norm:.4e}, ||y|| = {np.linalg.norm(self.y):.4e}")
        
        # Update weights
        w_temp = self.y - self.step_size * grad
        
        # Soft-thresholding
        alpha = self.loss_obj.alpha * self.step_size
        w_new = np.sign(w_temp) * np.maximum(np.abs(w_temp) - alpha, 0)
        
        # Update momentum
        t_new = (1 + np.sqrt(1 + 4 * self.t**2)) / 2
        
        # Update extrapolation point
        self.y = w_new + ((self.t - 1) / t_new) * (w_new - self.w_)
        
        # Update variables for next iteration
        self.w_ = w_new
        self.t = t_new

        self.profile_["grad_time"].append(t_grad1 - t_grad0)
        
        # Compute and return current loss
        return self.loss_obj.compute(X, y, self.w_)