import numpy as np
from typing import Dict, Union
from losses.base_loss import BaseLoss

class BaseSolver:
    """Base class for iterative optimization solvers.
    
    Parameters
    ----------
    loss_obj : BaseLoss
        Loss function object implementing compute() and gradient() methods
    step_size : float
        Learning rate/step size for gradient updates
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence checking
        
    Attributes
    ----------
    w_ : np.ndarray
        Learned weight vector (n_features,)
    history_ : Dict[str, list]
        Dictionary tracking loss values and other metrics during training
    """
    
    def __init__(self, loss_obj: BaseLoss, step_size: float = 0.01, 
                 max_iter: int = 1000, tol: float = 1e-4):
        self.loss_obj = loss_obj
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.history_ = {'loss': []}
        
    def _init_params(self, n_features: int) -> None:
        """Initialize model parameters (weights).
        
        Parameters
        ----------
        n_features : int
            Number of features in input data
        """
        self.w_ = np.zeros(n_features)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseSolver':
        """Fit model to training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        self : BaseSolver
            Returns the instance itself
        """
        # Input validation
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have same number of samples")
            
        n_samples, n_features = X.shape
        self._init_params(n_features)
        
        for _ in range(self.max_iter):
            loss = self._step(X, y)
            self.history_['loss'].append(loss)
            
            # Check convergence
            if len(self.history_['loss']) > 1:
                if abs(self.history_['loss'][-2] - loss) < self.tol:
                    break
                    
        return self
        
    def _step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform single optimization step.
        
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
        raise NotImplementedError("_step() must be implemented in child classes")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using learned weights.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,)
        """
        if not hasattr(self, 'w_'):
            raise RuntimeError("Model not fitted yet")
        if X.shape[1] != len(self.w_):
            raise ValueError("Number of features must match training data")
            
        return X @ self.w_