"""Test cases for loss functions."""
import numpy as np
import pytest
from losses.ridge import RidgeLoss

class TestRidgeLoss:
    """Test suite for RidgeLoss."""
    
    def setup_method(self):
        """Setup synthetic test data."""
        np.random.seed(42)
        self.n_samples = 5
        self.n_features = 3
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randn(self.n_samples)
        self.alpha = 0.1
        self.model = RidgeLoss(self.alpha)
        
    def test_compute_returns_scalar(self):
        """Test that compute() returns a scalar value."""
        w = np.random.randn(self.n_features)
        predictions = self.X @ w
        loss = self.model.compute(predictions, self.y)
        assert np.isscalar(loss)
        
    def test_gradient_matches_finite_difference(self):
        """Test that grad() matches finite difference approximation."""
        w = np.random.randn(self.n_features)
        predictions = self.X @ w
        
        # Compute analytical gradient
        grad_analytical = self.model.grad(predictions, self.y)
        
        # Compute finite difference gradient
        eps = 1e-7
        grad_fd = np.zeros_like(predictions)
        
        for i in range(len(predictions)):
            predictions_plus = predictions.copy()
            predictions_plus[i] += eps
            loss_plus = self.model.compute(predictions_plus, self.y)
            
            predictions_minus = predictions.copy()
            predictions_minus[i] -= eps
            loss_minus = self.model.compute(predictions_minus, self.y)
            
            grad_fd[i] = (loss_plus - loss_minus) / (2 * eps)
            
        # Compare gradients
        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-5)