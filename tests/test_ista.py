import numpy as np
import unittest
from unittest import TestCase
from numpy.testing import assert_array_less
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from losses.lasso import Lasso
from algorithm.ista import ISTA

class TestISTA(TestCase):
    def test_ista_decreasing_objective(self):
        """Test that ISTA objective decreases strictly each step."""
        np.random.seed(42)
        n_samples = 20
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        alpha = 0.1  # regularization parameter
        
        lasso = Lasso(alpha=alpha)
        ista = ISTA(lasso, step_size=0.01, max_iter=30, tol=1e-10)
        
        # Run ISTA for 30 iterations
        prev_obj = np.inf
        for _ in range(30):
            current_obj = ista._step(X, y)
            # Assert objective decreases strictly
            assert_array_less(current_obj, prev_obj)
            prev_obj = current_obj

if __name__ == '__main__':
    unittest.main()