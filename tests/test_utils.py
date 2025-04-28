import time
import pytest
import numpy as np
from utils.timer import Timer
from utils.metrics import mse, mae, r2

class TestTimer:
    def test_timer_accuracy(self):
        """Test that Timer correctly measures a 1-second sleep within ±0.05s."""
        with Timer() as timer:
            time.sleep(1)
        
        assert timer.elapsed is not None
        assert 0.95 <= timer.elapsed <= 1.05  # ±50ms tolerance

class TestMetrics:
    def test_mse(self):
        """Test MSE calculation with simple known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        
        expected = 0.01  # ((0.1**2 + 0.1**2 + 0.1**2) / 3)
        result = mse(y_true, y_pred)
        
        assert np.isclose(result, expected)

    def test_mae(self):
        """Test MAE calculation with simple known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        
        expected = 0.1  # (0.1 + 0.1 + 0.1) / 3
        result = mae(y_true, y_pred)
        
        assert np.isclose(result, expected)

    def test_r2(self):
        """Test R2 calculation with simple known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])  # Perfect prediction
        
        # Perfect prediction should give R2 = 1
        assert np.isclose(r2(y_true, y_pred), 1.0)
        
        # Imperfect prediction test
        y_pred = np.array([1.1, 1.9, 3.1])
        expected = 0.97  # Approximate expected value
        result = r2(y_true, y_pred)
        
        assert np.isclose(result, expected, rtol=0.01)

    def test_metrics_input_validation(self):
        """Test that metrics validate input shapes and emptiness."""
        with pytest.raises(ValueError, match="same shape"):
            mse([1, 2], [1, 2, 3])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mse([], [])