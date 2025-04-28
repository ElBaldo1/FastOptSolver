import numpy as np
import pytest
from data_loader import load_housing

def test_load_housing_shape_consistency():
    """Test that returned arrays have consistent shapes."""
    X_train, X_test, y_train, y_test = load_housing(random_state=42)
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions should match"
    assert len(y_train.shape) == 1, "y_train should be 1D"
    assert len(y_test.shape) == 1, "y_test should be 1D"

def test_load_housing_reproducibility():
    """Test that same random_state produces identical splits."""
    X_train1, X_test1, y_train1, y_test1 = load_housing(random_state=42)
    X_train2, X_test2, y_train2, y_test2 = load_housing(random_state=42)
    
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(X_test1, X_test2)
    np.testing.assert_array_equal(y_train1, y_train2)
    np.testing.assert_array_equal(y_test1, y_test2)

def test_load_housing_without_normalization():
    """Test that normalization parameter works correctly."""
    X_train, _, _, _ = load_housing(normalize=False, random_state=42)
    assert not np.allclose(X_train.mean(axis=0), 0), "Data should not be normalized"