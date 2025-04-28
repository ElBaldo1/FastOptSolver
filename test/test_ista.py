import numpy as np
from algorithms.ista import ISTA

def test_ista_reproducibility():
    # Set seeds for reproducibility
    np.random.seed(42)
    random_seed = 42

    # Generate random data
    n_samples = 100
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    l1_ratio = 0.5

    # Run ISTA twice with same seeds
    ista1 = ISTA(random_state=random_seed)
    ista1.fit(X, y, l1_ratio)
    obj1 = ista1.objective(X, y, l1_ratio)

    ista2 = ISTA(random_state=random_seed)
    ista2.fit(X, y, l1_ratio)
    obj2 = ista2.objective(X, y, l1_ratio)

    # Assert objectives match to 6 decimal places
    assert abs(obj1 - obj2) < 1e-6