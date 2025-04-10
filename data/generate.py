import numpy as np


def generate_synthetic_lasso_data(m=100, n=200, sparsity=10, noise_std=0.01, seed=42):
    """
    Generates a synthetic LASSO problem: A, b, x_true

    Parameters
    ----------
    m : int
        Number of samples (rows of A)
    n : int
        Number of features (columns of A)
    sparsity : int
        Number of non-zero entries in x_true
    noise_std : float
        Standard deviation of Gaussian noise added to b
    seed : int
        Random seed

    Returns
    -------
    A : np.ndarray of shape (m, n)
    b : np.ndarray of shape (m,)
    x_true : np.ndarray of shape (n,)
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(m, n))

    x_true = np.zeros(n)
    nonzero_idx = rng.choice(n, size=sparsity, replace=False)
    x_true[nonzero_idx] = rng.standard_normal(size=sparsity)

    noise = rng.normal(0, noise_std, size=m)
    b = A @ x_true + noise

    return A, b, x_true
