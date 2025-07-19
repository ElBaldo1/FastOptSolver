import numpy as np

# ---------------------------------------------------------------------
# Data generation: 5 features, more samples, higher noise
# ---------------------------------------------------------------------

def generate_correlated_boston_like_data(
    m: int = 2000,
    seed: int = 42,
    noise_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 5-feature data with two correlated blocks:
      - rooms & crime correlated (ρ=0.8)
      - tax & age   correlated (ρ=0.9)
      - distance independent
    Larger sample size and higher noise make convergence differences clearer.
    Returns:
      A (m×5), b (length m), x_true (length 5).
    """
    rng = np.random.default_rng(seed)

    # Block1: rooms & crime
    cov1   = np.array([[1.0, 0.8], [0.8, 1.0]]) * 0.25
    block1 = rng.multivariate_normal([6, 0.2], cov1, size=m)

    # Block2: tax & age
    cov2   = np.array([[1.0, 0.9], [0.9, 1.0]]) * 100
    block2 = rng.multivariate_normal([300, 60], cov2, size=m)

    # Independent feature: distance
    distance = rng.normal(4, 1.0, size=(m,1))

    # Assemble design matrix
    A = np.hstack([block1, block2, distance])

    # True sparse coefficients for 5 features
    x_true = np.array([5.0, 0.0, -0.02, -0.05, 1.5])

    # Generate response with higher noise
    noise = rng.normal(0, noise_std, size=m)
    b     = A @ x_true + noise
    return A, b, x_true
