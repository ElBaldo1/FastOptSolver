import numpy as np

def generate_correlated_boston_like_data(m=800, seed=42, noise_std=0.1):
    """
    Generate 5-feature data with two correlated blocks:
      - rooms & crime correlated (ρ=0.8)
      - tax & age correlated (ρ=0.9)
    Returns A, b, x_true.
    """
    rng = np.random.default_rng(seed)
    # Block1: rooms & crime
    cov1 = np.array([[1.0, 0.8], [0.8, 1.0]]) * 0.25
    block1 = rng.multivariate_normal([6, 0.2], cov1, size=m)
    # Block2: tax & age
    cov2 = np.array([[1.0, 0.9], [0.9, 1.0]]) * 100
    block2 = rng.multivariate_normal([300, 60], cov2, size=m)
    # Independent distance feature
    distance = rng.normal(4, 1.0, size=(m, 1))
    # Assemble
    A = np.hstack([block1, distance, block2])
    # True coefficients: sparse in crime
    x_true = np.array([5.0, 0.0, 1.5, -0.02, -0.05])
    # Generate target
    noise = rng.normal(0, noise_std, size=m)
    b = A @ x_true + noise
    return A, b, x_true


def generate_easy_boston_like_data(m=800, seed=42, noise_std=0.1):
    """
    Generate uncorrelated 5-feature data for easy testing.
    """
    rng = np.random.default_rng(seed)
    rooms = rng.normal(6, 0.5, (m,1))
    crime = rng.normal(0.2, 0.1, (m,1))
    distance = rng.normal(4, 1.0, (m,1))
    tax = rng.normal(300, 25, (m,1))
    age = rng.normal(60, 15, (m,1))
    A = np.hstack([rooms, crime, distance, tax, age])
    x_true = np.array([5.0, -10.0, 1.5, -0.02, -0.05])
    noise = rng.normal(0, noise_std, size=m)
    b = A @ x_true + noise
    return A, b, x_true