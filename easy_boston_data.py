
import numpy as np

def generate_easy_boston_like_data(m=800, seed=42, noise_std=0.1):
    """
    Generate a simple but realistic regression dataset inspired by Boston Housing.
    Suitable for testing convergence of ISTA, FISTA, L-BFGS, and Dual-FISTA.

    Feature inspiration:
    ---------------------
    We simulate 5 key features based on the original Boston Housing dataset:
    - Rooms: average number of rooms per dwelling
    - CrimeRate: per capita crime by town
    - Distance: to employment centers
    - TaxRate: property-tax rate
    - Age: proportion of owner-occupied units built before 1940

    These are simplified (Gaussian) to avoid complexity and ensure convergence.

    Parameters
    ----------
    m : int
        Number of samples
    seed : int
        Random seed
    noise_std : float
        Standard deviation of Gaussian noise

    Returns
    -------
    A : np.ndarray of shape (m, 5)
        Design matrix (features)
    b : np.ndarray of shape (m,)
        Target variable (house price proxy)
    x_true : np.ndarray of shape (5,)
        Ground-truth coefficients (weights used to generate b)
    """
    rng = np.random.default_rng(seed)

    # Simulated features — inspired by Boston Housing
    rooms = rng.normal(loc=6, scale=0.5, size=(m, 1))        # avg. rooms: typically 4–8
    crime = rng.normal(loc=0.2, scale=0.1, size=(m, 1))       # crime rate: low values
    distance = rng.normal(loc=4, scale=1.0, size=(m, 1))      # distance to city center
    tax = rng.normal(loc=300, scale=25, size=(m, 1))          # property tax rate
    age = rng.normal(loc=60, scale=15, size=(m, 1))           # house age

    # Combine into design matrix
    A = np.hstack([rooms, crime, distance, tax, age])

    # True coefficients — interpretable and structured
    # More rooms increases price, crime reduces it, others have moderate effect
    x_true = np.array([5.0, -10.0, 1.5, -0.02, -0.05])

    # Generate target: linear model + small Gaussian noise
    noise = rng.normal(0, noise_std, size=m)
    b = A @ x_true + noise

    return A, b, x_true
