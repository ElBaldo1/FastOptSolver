import numpy as np

def generate_correlated_boston_like_data(m=800, seed=42, noise_std=0.1):
    """
    Versione con feature correlate:
      - rooms e crime correlate (rho=0.8)
      - tax e age correlate   (rho=0.9)
    Coefficienti x_true parzialmente sparsi.
    """
    rng = np.random.default_rng(seed)

    # costruiamo un blocco 2x2 con correlazione 0.8
    cov1 = np.array([[1.0, 0.8],
                     [0.8, 1.0]])
    block1 = rng.multivariate_normal(mean=[6, 0.2], cov=cov1*0.25, size=m)

    # blocco 2x2 con correlazione 0.9
    cov2 = np.array([[1.0, 0.9],
                     [0.9, 1.0]])
    block2 = rng.multivariate_normal(mean=[300, 60], cov=cov2*100, size=m)

    # distanza indipendente
    distance = rng.normal(loc=4, scale=1.0, size=(m,1))

    # combiniamo
    rooms, crime = block1[:,0:1], block1[:,1:2]
    tax,   age   = block2[:,0:1], block2[:,1:2]
    A = np.hstack([rooms, crime, distance, tax, age])

    # x_true: rooms≠0, crime=0, distance≠0, tax≠0, age≠0
    x_true = np.array([5.0,  0.0,   1.5,  -0.02,  -0.05])

    # target
    noise = rng.normal(0, noise_std, size=m)
    b = A @ x_true + noise

    return A, b, x_true
