import numpy as np

def generate_correlated_boston_like_data(
    m: int = 2000,
    seed: int = 42,
    noise_std: float = 2.0,
    rho1: float = 0.8,
    rho2: float = 0.9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    # Block1: rooms & crime correlated
    cov1   = np.array([[1.0, rho1], [rho1, 1.0]]) * 0.25
    block1 = rng.multivariate_normal([6, 0.2], cov1, size=m)

    # Block2: tax & age correlated
    cov2   = np.array([[1.0, rho2], [rho2, 1.0]]) * 100
    block2 = rng.multivariate_normal([300, 60], cov2, size=m)

    # Independent feature: distance
    distance = rng.normal(4, 1.0, size=(m,1))

    # Assemble original design matrix
    A = np.hstack([block1, block2, distance])

    # ── Whitening step ──
    # 1) Compute sample covariance of A (5×5)
    C = np.cov(A, rowvar=False)
    # 2) Eigen-decompose C = V diag(e) Vᵀ
    eigvals, eigvecs = np.linalg.eigh(C)
    # 3) Build whitening matrix W = V diag(1/√e) Vᵀ
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    # 4) Apply to A so that cov(A_new) ≈ I
    A = A @ W

    # True sparse coefficients
    x_true = np.array([5.0, 0.0, -0.02, -0.05, 1.5])

    # Response with noise
    noise = rng.normal(0, noise_std, size=m)
    b     = A @ x_true + noise

    return A, b, x_true
