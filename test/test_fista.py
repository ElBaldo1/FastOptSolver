import numpy as np
from numpy.testing import assert_array_less
from algorithms.ista import ISTA
from algorithms.fista import FISTA
from losses.lasso import Lasso

def test_fista_vs_ista():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + 0.1 * np.random.randn(n_samples)

    # Set up Lasso problem with small alpha for comparison
    alpha = 0.01
    lasso_loss = Lasso(alpha=alpha)

    # Run ISTA
    ista = ISTA(loss_obj=lasso_loss, step_size=0.01, max_iter=50, tol=1e-6)
    ista.fit(X, y)
    ista_final_loss = ista.loss_obj.compute(X, y, ista.w_)

    # Run FISTA
    fista = FISTA(loss_obj=lasso_loss, step_size=0.01, max_iter=50, tol=1e-6)
    fista.fit(X, y)
    fista_final_loss = fista.loss_obj.compute(X, y, fista.w_)

    # Assert FISTA performs at least as well as ISTA
    assert_array_less(fista_final_loss, ista_final_loss + 1e-6)