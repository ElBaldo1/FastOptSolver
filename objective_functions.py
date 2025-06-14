
import numpy as np

def compute_objective(x, A, b, reg_type="lasso", alpha1=0.1, alpha2=0.1):
    """
    Compute the value of f(x) = g(x) + h(x) for different regularizations:
    - Lasso: h(x) = alpha1 * ||x||_1
    - Ridge: h(x) = alpha2 * ||x||_2^2 (added into g)
    - Elastic Net: h(x) = alpha1 * ||x||_1 + alpha2 * ||x||_2^2

    Parameters
    ----------
    x : np.ndarray
        Current point
    A : np.ndarray
        Design matrix
    b : np.ndarray
        Target vector
    reg_type : str
        One of ["lasso", "ridge", "elasticnet"]
    alpha1 : float
        L1 regularization weight
    alpha2 : float
        L2 regularization weight

    Returns
    -------
    float
        Objective value f(x)
    """
    residual = A @ x - b
    loss = 0.5 * np.dot(residual, residual)

    if reg_type == "lasso":
        reg = alpha1 * np.linalg.norm(x, 1)
    elif reg_type == "ridge":
        loss += alpha2 * np.dot(x, x)
        reg = 0
    elif reg_type == "elasticnet":
        reg = alpha1 * np.linalg.norm(x, 1) + alpha2 * np.dot(x, x)
    else:
        raise ValueError("Unsupported reg_type. Choose 'lasso', 'ridge', or 'elasticnet'.")

    return loss + reg
