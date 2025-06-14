
import numpy as np

def prox_l1(v, tau):
    """
    Proximal operator for the L1 norm (Lasso).
    Soft-thresholding: prox_{tau * ||.||_1}(v)
    """
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def prox_elastic_net(v, tau, alpha1, alpha2):
    """
    Proximal operator for Elastic Net:
    h(x) = alpha1 * ||x||_1 + alpha2 * ||x||_2^2

    Solution:
    prox = SoftThreshold(v, tau * alpha1) / (1 + 2 * tau * alpha2)
    """
    shrink = prox_l1(v, tau * alpha1)
    return shrink / (1 + 2 * tau * alpha2)


def select_prox_operator(v, tau, reg_type="lasso", alpha1=0.1, alpha2=0.1):
    """
    Unified interface to call the right prox operator depending on regularization type.

    Parameters
    ----------
    v : np.ndarray
        Input vector (typically x - tau * grad)
    tau : float
        Step size
    reg_type : str
        "lasso" or "elasticnet"
    alpha1 : float
        L1 regularization weight
    alpha2 : float
        L2 regularization weight

    Returns
    -------
    np.ndarray
        Result of the prox operator
    """
    if reg_type == "lasso":
        return prox_l1(v, tau * alpha1)
    elif reg_type == "elasticnet":
        return prox_elastic_net(v, tau, alpha1, alpha2)
    else:
        raise ValueError("Unsupported reg_type: choose 'lasso' or 'elasticnet'")
