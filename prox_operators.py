import numpy as np

def prox_l1(v, tau):
    """Soft-thresholding for L1."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def prox_elastic_net(v, tau, alpha1, alpha2):
    """Elastic-Net prox: shrink then divide."""
    s = prox_l1(v, tau * alpha1)
    return s / (1 + 2 * tau * alpha2)

def select_prox_operator(v, tau, reg_type, alpha1, alpha2):
    if reg_type == "lasso":
        return prox_l1(v, tau * alpha1)
    elif reg_type == "elasticnet":
        return prox_elastic_net(v, tau, alpha1, alpha2)
    else:
        raise ValueError("Unsupported reg_type")