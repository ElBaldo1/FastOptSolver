import numpy as np

def prox_l1(v, tau):
    """
    Proximal operator for L1 (soft-thresholding):
      shrink(y;τ) = sign(y) · max(|y| - τ, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def prox_elastic_net(v, tau, alpha1, alpha2):
    """
    Proximal operator for Elastic-Net:
      prox = soft_threshold(v, τ·α1) / (1 + 2τ·α2)
    """
    s = prox_l1(v, tau * alpha1)
    return s / (1 + 2 * tau * alpha2)