import numpy as np
from .prox_l1 import prox_l1

def prox_elasticnet(v, lam1, lam2):
    r"""
    Proximity operator for the **Elastic Net** regularization term:
    \[
        \lambda_1 \|x\|_1 + \frac{\lambda_2}{2} \|x\|_2^2.
    \]

    This function computes the solution to:
    \[
        \mathrm{prox}_{\lambda_1 \|\cdot\|_1 + \frac{\lambda_2}{2} \|\cdot\|_2^2}(v)
        = \arg\min_{x \in \mathbb{R}^n}
        \left\{
            \frac{1}{2} \|x - v\|_2^2
            + \lambda_1 \|x\|_1
            + \frac{\lambda_2}{2} \|x\|_2^2
        \right\}.
    \]

    Using the identity:
    \[
        \mathrm{prox}_{\lambda_1 \|\cdot\|_1 + \frac{\lambda_2}{2} \|\cdot\|_2^2}(v)
        = \mathrm{prox}_{\frac{\lambda_1}{1+\lambda_2} \|\cdot\|_1}
        \left( \frac{v}{1 + \lambda_2} \right),
    \]
    we apply a soft-thresholding operator to a scaled input.

    Parameters
    ----------
    v : np.ndarray
        Input vector :math:`v \in \mathbb{R}^n`.
    lam1 : float
        Regularization coefficient for the L1 norm (:math:`\lambda_1 \geq 0`).
    lam2 : float
        Regularization coefficient for the L2 norm (:math:`\lambda_2 \geq 0`).

    Returns
    -------
    x : np.ndarray
        Proximal point :math:`x^* \in \mathbb{R}^n` resulting from the Elastic Net operator.

    Notes
    -----
    - This operator is widely used in sparse linear regression where the L1 term promotes sparsity
      and the L2 term adds stability (as in Ridge regression).
    - The closed-form relies on the fact that the sum of two proximable functions can be reduced
      to composition in special cases like Elastic Net.

    Real-world use (2 lines)
    ------------------------
    In statistical learning, Elastic Net is preferred over Lasso when features are highly correlated.
    Its proximity operator enables efficient updates in coordinate descent or proximal gradient methods.
    """
    scaled_v = v / (1.0 + lam2)
    scaled_lam1 = lam1 / (1.0 + lam2)
    return prox_l1(scaled_v, scaled_lam1)
