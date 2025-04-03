import numpy as np

def prox_l2(v, lam):
    r"""
    Proximity operator for the L2-squared norm (also known as Tikhonov or ridge term).

    We want to solve the optimization problem:

    .. math::
        \mathrm{prox}_{ \tfrac{\lambda}{2}\,\|\cdot\|^2 }(\mathbf{v})
        \;=\;
        \underset{\mathbf{x} \in \mathbb{R}^n}{\mathrm{argmin}}
        \Bigl\{
            \tfrac{1}{2}\,\|\mathbf{x} - \mathbf{v}\|^2
            \;+\;
            \tfrac{\lambda}{2}\,\|\mathbf{x}\|^2
        \Bigr\},

    where :math:`\|\mathbf{x}\|^2 = \sum_i x_i^2` denotes the squared Euclidean norm.

    By setting the gradient of the objective function to zero and solving for
    :math:`\mathbf{x}`, one obtains the closed-form solution:

    .. math::
        \mathbf{x}^* \;=\; \frac{\mathbf{v}}{\,1 + \lambda\,}.

    Parameters
    ----------
    v : np.ndarray
        Input vector :math:`\mathbf{v}` in :math:`\mathbb{R}^n`.
    lam : float
        Nonnegative regularization parameter ( :math:`\lambda \ge 0` ).

    Returns
    -------
    x : np.ndarray
        The result of applying the proximity operator, i.e. :math:`\mathbf{v} / (1 + \lambda)`.

    Notes
    -----
    1. **Mathematical insight (2 lines):** In many optimization problems, adding
       a term :math:`\frac{\lambda}{2}\|\mathbf{x}\|^2` enforces a preference for
       small norm solutions (often called “Tikhonov” or “ridge” regularization).
       The prox operator shrinks :math:`\mathbf{v}` by the factor :math:`\tfrac{1}{1+\lambda}`.

    2. **Real-world usage (2 lines):** This operator appears in Ridge Regression
       or weight decay in neural networks, controlling large parameter growth.
       Its simplicity (a uniform scaling of :math:`\mathbf{v}`) makes it computationally efficient.
    """

    # Element-wise shrinkage:
    return v / (1.0 + lam)
