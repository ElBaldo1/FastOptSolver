import numpy as np

def prox_l1(v, lam):
    r"""
    Proximity operator for the L1 norm (soft-thresholding operator).

    It solves the problem:
    \[
        \mathrm{prox}_{\lambda \|\cdot\|_1}(v)
        \;=\;
        \arg\min_{x \in \mathbb{R}^n}
        \left\{
            \tfrac{1}{2}\,\|x - v\|_2^2 \;+\; \lambda\,\|x\|_1
        \right\}.
    \]

    In closed form, the solution is obtained component-wise:
    \[
        x_i \;=\; \mathrm{sign}(v_i)\,\max\bigl(\,\lvert v_i\rvert - \lambda,\; 0\bigr).
    \]

    Parameters
    ----------
    v : np.ndarray
        Input vector (or array) of size n.
    lam : float
        Regularization parameter (λ ≥ 0).

    Returns
    -------
    x : np.ndarray
        The result of applying soft-thresholding element by element on `v`.

    Notes
    -----
    - If `lam == 0`, the operator coincides with the identity (`x = v`).
    - For `lam > 0`, any component `v_i` with `|v_i| ≤ lam` gets shrunk to zero.
    - This proximity operator is widely used in solving sparse optimization
      problems such as Lasso, where the L1 norm promotes sparsity.
    """

    # Element-wise computation: x_i = sign(v_i) * max(|v_i| - lam, 0)
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)
