import numpy as np
from prox.prox_l1 import prox_l1

def ista(A, b, lam, step_size, max_iter=1000, tol=1e-6, verbose=False):
    r"""
    ISTA — Iterative Shrinkage-Thresholding Algorithm.

    Solves the convex optimization problem:
    \[
        \min_{x \in \mathbb{R}^n}
        \left\{
            f(x) + g(x) := \frac{1}{2}\|Ax - b\|_2^2 + \lambda \|x\|_1
        \right\}
    \]
    where:
    - :math:`f(x) = \frac{1}{2}\|Ax - b\|^2` is smooth with Lipschitz continuous gradient,
    - :math:`g(x) = \lambda \|x\|_1` is nonsmooth but proximable.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Design (measurement) matrix.
    b : np.ndarray, shape (m,)
        Observation vector.
    lam : float
        Regularization parameter :math:`\lambda \ge 0`.
    step_size : float
        Gradient descent step size, must satisfy :math:`\text{step\_size} \le 1 / \|A^T A\|_2` for convergence.
    max_iter : int
        Maximum number of iterations to perform.
    tol : float
        Convergence threshold based on :math:`\|x^{(k)} - x^{(k-1)}\|_2`.
    verbose : bool
        If True, prints iteration-wise objective value and residual norm.

    Returns
    -------
    x : np.ndarray, shape (n,)
        Estimated minimizer :math:`x^\star`.
    history : dict
        Tracks:
        - 'objective': list of objective values :math:`F(x^{(k)})` at each iteration,
        - 'residual' : list of update residuals :math:`\|x^{(k)} - x^{(k-1)}\|`.

    Notes
    -----
    - ISTA is a proximal gradient method with convergence rate :math:`O(1/k)` in function values.
    - It is commonly used for sparse recovery problems such as Lasso and basis pursuit.

    Real-world usage (2 lines)
    --------------------------
    ISTA is used in compressed sensing and image deblurring when solutions are sparse in a known basis.
    Its simplicity makes it a natural baseline before using accelerated versions like FISTA.
    """
    m, n = A.shape
    x = np.zeros(n)
    history = {'objective': [], 'residual': []}

    for k in range(max_iter):
        # Compute gradient ∇f(x) = Aᵀ(Ax - b)
        grad = A.T @ (A @ x - b)

        # Proximal gradient step for g(x) = λ‖x‖₁
        x_new = prox_l1(x - step_size * grad, lam * step_size)

        # Compute function value and update distance
        obj = 0.5 * np.linalg.norm(A @ x_new - b)**2 + lam * np.linalg.norm(x_new, 1)
        res = np.linalg.norm(x_new - x)

        history['objective'].append(obj)
        history['residual'].append(res)

        if verbose:
            print(f"Iter {k:4d}: Obj = {obj:.6f}, Residual = {res:.2e}")

        if res < tol:
            break

        x = x_new

    return x, history
