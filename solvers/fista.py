import numpy as np
from prox.prox_l1 import prox_l1

def fista(A, b, lam, step_size, max_iter=1000, tol=1e-6, verbose=False):
    r"""
    FISTA — Fast Iterative Shrinkage-Thresholding Algorithm.

    Solves the composite convex optimization problem:
    \[
        \min_{x \in \mathbb{R}^n}
        \left\{
            \tfrac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1
        \right\}
    \]
    where:
    - The smooth term :math:`f(x) = \tfrac{1}{2}\|Ax - b\|^2` has Lipschitz gradient,
    - The non-smooth term :math:`g(x) = \lambda \|x\|_1` is proximable.

    Parameters
    ----------
    A : np.ndarray of shape (m, n)
        Design matrix.
    b : np.ndarray of shape (m,)
        Observation vector.
    lam : float
        Regularization parameter :math:`\lambda \ge 0` (controls sparsity).
    step_size : float
        Gradient step size, must satisfy :math:`\text{step\_size} \le 1 / L`,
        where :math:`L = \|A^\top A\|_2` is the Lipschitz constant of ∇f.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance on the norm of the update (used as stopping criterion).
    verbose : bool
        If True, prints convergence information at each iteration.

    Returns
    -------
    x : np.ndarray of shape (n,)
        Final estimate of the minimizer.
    history : dict
        Dictionary with keys:
        - 'objective': list of objective values over iterations.
        - 'residual': list of residual norms :math:`\|x^{(k)} - x^{(k-1)}\|`.

    Notes
    -----
    - FISTA improves over ISTA by incorporating a momentum term (Nesterov acceleration).
    - The algorithm has theoretical convergence rate :math:`O(1/k^2)` for the objective value.

    Real-world usage (2 lines)
    --------------------------
    FISTA is commonly used in signal processing and compressed sensing to recover sparse signals.
    It combines fast convergence with low per-iteration cost, especially when prox operators are simple.
    """
    m, n = A.shape
    x = np.zeros(n)
    y = np.zeros(n)
    t = 1.0
    history = {'objective': [], 'residual': []}

    for k in range(max_iter):
        # Gradient of the smooth part: ∇f(y) = Aᵀ(Ay - b)
        grad = A.T @ (A @ y - b)

        # Proximal update: solve prox_{λ * step * ||·||₁}(y - step * grad)
        x_new = prox_l1(y - step_size * grad, lam * step_size)

        # Compute objective value and residual
        obj = 0.5 * np.linalg.norm(A @ x_new - b)**2 + lam * np.linalg.norm(x_new, 1)
        res = np.linalg.norm(x_new - x)

        history['objective'].append(obj)
        history['residual'].append(res)

        if verbose:
            print(f"Iter {k:4d}: Obj = {obj:.6f}, Residual = {res:.2e}")

        if res < tol:
            break

        # Nesterov acceleration
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

    return x, history
