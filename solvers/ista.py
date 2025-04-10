"""
ISTA Solver Module

This module implements the Iterative Soft-Thresholding Algorithm (ISTA) for solving 
ℓ1-regularized linear regression problems (LASSO). The method performs a gradient 
descent step on the least-squares term and then applies a soft-thresholding operator 
(proximal operator) for the ℓ1 penalty.

Convergence: ISTA is known to converge at a rate of O(1/k) in function value.
"""

import numpy as np

def ista(grad_f, prox_g, x0, step_size, max_iter=1000, tol=1e-6, verbose=False, loss_f=None, track_sparsity=False):
    """
    Implements the ISTA algorithm.

    Parameters:
        grad_f (function): Function to compute the gradient of the smooth loss, f(x) = ½‖Ax-b‖².
        prox_g (function): Proximal operator for the nonsmooth regularization term.
        x0 (np.ndarray): Initial guess for the solution.
        step_size (float): Step-size for the gradient update, typically 1/L (L is Lipschitz constant).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        verbose (bool): If True, prints progress every 10 iterations.
        loss_f (function, optional): Function to compute the objective value for logging.
        track_sparsity (bool): If True, records sparsity (fraction of near-zero elements) per iteration.

    Returns:
        dict: Dictionary with keys:
              - "x": Final solution
              - "history": List of objective function values (if loss_f provided)
              - "sparsity_history": (Optional) List of sparsity metrics per iteration
    """
    x = x0.copy()
    history = []
    sparsity_history = []

    for k in range(max_iter):
        grad = grad_f(x)
        x_next = prox_g(x - step_size * grad, step_size)
        if np.linalg.norm(x_next - x) < tol:
            break
        x = x_next
        if loss_f:
            history.append(loss_f(x))
        else:
            history.append(None)
        if track_sparsity:
            sparsity = np.sum(np.abs(x) < 1e-6) / len(x)
            sparsity_history.append(sparsity)
        if verbose and k % 10 == 0:
            current_loss = loss_f(x) if loss_f is not None else np.nan
            logging.info(f"[ISTA] Iter {k:4d} - Loss: {current_loss:.4e}, Step norm: {np.linalg.norm(x_next - x):.2e}")

    result = {"x": x, "history": history}
    if track_sparsity:
        result["sparsity_history"] = sparsity_history
    return result
