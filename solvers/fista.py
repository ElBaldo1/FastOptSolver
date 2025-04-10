"""
FISTA Solver Module

This module implements the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) which 
accelerates ISTA by introducing a momentum (extrapolation) term. FISTA is used to solve 
composite convex optimization problems such as LASSO and Elastic Net.

Convergence: FISTA typically converges at a rate of O(1/k²) in function value.
"""

import numpy as np

def fista(grad_f, prox_g, x0, step_size, max_iter=1000, tol=1e-6, verbose=False, loss_f=None, track_sparsity=False):
    """
    Implements the FISTA algorithm.

    Parameters: 
        grad_f (function): Function to compute the gradient of the smooth loss.
        prox_g (function): Proximal operator for the nonsmooth regularization term.
        x0 (np.ndarray): Initial solution estimate.
        step_size (float): Step size for the gradient update (ideally 1/L).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        verbose (bool): If True, prints progress every 10 iterations.
        loss_f (function, optional): Function to compute current objective value.
        track_sparsity (bool): If True, records and returns the sparsity of x per iteration.

    Returns:
        dict: Containing final solution 'x', 'history' of objective values, and optionally 'sparsity_history'.
    """
    x = x0.copy()
    y = x0.copy()
    t = 1
    history = []
    sparsity_history = []

    for k in range(max_iter):
        grad = grad_f(y)
        x_next = prox_g(y - step_size * grad, step_size)
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_next + ((t - 1) / t_next) * (x_next - x)
        if np.linalg.norm(x_next - x) < tol:
            break
        x = x_next
        t = t_next
        if loss_f:
            history.append(loss_f(x))
        else:
            history.append(None)
        if track_sparsity:
            sparsity = np.sum(np.abs(x) < 1e-6) / len(x)
            sparsity_history.append(sparsity)
        if verbose and k % 10 == 0:
            current_loss = loss_f(x) if loss_f is not None else np.nan
            logging.info(f"[FISTA] Iter {k:4d} - Loss: {current_loss:.4e}, Step norm: {np.linalg.norm(x_next - x):.2e}")

    result = {"x": x, "history": history}
    if track_sparsity:
        result["sparsity_history"] = sparsity_history
    return result
