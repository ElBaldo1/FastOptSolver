"""
Gradient Descent Solver Module

Implements plain gradient descent for minimizing smooth functions such as the least-squares loss.
"""
import numpy as np

def gradient_descent(grad_f, x0, step_size, max_iter=1000, tol=1e-6, verbose=False, loss_f=None):
    """
    Performs gradient descent.

    Parameters:
        grad_f (function): Gradient of the loss function.
        x0 (np.ndarray): Starting point.
        step_size (float): Learning rate.
        max_iter (int): Maximum iterations.
        tol (float): Tolerance for convergence.
        verbose (bool): If True, prints progress logs.
        loss_f (function, optional): Function to compute objective value.

    Returns:
        dict: Contains final solution 'x' and a 'history' list with objective values per iteration.
    """
    x = x0.copy()
    history = []
    for k in range(max_iter):
        grad = grad_f(x)
        x_next = x - step_size * grad
        if np.linalg.norm(x_next - x) < tol:
            break
        x = x_next
        if loss_f:
            history.append(loss_f(x))
        else:
            history.append(None)
        if verbose and k % 10 == 0:
            logging.info(f"[GD] Iter {k:4d} - Step norm: {np.linalg.norm(x_next - x):.2e}")
    return {"x": x, "history": history}
