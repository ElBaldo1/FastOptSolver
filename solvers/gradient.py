# solvers/gradient.py

import numpy as np

def gradient_descent(
    grad_f,
    x0,
    step_size,
    max_iter=1000,
    tol=1e-6,
    verbose=False,
    loss_f=None
):
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
            current_loss = loss_f(x) if loss_f is not None else np.nan
            print(f"[GD] Iter {k:4d} - Loss: {current_loss:.4e}, Step norm: {np.linalg.norm(x_next - x):.2e}")

    return {
        "x": x,
        "history": history
    }
