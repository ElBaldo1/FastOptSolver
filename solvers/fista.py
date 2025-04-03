# solvers/fista.py

import numpy as np

def fista(
    grad_f,
    prox_g,
    x0,
    step_size,
    max_iter=1000,
    tol=1e-6,
    verbose=False,
    loss_f=None,
    track_sparsity=False
):
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
            print(f"[FISTA] Iter {k:4d} - Step norm: {np.linalg.norm(x_next - x):.2e}")

    result = {"x": x, "history": history}
    if track_sparsity:
        result["sparsity_history"] = sparsity_history
    return result
