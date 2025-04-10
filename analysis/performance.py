import numpy as np
import time

def evaluate_solver(x, x_true, history, name="Solver"):
    """
    Evaluate performance of a solver: error, sparsity, time, etc.

    Parameters
    ----------
    x : np.ndarray
        Final solution vector
    x_true : np.ndarray
        Ground truth solution
    history : dict
        Dictionary with 'objective' and/or 'residual'
    name : str
        Name of the solver

    Returns
    -------
    results : dict
        Dictionary with all performance metrics
    """
    error = np.linalg.norm(x - x_true)
    sparsity = np.count_nonzero(x)
    time_elapsed = history.get("time", None)
    objective = history.get("objective", [])
    residual = history.get("residual", [])

    return {
        "name": name,
        "x": x,
        "error": error,
        "sparsity": sparsity,
        "objective": objective,
        "residual": residual,
        "time": time_elapsed if time_elapsed else "N/A"
    }
