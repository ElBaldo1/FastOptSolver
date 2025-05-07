"""
experiments.runner
==================

Utility to benchmark a solver class for a fixed number of iterations.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from utils.timer import Timer
from algorithms.base_solver import BaseSolver
from losses.base_loss import BaseLoss


def run_experiment(
    solver_cls: Callable[..., BaseSolver],
    loss_obj: BaseLoss,
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int,
    step_size: float = 1e-2,
    verbose: bool = False
) -> Dict[str, object]:
    """
    Run an optimisation experiment for *n_iter* iterations.

    Parameters
    ----------
    solver_cls : Callable[..., BaseSolver]
        Solver class (e.g. ISTA, FISTA).
    loss_obj : BaseLoss
        Loss function providing loss(X, y, w).
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.
    n_iter : int
        Number of iterations.
    step_size : float
        Learning rate / step size for the solver.

    Returns
    -------
    dict
        {
            "solver": str,
            "final_obj": float,
            "elapsed": float,
            "iter": int,
            "history": List[float],
        }
    """
    if step_size >= 1.0:
        raise ValueError("Step size (lambda) must be less than 1 to guarantee convergence.")

    # ---------------------------------------------------- Instantiate solver
    solver = solver_cls(
        loss_obj,
        step_size=step_size,
        max_iter=n_iter
    )
    solver.verbose = verbose
    history: List[float] = []

    with Timer() as timer:
        # Optional: log initial objective (w = 0) for nicer convergence curve
        solver.step(X, y)  # first step initialises weights and updates once
        history.append(loss_obj.loss(X, y, solver.w))

        for _ in range(n_iter - 1):
            solver.step(X, y)
            history.append(loss_obj.loss(X, y, solver.w))

    final_obj = history[-1]

    # -----------------------------------------------------------
    # ADD RECAP LOGGING
    # -----------------------------------------------------------
    if verbose:
        print(f"[Recap] Solver: {solver.__class__.__name__} | Final Loss: {final_obj:.6f} | Time: {timer.elapsed:.2f}s")

    return {
        "solver": solver.__class__.__name__,
        "final_obj": history[-1],
        "elapsed": timer.elapsed,
        "iter": n_iter,
        "history": history,
    }