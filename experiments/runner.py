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
) -> Dict[str, object]:
    """
    Run an optimisation experiment for *n_iter* iterations.

    Parameters
    ----------
    solver_cls : Callable[..., BaseSolver]
        Solver class (e.g. ISTA, FISTA) – it will be instantiated as
        ``solver_cls(loss_obj)``.
    loss_obj : BaseLoss
        Loss function providing ``loss(X, y, w)``.
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.
    n_iter : int
        Number of calls to ``solver.step`` to perform.

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
    solver = solver_cls(loss_obj)
    history: List[float] = []

    with Timer() as timer:
        # Optional: log initial objective (w = 0) for nicer convergence curve
        solver.step(X, y)  # first step initialises weights and updates once
        history.append(loss_obj.loss(X, y, solver.w))

        for _ in range(n_iter - 1):
            solver.step(X, y)
            history.append(loss_obj.loss(X, y, solver.w))

    return {
        "solver": solver.__class__.__name__,
        "final_obj": history[-1],
        "elapsed": timer.elapsed,
        "iter": n_iter,
        "history": history,
    }
