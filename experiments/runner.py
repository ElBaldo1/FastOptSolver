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
from algorithms.lbfgs import LBFGSSolver
from algorithms.dual_fista import DualFISTA


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

    If the solver does not support step(), fall back to full fit().
    """

    if solver_cls == DualFISTA:
        solver = solver_cls(
            loss_obj=loss_obj,
            step_size=step_size,
            max_iter=n_iter
        )
    elif solver_cls == LBFGSSolver:
        solver = solver_cls(
            loss_obj=loss_obj,
            max_iter=n_iter
        )
    else:
        solver = solver_cls(
            loss_obj=loss_obj,
            step_size=step_size,
            max_iter=n_iter
        )
    solver.verbose = verbose

    history = []

    with Timer() as timer:
        if isinstance(solver, (LBFGSSolver, DualFISTA)):
            solver.fit(X, y)
            history = solver.history_["loss"]
        else:
            solver.step(X, y)
            history.append(loss_obj.loss(X, y, solver.w))
            for _ in range(n_iter - 1):
                solver.step(X, y)
                history.append(loss_obj.loss(X, y, solver.w))

    final_obj = history[-1]

    if verbose:
        print(f"[Recap] Solver: {solver.__class__.__name__} | Final Loss: {final_obj:.6f} | Time: {timer.elapsed:.2f}s")

    return {
        "solver": solver.__class__.__name__,
        "final_obj": final_obj,
        "elapsed": timer.elapsed,
        "iter": n_iter,
        "history": history,
    }