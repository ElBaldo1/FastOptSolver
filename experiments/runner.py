import numpy as np
from typing import Callable, Dict, List
from utils.timer import Timer
from algorithms.base_solver import BaseSolver
from losses.base_loss import BaseLoss

def run_experiment(
    solver_cls: Callable[..., BaseSolver],
    loss_obj: BaseLoss,
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int
) -> Dict[str, float]:
    """
    Run optimization experiment with given solver and loss function.
    
    Args:
        solver_cls: Solver class to use (e.g. ISTA, FISTA)
        loss_obj: Loss function to optimize
        X: Input feature matrix (n_samples, n_features)
        y: Target values (n_samples, 1)
        n_iter: Number of iterations to run
        
    Returns:
        Dictionary containing:
        - solver: Solver name
        - final_obj: Final objective value
        - elapsed: Total time in seconds
        - iter: Number of iterations completed
        - history: List of objective values at each iteration
    """
    solver = solver_cls(loss_obj)
    history = []
    
    with Timer() as timer:
        for i in range(n_iter):
            solver.step(X, y)
            current_obj = loss_obj.loss(solver.w, X, y)
            history.append(current_obj)
            
    return {
        'solver': solver.__class__.__name__,
        'final_obj': history[-1],
        'elapsed': timer.elapsed,
        'iter': n_iter,
        'history': history
    }