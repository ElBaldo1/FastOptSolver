import pandas as pd
import numpy as np
from data_loader import load_housing
from losses.lasso import LassoLoss
from losses.ridge import RidgeLoss
from losses.elastic_net import ElasticNetLoss
from algorithms.ista import ISTA
from algorithms.fista import FISTA
from experiments.runner import run_experiment
from utils.plotting import plot_convergence

def run_solver_on_housing(
    solver_cls,
    loss_name,
    alpha,
    alpha2=None,
    step_size=0.01,
    n_iter=100,
    normalize=True,
    random_state=None,
    verbose=False,
    adaptive=False
):
    """
    Run specified solver and loss on housing dataset.

    Parameters
    ----------
    solver_cls : class
        Solver class (e.g. ISTA, FISTA).
    loss_name : str
        Loss type: 'lasso', 'ridge' or 'elasticnet'.
    alpha : float
        Regularization strength (L1 for lasso and elasticnet, L2 for ridge).
    alpha2 : float or None
        L2 penalty for ElasticNet (ignored otherwise).
    step_size : float
        Step size / learning rate for the solver.
    n_iter : int
        Number of iterations.
    normalize : bool
        Whether to normalize features.
    adaptive : bool
        Whether to stop based on solver tolerance (instead of fixed iteration count).

    Returns
    -------
    dict
        Experiment result dictionary.
    """
    # ------------------------------- Load data
    X_train, _, y_train, _ = load_housing(normalize=normalize, random_state=random_state)

    # ------------------------------- Define loss
    if loss_name.lower() == "lasso":
        loss = LassoLoss(alpha=alpha)
    elif loss_name.lower() == "ridge":
        loss = RidgeLoss(alpha=alpha)
    elif loss_name.lower() == "elasticnet":
        if alpha2 is None:
            raise ValueError("ElasticNet requires both alpha and alpha2.")
        loss = ElasticNetLoss(alpha1=alpha, alpha2=alpha2)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    # ------------------------------- Run experiment
    results = run_experiment(
        solver_cls,
        loss,
        X_train,
        y_train,
        n_iter,
        step_size=step_size,
        verbose=verbose,
        adaptive=adaptive
    )

    return results