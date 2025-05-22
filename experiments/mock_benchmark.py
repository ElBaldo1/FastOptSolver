import pandas as pd
import numpy as np
from data_loader import load_mock_data
from losses.lasso import LassoLoss
from losses.ridge import RidgeLoss
from losses.elastic_net import ElasticNetLoss
from algorithms.ista import ISTA
from algorithms.fista import FISTA
from experiments.runner import run_experiment
from utils.plotting import plot_convergence

def run_solver_on_mock(
    solver_cls,
    loss_name="lasso",
    alpha=0.1,
    alpha2=None,
    step_size=1e-2,
    n_iter=100,
    normalize=True,
    n_samples=100,
    n_features=50,
    n_nonzero=10,
    noise_std=0.1,
    random_state=None,
    verbose=False,
    adaptive=False
):
    """
    Run specified solver and loss on synthetic mock dataset.

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
    n_samples : int
        Number of samples in the synthetic dataset.
    n_features : int
        Number of features.
    n_nonzero : int
        Number of non-zero weights in the true weight vector.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_state : int or None
        Random seed.

    Returns
    -------
    dict
        Experiment result dictionary.
    """
    # ------------------------------- Load data
    X_train, _, y_train, _ = load_mock_data(
        n_samples=n_samples,
        n_features=n_features,
        n_nonzero=n_nonzero,
        noise_std=noise_std,
        random_state=random_state,
        normalize=normalize
    )

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