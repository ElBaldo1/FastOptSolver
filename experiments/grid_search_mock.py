import pandas as pd
import numpy as np
from algorithms.ista import ISTA
from algorithms.fista import FISTA
from experiments.mock_benchmark import run_solver_on_mock
from algorithms.lbfgs import LBFGSSolver
from algorithms.dual_fista import DualFISTA

def run_grid_search_mock(
    n_samples=100,
    n_features=50,
    n_nonzero=10,
    noise_std=0.1,
    random_state=42
) -> pd.DataFrame:
    """
    Run grid search over solver, loss, alpha, alpha2 and step_size on mock data.

    Parameters
    ----------
    n_samples : int
        Number of samples in the synthetic dataset.
    n_features : int
        Number of features.
    n_nonzero : int
        Number of non-zero weights in the true weight vector.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Summary results for all experiment configurations.
    """
    solvers = [ISTA, FISTA, LBFGSSolver, DualFISTA]
    losses = ["lasso", "ridge", "elasticnet"]

    alphas = [0.001, 0.01, 0.1]
    alpha2s = [0.001, 0.01]  # Only used for ElasticNet
    step_sizes = [0.001, 0.01, 0.1]
    n_iter = 100

    results = []

    for solver in solvers:
        for loss in losses:
            for alpha in alphas:
                for step_size in step_sizes:

                    # Skip L-BFGS for Lasso (not differentiable)
                    if loss == "lasso" and solver == LBFGSSolver:
                        continue

                    # Skip DualFISTA for Ridge and ElasticNet (not a dual-friendly form)
                    if loss in ["ridge", "elasticnet"] and solver == DualFISTA:
                        continue

                    if loss == "elasticnet":
                        for alpha2 in alpha2s:
                            adaptive = solver in [LBFGSSolver, DualFISTA]
                            # ----------------------- ElasticNet experiment
                            res = run_solver_on_mock(
                                solver_cls=solver,
                                loss_name=loss,
                                alpha=alpha,
                                alpha2=alpha2,
                                step_size=step_size,
                                n_iter=n_iter,
                                n_samples=n_samples,
                                n_features=n_features,
                                n_nonzero=n_nonzero,
                                noise_std=noise_std,
                                random_state=random_state,
                                adaptive=adaptive
                            )
                            res.update({
                                "solver": solver.__name__,
                                "loss": loss,
                                "alpha": alpha,
                                "alpha2": alpha2,
                                "step_size": step_size
                            })

                            # ---------------- Recap stampa
                            print(f"[{solver.__name__} | {loss} | α={alpha}, α2={alpha2}, step={step_size}] "
                                  f"Final Obj: {res['final_obj']:.4e} in {res['iter']} iters")

                            results.append(res)
                    else:
                        adaptive = solver in [LBFGSSolver, DualFISTA]
                        # ----------------------- Lasso / Ridge experiment
                        res = run_solver_on_mock(
                            solver_cls=solver,
                            loss_name=loss,
                            alpha=alpha,
                            step_size=step_size,
                            n_iter=n_iter,
                            n_samples=n_samples,
                            n_features=n_features,
                            n_nonzero=n_nonzero,
                            noise_std=noise_std,
                            random_state=random_state,
                            adaptive=adaptive
                        )
                        res.update({
                            "solver": solver.__name__,
                            "loss": loss,
                            "alpha": alpha,
                            "alpha2": None,
                            "step_size": step_size
                        })

                        # ---------------- Recap stampa
                        print(f"[{solver.__name__} | {loss} | α={alpha}, step={step_size}] "
                              f"Final Obj: {res['final_obj']:.4e} in {res['iter']} iters")

                        results.append(res)

    df = pd.DataFrame(results)
    return df