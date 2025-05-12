import pandas as pd
import numpy as np
from algorithms.ista import ISTA
from algorithms.fista import FISTA
from experiments.housing_benchmark import run_solver_on_housing
from algorithms.lbfgs import LBFGSSolver

def run_grid_search():
    """
    Run grid search over solver, loss, alpha, alpha2 and step_size.

    Returns
    -------
    pd.DataFrame
        Summary results for all experiment configurations.
    """
    solvers = [ISTA, FISTA, LBFGSSolver]
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

                    if loss == "elasticnet":
                        for alpha2 in alpha2s:
                            # ----------------------- ElasticNet experiment
                            res = run_solver_on_housing(
                                solver_cls=solver,
                                loss_name=loss,
                                alpha=alpha,
                                alpha2=alpha2,
                                step_size=step_size,
                                n_iter=n_iter
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
                        # ----------------------- Lasso / Ridge experiment
                        res = run_solver_on_housing(
                            solver_cls=solver,
                            loss_name=loss,
                            alpha=alpha,
                            step_size=step_size,
                            n_iter=n_iter
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

    # Collect results
    df = pd.DataFrame(results)
    return df