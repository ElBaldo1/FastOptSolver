import numpy as np
from tqdm import tqdm
from experiments.mock_benchmark import run_solver_on_mock

def profile_solver_multiple_runs_mock(
    solver_cls,
    loss_name,
    alpha,
    alpha2=None,
    step_size=0.01,
    n_iter=100,
    n_runs=5,
    random_state=42,
    normalize=True,
    **kwargs
):
    times, iters, objs, sparsities = [], [], [], []

    for seed in tqdm(range(n_runs), desc=f"{solver_cls.__name__} profiling [mock data]"):
        result = run_solver_on_mock(
            solver_cls=solver_cls,
            loss_name=loss_name,
            alpha=alpha,
            alpha2=alpha2,
            step_size=step_size,
            n_iter=n_iter,
            normalize=normalize,
            random_state=seed,
            verbose=False,
            adaptive=True,
            **kwargs
        )

        w = result.get("w", None)
        sparsity = np.mean(w == 0) if w is not None else np.nan

        times.append(result["elapsed"])
        iters.append(result["iter"])
        objs.append(result["final_obj"])
        sparsities.append(sparsity)

    return {
        "solver": solver_cls.__name__,
        "alpha": alpha,
        "step_size": step_size,
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "iter_mean": np.mean(iters),
        "iter_std": np.std(iters),
        "obj_mean": np.mean(objs),
        "obj_std": np.std(objs),
        "sparsity_mean": np.mean(sparsities),
        "sparsity_std": np.std(sparsities)
    }

def profile_solver_adaptive_mock(
    solver_cls,
    loss_name,
    alpha,
    alpha2=None,
    step_size=0.01,
    n_runs=5,
    random_state=42,
    normalize=True,
    **kwargs
):
    times, iters, objs, sparsities = [], [], [], []

    for seed in tqdm(range(n_runs), desc=f"{solver_cls.__name__} adaptive [mock data]"):
        result = run_solver_on_mock(
            solver_cls=solver_cls,
            loss_name=loss_name,
            alpha=alpha,
            alpha2=alpha2,
            step_size=step_size,
            adaptive=True,
            normalize=normalize,
            random_state=seed,
            verbose=False,
            **kwargs
        )

        w = result.get("w", None)
        sparsity = np.mean(w == 0) if w is not None else np.nan

        times.append(result["elapsed"])
        iters.append(result["iter"])
        objs.append(result["final_obj"])
        sparsities.append(sparsity)

    return {
        "solver": solver_cls.__name__,
        "alpha": alpha,
        "step_size": step_size,
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "iter_mean": np.mean(iters),
        "iter_std": np.std(iters),
        "obj_mean": np.mean(objs),
        "obj_std": np.std(objs),
        "sparsity_mean": np.mean(sparsities),
        "sparsity_std": np.std(sparsities)
    }