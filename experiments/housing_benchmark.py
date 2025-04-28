import pandas as pd
import numpy as np
from data_loader import load_housing
from losses.lasso import LassoLoss
from algorithms.ista import ISTA
from algorithms.fista import FISTA
from experiments.runner import run_experiment
from utils.plotting import plot_convergence

def run_ista_vs_fista_on_housing(n_iter=100):
    """Run comparison experiment between ISTA and FISTA on housing dataset.
    
    Args:
        n_iter: Number of iterations to run each algorithm
        
    Returns:
        pd.DataFrame: Results dataframe with columns:
            - solver: Solver name
            - final_obj: Final objective value
            - elapsed: Total time in seconds
            - iter: Number of iterations completed
            - history: List of objective values at each iteration
    """
    # Load and prepare data
    X_train, _, y_train, _ = load_housing(normalize=True)
    
    # Create loss function
    loss = LassoLoss(alpha=0.1)
    
    # Run experiments
    ista_results = run_experiment(ISTA, loss, X_train, y_train, n_iter)
    fista_results = run_experiment(FISTA, loss, X_train, y_train, n_iter)
    
    # Create results DataFrame
    results = pd.DataFrame([ista_results, fista_results])
    
    # Plot convergence
    plot_convergence({
        'ISTA': ista_results['history'],
        'FISTA': fista_results['history']
    })
    
    return results