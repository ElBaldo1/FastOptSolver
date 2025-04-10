import numpy as np
from data.generate import generate_synthetic_lasso_data
from solvers.ista import ista
from solvers.fista import fista
from analysis.performance import evaluate_solver
from analysis.plotting import plot_history

# Generate synthetic data
A, b, x_true = generate_synthetic_lasso_data(m=100, n=200, sparsity=10, noise_std=0.01)

# Set parameters
lambda_ = 0.1
L = np.linalg.norm(A.T @ A, 2)
step_size = 1.0 / L
max_iter = 1000
tol = 1e-6

# Run ISTA
x_ista, hist_ista = ista(A, b, lambda_, step_size, max_iter, tol, verbose=True)
results_ista = evaluate_solver(x_ista, x_true, hist_ista, name="ISTA")

# Run FISTA
x_fista, hist_fista = fista(A, b, lambda_, step_size, max_iter, tol, verbose=True)
results_fista = evaluate_solver(x_fista, x_true, hist_fista, name="FISTA")

# Plot results
plot_history([results_ista, results_fista])
