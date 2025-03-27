import numpy as np
import matplotlib.pyplot as plt

from solvers.ista import ista
from solvers.fista import fista

def generate_synthetic_data(m=100, n=50, sparsity=0.3, noise_std=0.05, seed=42):
    """
    Generate a random sparse regression problem.
    """
    np.random.seed(seed)
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    x_true[int(sparsity * n):] = 0
    np.random.shuffle(x_true)
    b = A @ x_true + noise_std * np.random.randn(m)
    return A, b, x_true

def compare_ista_fista():
    # Create data
    A, b, x_true = generate_synthetic_data()
    lam = 0.1
    L = np.linalg.norm(A.T @ A, 2)  # Lipschitz constant of ∇f
    step_size = 1.0 / L             # Step size for ISTA/FISTA

    # Run ISTA
    print("Running ISTA...")
    _, ista_hist = ista(A, b, lam, step_size, max_iter=200, tol=1e-8)

    # Run FISTA
    print("Running FISTA...")
    _, fista_hist = fista(A, b, lam, step_size, max_iter=200, tol=1e-8)

    # Compute the minimum objective value to normalize both curves
    min_obj = min(min(ista_hist['objective']), min(fista_hist['objective']))

    plt.figure(figsize=(8, 5))

    # Plot ISTA objective gap (above optimal value)
    plt.plot(
        np.array(ista_hist['objective']) - min_obj,
        label="ISTA",
        linewidth=2,
        linestyle='--',
        color='tab:blue'
    )

    # Plot FISTA objective gap (above optimal value)
    plt.plot(
        np.array(fista_hist['objective']) - min_obj,
        label="FISTA",
        linewidth=2,
        linestyle='-',
        color='tab:orange'
    )

    plt.yscale("log")  # Log scale reveals convergence speed differences
    plt.xlabel("Iteration")
    plt.ylabel("Objective gap (log scale)")
    plt.title("ISTA vs FISTA — Objective Convergence")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_ista_fista()
