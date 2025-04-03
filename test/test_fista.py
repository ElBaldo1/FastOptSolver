import numpy as np
import matplotlib.pyplot as plt
from solvers.fista import fista

def generate_synthetic_data(m=100, n=50, sparsity=0.3, noise_std=0.05, seed=0):
    """
    Generates synthetic data for sparse linear regression:
        b = A x_true + noise,
    where x_true is sparse.
    """
    np.random.seed(seed)
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    x_true[int(sparsity * n):] = 0
    np.random.shuffle(x_true)
    b = A @ x_true + noise_std * np.random.randn(m)
    return A, b, x_true

def test_fista():
    """
    Runs and evaluates the FISTA algorithm for solving:
        min_x ½‖Ax - b‖² + λ‖x‖₁,
    using a synthetic dataset with known ground truth.
    """
    A, b, x_true = generate_synthetic_data()
    lam = 0.1

    # Compute Lipschitz constant of ∇f = AᵀA, needed for step size
    L = np.linalg.norm(A.T @ A, 2)
    step_size = 1.0 / L

    # Solve the optimization problem with FISTA
    x_est, history = fista(A, b, lam, step_size, max_iter=500, tol=1e-6, verbose=True)

    # Display comparison of sparsity pattern
    print("\n--- Sparsity pattern comparison ---")
    print("True non-zero indices:     ", np.nonzero(x_true)[0])
    print("Estimated non-zero indices:", np.nonzero(np.abs(x_est) > 1e-3)[0])

    # ℓ² error on the estimated solution
    l2_error = np.linalg.norm(x_est - x_true)
    print(f"\nℓ² error between estimated and true x: {l2_error:.6f}")

    # Plot convergence of the objective function
    plt.figure(figsize=(8, 4))
    plt.plot(history['objective'], label="FISTA objective")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("FISTA Convergence — Objective Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_fista()
