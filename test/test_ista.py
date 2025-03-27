import numpy as np
import matplotlib.pyplot as plt
from solvers.ista import ista

def generate_synthetic_data(m=100, n=50, sparsity=0.3, noise_std=0.05, seed=42):
    np.random.seed(seed)
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    x_true[int(sparsity * n):] = 0  # Make x sparse
    np.random.shuffle(x_true)
    b = A @ x_true + noise_std * np.random.randn(m)
    return A, b, x_true

def test_ista():
    # Generate data
    A, b, x_true = generate_synthetic_data()

    # Set hyperparameters
    lam = 0.1
    L = np.linalg.norm(A.T @ A, 2)  # Lipschitz constant of ∇f
    step_size = 1.0 / L

    # Run ISTA
    x_est, history = ista(A, b, lam, step_size, max_iter=500, tol=1e-6, verbose=True)

    # Report results
    print("\nGround truth x (non-zero entries):", np.nonzero(x_true)[0])
    print("Estimated x (non-zero entries):", np.nonzero(np.abs(x_est) > 1e-3)[0])
    print("ℓ2 error: ", np.linalg.norm(x_est - x_true))

    # Plot convergence
    plt.figure(figsize=(8, 4))
    plt.plot(history['objective'], label='Objective')
    plt.title("ISTA Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_ista()
