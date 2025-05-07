import matplotlib.pyplot as plt

def plot_convergence(histories: dict):
    """
    Plot convergence curves for multiple solvers.

    Parameters
    ----------
    histories : dict
        Dictionary where keys are solver names and values are lists of objective values.
        Example:
            {
                'ISTA': [loss_1, loss_2, ..., loss_n],
                'FISTA': [loss_1, loss_2, ..., loss_n],
                ...
            }
    """
    plt.figure(figsize=(8, 5))

    for label, history in histories.items():
        plt.plot(history, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("Convergence comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()