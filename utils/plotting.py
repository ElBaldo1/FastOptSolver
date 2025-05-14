import matplotlib.pyplot as plt

def plot_convergence(histories: dict, title="Convergence Plot", xlabel="Iteration", ylabel="Objective value"):
    """
    Plot convergence curves for multiple solvers.

    Parameters
    ----------
    histories : dict
        Dictionary where keys are solver names and values are lists of objective values.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    """
    plt.figure(figsize=(8, 5))

    for label, history in histories.items():
        plt.plot(history, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()