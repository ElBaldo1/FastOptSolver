import matplotlib.pyplot as plt

def plot_history(results_list):
    """
    Plot objective and residual histories for one or more solvers.

    Parameters
    ----------
    results_list : list of dict
        Each dict contains keys 'name', 'objective', 'residual'
    """
    plt.figure(figsize=(12, 4))

    # Plot objective values
    plt.subplot(1, 2, 1)
    for res in results_list:
        if res["objective"]:
            plt.plot(res["objective"], label=res["name"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("Objective vs Iteration")
    plt.grid(True)
    plt.legend()

    # Plot residuals
    plt.subplot(1, 2, 2)
    for res in results_list:
        if res["residual"]:
            plt.plot(res["residual"], label=res["name"])
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Residual vs Iteration")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
