import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import numpy as np

def plot_convergence(history_dict: Dict[str, List[float]], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot convergence history of optimization algorithms.
    
    Args:
        history_dict: Dictionary mapping algorithm names to their objective values over iterations
        ax: Optional matplotlib Axes object to plot on. If None, creates new figure.
        
    Returns:
        plt.Axes: The Axes object containing the plot
        
    Example:
        >>> history = {'GD': [10, 5, 3, 2], 'SGD': [10, 6, 4, 3]}
        >>> ax = plot_convergence(history)
        >>> plt.show()
    """
    if not history_dict:
        raise ValueError("history_dict cannot be empty")
        
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
        
    for name, values in history_dict.items():
        iterations = np.arange(1, len(values) + 1)
        ax.plot(iterations, values, label=name, marker='o', markersize=4)
        
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Convergence History')
    ax.grid(True)
    ax.legend()
    
    return ax