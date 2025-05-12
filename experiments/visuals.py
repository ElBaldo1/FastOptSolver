# experiments/visuals.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_all_heatmaps(df: pd.DataFrame, dataset_label: str = "Housing") -> None:
    """
    Display a grid of heatmaps showing the sensitivity of final objective value 
    to alpha and step size for each solver + loss combination.

    Parameters
    ----------
    df : pd.DataFrame
        Grid search results with columns: solver, loss, alpha, step_size, final_obj
    dataset_label : str
        Label for display purposes (e.g. 'Housing', 'Mock')
    """
    solvers = df["solver"].unique()
    losses = df["loss"].unique()

    for solver in solvers:
        for loss in losses:
            df_sub = df[(df["solver"] == solver) & (df["loss"] == loss)]

            if df_sub.empty:
                continue

            pivot = df_sub.pivot_table(index="alpha", columns="step_size", values="final_obj")

            if pivot.isnull().all().all():
                continue

            plt.figure(figsize=(6, 4))
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu")
            plt.title(f"{dataset_label} – {solver} + {loss.capitalize()}")
            plt.xlabel("Step Size")
            plt.ylabel("Alpha")
            plt.tight_layout()
            plt.show()