# FastOptSolver

FastOptSolver is a modular Python implementation of first-order and quasi-Newton solvers for sparse linear regression tasks. The project is centered around a Jupyter Notebook (`FastOptSolver.ipynb`) that guides the reader through the theoretical foundations, algorithmic differences, and experimental comparisons among multiple solvers.

## Overview

This project was developed in the context of the INFO-F524 (Continuous Optimization) course. It aims to provide a reproducible and extensible framework to compare the performance of:

- ISTA (Iterative Shrinkage-Thresholding Algorithm)
- FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
- Dual-FISTA (dual approach for Lasso)
- L-BFGS (Limited-memory BFGS using `scipy.optimize`)

All solvers are tested on both synthetic and real datasets, under different regularization strategies (L1, L2, Elastic Net).

## Main Interface

The main entry point is the `FastOptSolver.ipynb` Jupyter Notebook. It contains:

- Background theory and convergence analysis
- Implementation notes for each algorithm
- Comparative experiments across different losses, solvers, and hyperparameters
- Visual plots and convergence curves
- Discussion of numerical stability and gradient validity

The notebook is fully annotated with Markdown cells for pedagogical clarity and is structured to be self-contained.

## Project Structure

.
├── algorithms/            # Core optimization algorithms (ISTA, FISTA, DualFISTA, LBFGS)
├── losses/               # Lasso, Ridge, ElasticNet loss functions
├── experiments/          # Experiment runner, grid search, baselines, plotting tools
├── dataset/              # Contains Housing.csv (real dataset)
├── data_loader.py        # Robust loader for real and synthetic datasets
├── utils/                # Timing utilities, helpers
├── FastOptSolver.ipynb   # Main notebook (report-style)

## Features

- Modular solvers with step-wise and batch optimization
- Dual formulation of Lasso via Dual-FISTA
- Gradient validation and numerical safety checks
- Integration of sklearn baselines (Lasso, Ridge, ElasticNet)
- Grid search over solver × loss × hyperparameter combinations
- Heatmaps and visual performance comparisons

## Requirements

- Python ≥ 3.8
- NumPy, SciPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

To install dependencies:

```
pip install -r requirements.txt
```

## Running the Project

To explore the solvers and experiments:

1. Launch Jupyter:
   ```
   jupyter notebook
   ```
2. Open `FastOptSolver.ipynb`
3. Follow each section sequentially for code, theory, and experiments

## Dataset

- Real dataset: `Housing.csv` (Boston Housing-style)
- Synthetic dataset generator: sparse regression problems with controllable dimensionality and noise

## License

This project is distributed for academic and evaluation purposes only. No license is granted for commercial use.

Feel free to modify the notebook or the solvers to suit new models, constraints, or evaluation metrics.
