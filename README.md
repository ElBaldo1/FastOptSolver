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

The main entry point is the Jupyter Notebook `FastOptSolver.ipynb`. It contains:

- Background theory and convergence analysis
- Implementation notes for each algorithm
- Comparative experiments across different losses, solvers, and hyperparameters
- Visual plots and convergence curves
- Discussion of numerical stability and gradient validity

The notebook is fully annotated with Markdown cells for pedagogical clarity and is structured to be self-contained.

## Project Structure

```
.
├── algorithms/            # Core optimization algorithms (ISTA, FISTA, DualFISTA, LBFGS)
├── losses/                # Lasso, Ridge, ElasticNet loss functions
├── experiments/           # Experiment runner, grid search, baselines, plotting tools
├── dataset/               # Contains Housing.csv (real dataset)
├── data_loader.py         # Robust loader for real and synthetic datasets
├── utils/                 # Timing utilities, helpers
├── FastOptSolver.ipynb    # Main notebook (report-style)
```

## Requirements

- Python ≥ 3.8
- NumPy
- SciPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Project

To explore the solvers and run the experiments:

1. Navigate to the project folder in your terminal
2. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
3. In the Jupyter interface, open the file `FastOptSolver.ipynb`
4. Execute the notebook cell by cell, following the included Markdown explanations for each section (theory, implementation, experiments)

## Dataset

- **Real dataset**: `Housing.csv` in the `dataset/` directory
- **Synthetic dataset**: generated on the fly using `load_mock_data()` from `data_loader.py`

## License

This project is distributed for academic and evaluation purposes only. No license is granted for commercial use.

Feel free to modify the notebook or the solvers to suit new models, constraints, or evaluation metrics.
