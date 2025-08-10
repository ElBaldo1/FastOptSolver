# FastOptSolver

FastOptSolver provides reference implementations of several first-order and quasi-Newton algorithms for solving quadratic problems with L1 and L2 regularization. The project is organized to facilitate comparisons between different solvers and configurations.

## Modules

- **`iterative_solvers.py`** – infrastructure for timing and three iterative methods:
  - `ista`: proximal gradient with optional Armijo backtracking and logging of step sizes and convergence metrics.
  - `fista`: accelerated variant supporting backtracking, adaptive restarts, and multiple stopping criteria.
  - `fista_delta`: FISTA with a fixed momentum parameter \(\delta > 2\).
- **`lbfgs.py`** – wrapper around SciPy's L-BFGS-B for ridge and smooth elastic-net problems.
- **`objective_functions.py`** – computes the composite objective \(f(x)=g(x)+h(x)\) combining squared loss and optional L1 or L2 penalties.
- **`prox_operators.py`** – proximal operators for L1 and elastic-net penalties.
- **`easy_boston_data.py`** – synthetic data generator with correlated feature blocks.
- **`benchmark.ipynb`** – example notebook for experimenting with the solvers.

## Usage

```python
from easy_boston_data import generate_correlated_boston_like_data
from iterative_solvers import fista, ista, fista_delta, get_metrics
from lbfgs import LBFGSSolver
from prox_operators import prox_l1
import numpy as np

A, b, x_true = generate_correlated_boston_like_data()

# ISTA for Lasso
L = np.linalg.norm(A, 2)**2
x0 = np.zeros(A.shape[1])

def g(z):
    r = A @ z - b
    return 0.5 * r.dot(r)

def grad_g(z):
    return A.T @ (A @ z - b)

x_hat = ista(x0, g, grad_g, prox_l1, L, backtracking=True)

# FISTA with adaptive restart
x_fista, history = fista(A, b, reg_type="lasso", alpha1=0.1, alpha2=0.0,
                         backtracking=True, adaptive_restart=True,
                         return_history=True)

# FISTA-Δ with δ = 3
x_delta = fista_delta(A, b, reg_type="lasso", alpha1=0.1, alpha2=0.0,
                      delta=3.0, backtracking=True)

# L-BFGS for ridge regression
solver = LBFGSSolver("ridge", alpha1=0.0, alpha2=0.1)
solver.fit(A, b)

metrics = get_metrics()
```

The example above generates a dataset, applies several solvers, and collects timing metrics. The repository can be extended with additional algorithms or benchmark scripts to further compare ISTA, FISTA, and L-BFGS variants.

## Requirements

- Python 3.12+
- NumPy
- SciPy
- PyTest (for running the test suite)

Install dependencies with `pip install -r requirements.txt` if such a file is provided, or install the packages manually.

## Testing

Run the (empty) test suite with:

```bash
python -m pytest
```

## License

Specify your preferred license here.

