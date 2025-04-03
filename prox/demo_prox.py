# prox/demo_prox.py

import numpy as np
from prox.prox_l1 import prox_l1
from prox.prox_l2 import prox_l2
from prox.prox_elasticnet import prox_elasticnet

def demo_prox_operators():
    """
    Demonstrates the behavior of prox_l1, prox_l2, and prox_elasticnet
    with a small input vector and fixed parameters.

    To be run from the notebook, not as a script.
    """
    v = np.array([3.0, -1.5, 0.3, 0.0])
    lam = 1.0
    alpha = 0.5

    print("Input vector:", v)
    print("")

    # --- prox_l1 ---
    out_l1 = prox_l1(v, lam)
    print("prox_l1(v, lam):")
    print(out_l1)
    print("")

    # --- prox_l2 ---
    out_l2 = prox_l2(v, lam)
    print("prox_l2(v, lam):")
    print(out_l2)
    print("")

    # --- prox_elasticnet ---
    out_en = prox_elasticnet(v, lam, alpha)
    print(f"prox_elasticnet(v, lam={lam}, alpha={alpha}):")
    print(out_en)
    print("")
