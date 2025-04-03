import numpy as np

from prox.prox_l1 import prox_l1
from prox.prox_l2 import prox_l2
from prox.prox_elasticnet import prox_elasticnet


def run_prox_tests():
    """
    Executes test cases for the proximity operators of:
    - L1 norm        → soft-thresholding
    - L2 squared norm → uniform shrinkage
    - Elastic Net     → composition of both

    These tests verify the correct behavior of the operators
    on a fixed input vector.
    """

    # Define input vector and regularization parameters
    v = np.array([3.0, -1.5, 0.3, 0.0])
    lam1 = 1.0  # used for L1 and Elastic Net
    lam2 = 2.0  # used for L2 and Elastic Net

    print("=== Proximity Operator Tests ===")
    print(f"Input vector v: {v}")
    print(f"L1 parameter λ₁ = {lam1}")
    print(f"L2 parameter λ₂ = {lam2}")
    print("")

    # --- Test L1 Proximity Operator (Soft-Thresholding) ---
    print("prox_l1(v, λ₁):  [solves min_x ½‖x - v‖² + λ₁‖x‖₁]")
    out_l1 = prox_l1(v, lam1)
    print("Output:", out_l1)
    print("")

    # --- Test L2 Proximity Operator (Shrinkage) ---
    print("prox_l2(v, λ₂):  [solves min_x ½‖x - v‖² + (λ₂/2)‖x‖²]")
    out_l2 = prox_l2(v, lam2)
    print("Output:", out_l2)
    print("")

    # --- Test Elastic Net Proximity Operator ---
    print("prox_elasticnet(v, λ₁, λ₂):  [min_x ½‖x - v‖² + λ₁‖x‖₁ + (λ₂/2)‖x‖²]")
    out_elastic = prox_elasticnet(v, lam1, lam2)
    print("Output:", out_elastic)
    print("")

if __name__ == "__main__":
    run_prox_tests()
