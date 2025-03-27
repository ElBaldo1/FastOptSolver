import numpy as np
from prox_l1 import prox_l1
from prox_l2 import prox_l2
from prox_elasticnet import prox_elasticnet

def run_prox_tests():
    v = np.array([3.0, -1.5, 0.3, 0.0])
    lam1 = 1.0
    lam2 = 2.0

    print("Input vector:", v)
    print("")

    # Test prox_l1
    out_l1 = prox_l1(v, lam1)
    print("prox_l1(v, lam1):")
    print(out_l1)
    print("")

    # Test prox_l2
    out_l2 = prox_l2(v, lam2)
    print("prox_l2(v, lam2):")
    print(out_l2)
    print("")

    # Test prox_elasticnet
    out_elastic = prox_elasticnet(v, lam1, lam2)
    print("prox_elasticnet(v, lam1, lam2):")
    print(out_elastic)
    print("")

if __name__ == "__main__":
    run_prox_tests()
