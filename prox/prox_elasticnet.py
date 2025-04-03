# prox/test_prox.py

import numpy as np
from prox.prox_l1 import prox_l1
from prox.prox_l2 import prox_l2
from prox.prox_elasticnet import prox_elasticnet

def test_prox_l1():
    x = np.array([3.0, -1.0, 0.5, -0.2, 0.0])
    lam = 0.5
    expected = np.array([2.5, -0.5, 0.0, -0.0, 0.0])
    result = prox_l1(x, lam)
    assert np.allclose(result, expected), "prox_l1 failed!"
    print("✅ prox_l1 passed")

def test_prox_l2():
    x = np.array([2.0, -4.0])
    lam = 0.5
    # prox for λ * ||x||² is x / (1 + 2λ)
    expected = x / (1 + 2 * lam)
    result = prox_l2(x, lam)
    assert np.allclose(result, expected), "prox_l2 failed!"
    print("✅ prox_l2 passed")

def test_prox_elasticnet():
    x = np.array([3.0, -1.0])
    lam = 1.0
    alpha = 0.5
    l1 = alpha * lam     # 0.5
    l2 = (1 - alpha) * lam  # 0.5
    scaled = x / (1 + 2 * l2)  # x / 2
    expected = prox_l1(scaled, l1)
    result = prox_elasticnet(x, lam, alpha)
    assert np.allclose(result, expected), "prox_elasticnet failed!"
    print("✅ prox_elasticnet passed")

if __name__ == "__main__":
    test_prox_l1()
    test_prox_l2()
    test_prox_elasticnet()
