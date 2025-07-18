import numpy as np

def compute_objective(x, A, b, reg_type="lasso", alpha1=0.1, alpha2=0.1):
    """
    Compute f(x) = ½||Ax - b||² + h(x)
    - Lasso: h(x)=α1||x||₁
    - Ridge: h(x)=α2||x||² (added inside loss)
    - Elastic-Net: α1||x||₁ + α2||x||²
    """
    r = A @ x - b
    loss = 0.5 * r.dot(r)
    if reg_type == "lasso":
        reg = alpha1 * np.linalg.norm(x, 1)
    elif reg_type == "ridge":
        loss += alpha2 * x.dot(x)
        reg = 0.0
    elif reg_type == "elasticnet":
        reg = alpha1 * np.linalg.norm(x, 1) + alpha2 * x.dot(x)
    else:
        raise ValueError("Unsupported reg_type")
    return loss + reg