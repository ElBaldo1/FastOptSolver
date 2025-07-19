import numpy as np

def compute_objective(x, A, b, reg_type, alpha1, alpha2):
    """
    Compute f(x) = ½‖Ax - b‖² + h(x), where
      - h(x) = α1‖x‖₁ for Lasso
      - h(x) = α2‖x‖₂² for Ridge
      - h(x) = α1‖x‖₁ + α2‖x‖₂² for Elastic-Net
    """
    residual = A @ x - b
    loss = 0.5 * residual.dot(residual)
    if reg_type == 'lasso':
        reg = alpha1 * np.linalg.norm(x, 1)
    elif reg_type == 'ridge':
        loss += alpha2 * x.dot(x)
        reg = 0.0
    elif reg_type == 'elasticnet':
        reg = alpha1 * np.linalg.norm(x, 1) + alpha2 * x.dot(x)
    else:
        raise ValueError("Unsupported reg_type")
    return loss + reg