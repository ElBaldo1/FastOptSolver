import numpy as np

def compute_objective(x, A, b, reg_type, alpha1, alpha2):
    """
    Compute f(x) = g(x) + h(x) with
      g(x) = ½||A x - b||₂²
             + ½*alpha2*||x||₂²   if reg_type in {"ridge","elasticnet"}
      h(x) =
             alpha1*||x||₁        if reg_type in {"lasso","elasticnet"}
             0                     if reg_type == "ridge"
    """
    # residual
    r = A @ x - b

    # g(x): always the data-fit term + ridge if needed
    g = 0.5 * r.dot(r)
    if reg_type in ("ridge", "elasticnet"):
        g += 0.5 * alpha2 * x.dot(x)

    # h(x): l1 only for lasso or elasticnet
    if reg_type == "lasso":
        h = alpha1 * np.linalg.norm(x, 1)
    elif reg_type == "elasticnet":
        h = alpha1 * np.linalg.norm(x, 1)
    elif reg_type == "ridge":
        h = 0.0
    else:
        raise ValueError(f"Unsupported reg_type='{reg_type}'")

    return g + h


def compute_reg_and_grad(x, reg_type, alpha1, alpha2):
    """
    Restituisce:
      - reg_loss = g(x)           (solo il termine di regolarizzazione)
      - reg_grad = ∇g(x)          (gradiente di quel termine)
    """
    if reg_type == "lasso":
        loss = alpha1 * np.linalg.norm(x, 1)
        grad = alpha1 * np.sign(x)
    elif reg_type == "ridge":
        loss = 0.5 * alpha2 * x.dot(x)
        grad = alpha2 * x
    elif reg_type == "elasticnet":
        loss = alpha1 * np.linalg.norm(x, 1) + 0.5 * alpha2 * x.dot(x)
        grad = alpha1 * np.sign(x) + alpha2 * x
    else:
        raise ValueError(...)
    return loss, grad
