import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from data_loader import load_housing

def run_sklearn_baselines(alpha=0.1, alpha2=None, normalize=True):
    """
    Train Scikit-Learn baselines (Lasso, Ridge, ElasticNet) on housing data.

    Parameters
    ----------
    alpha : float
        Regularization strength for Lasso and Ridge.
    alpha2 : float or None
        L2 penalty for ElasticNet. Ignored for Lasso/Ridge.
    normalize : bool
        Whether to normalize features.

    Returns
    -------
    pd.DataFrame
        Summary of results with model type and final objective value.
    """
    X_train, _, y_train, _ = load_housing(normalize=normalize)

    results = []

    # ------------------------ Lasso
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_train, y_train)
    mse_lasso = np.mean((X_train @ lasso.coef_ - y_train) ** 2)
    l1_penalty = alpha * np.sum(np.abs(lasso.coef_))
    obj_lasso = 0.5 * mse_lasso + l1_penalty
    results.append({
        "model": "Lasso",
        "alpha": alpha,
        "alpha2": None,
        "final_obj": obj_lasso
    })

    # ------------------------ Ridge
    ridge = Ridge(alpha=alpha, max_iter=1000)
    ridge.fit(X_train, y_train)
    mse_ridge = np.mean((X_train @ ridge.coef_ - y_train) ** 2)
    l2_penalty = alpha * np.dot(ridge.coef_, ridge.coef_)
    obj_ridge = 0.5 * mse_ridge + l2_penalty
    results.append({
        "model": "Ridge",
        "alpha": alpha,
        "alpha2": None,
        "final_obj": obj_ridge
    })

    # ------------------------ ElasticNet
    if alpha2 is not None:
        enet = ElasticNet(alpha=alpha + alpha2, l1_ratio=alpha / (alpha + alpha2), max_iter=1000)
        enet.fit(X_train, y_train)
        mse_enet = np.mean((X_train @ enet.coef_ - y_train) ** 2)
        l1 = alpha * np.sum(np.abs(enet.coef_))
        l2 = alpha2 * np.dot(enet.coef_, enet.coef_)
        obj_enet = 0.5 * mse_enet + l1 + l2
        results.append({
            "model": "ElasticNet",
            "alpha": alpha,
            "alpha2": alpha2,
            "final_obj": obj_enet
        })

    return pd.DataFrame(results)