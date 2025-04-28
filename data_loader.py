"""
FastOptSolver – data_loader.py
--------------------------------
Robust loader for ``./dataset/Housing.csv``.

Features
~~~~~~~~
* **Target detection**      – use the column named ``MEDV`` when present,
  otherwise fall back to the last column.
* **Categorical handling**  – every non-numeric feature column is converted:
    • binary **yes / no** (case-insensitive, trimmed) → 1 / 0  
    • all other object/category columns → one-hot encoded
      with ``pandas.get_dummies(drop_first=True)``
* **Numeric guarantee**     – the returned feature matrix is *pure float64*.
* **Optional scaling**      – if ``normalize=True`` a
  ``StandardScaler`` is **fitted on *X_train*** and then applied to both
  train and test sets (avoids data leakage).

Public API
~~~~~~~~~~
>>> from data_loader import load_housing
>>> X_train, X_test, y_train, y_test = load_housing(normalize=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------#
# Internal helpers
# ---------------------------------------------------------------------------#
def _dataset_path() -> Path:
    """Return absolute path to ``./dataset/Housing.csv``."""
    return Path(__file__).resolve().parent / "dataset" / "Housing.csv"


def _clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every non-numeric column in *df* to numeric.

    Steps
    -----
    1. Lower-case & trim whitespace in object/category columns.
    2. Map binary ``yes`` / ``no`` → 1 / 0.
    3. One-hot encode remaining object/category columns
       (``drop_first=True`` to avoid dummy trap).
    4. Cast the whole frame to ``float64``.

    Returns
    -------
    pd.DataFrame
        New DataFrame guaranteed to be numeric only.
    """
    out = df.copy()

    # -- 1: normalise string columns
    obj_cols = out.select_dtypes(include=["object", "category"]).columns
    for col in obj_cols:
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # -- 2: yes/no mapping (after lower-casing)
    bin_cols = [
        col for col in obj_cols if set(out[col].unique()) <= {"yes", "no"}
    ]
    for col in bin_cols:
        out[col] = out[col].map({"yes": 1, "no": 0})

    # -- 3: one-hot encode any still-categorical columns
    remaining = out.select_dtypes(include=["object", "category"]).columns
    if len(remaining) > 0:
        out = pd.get_dummies(out, columns=remaining, drop_first=True)

    # -- 4: ensure float64 dtype
    return out.astype("float64")


# ---------------------------------------------------------------------------#
# Public loader
# ---------------------------------------------------------------------------#
def load_housing(
    normalize: bool = False,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the Housing data set.

    Parameters
    ----------
    normalize : bool, default=False
        Standard-scale features if ``True`` (scaler fit on *train* split).

    test_size : float, default=0.2
        Proportion of samples allocated to the test set.

    random_state : int or None, default=None
        Seed used by ``train_test_split`` for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        All arrays have ``dtype=float64`` and are free of object dtypes.
    """
    csv_path = _dataset_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Housing dataset not found at {csv_path}")

    # ------------------------------- read CSV
    df_raw = pd.read_csv(csv_path)

    # -------------------------- choose target column
    target_col = "MEDV" if "MEDV" in df_raw.columns else df_raw.columns[-1]
    y = df_raw[target_col].to_numpy(dtype=np.float64)

    # --------------------------- clean feature matrix
    X_df = _clean_categoricals(df_raw.drop(columns=target_col))
    X = X_df.to_numpy()

    # ------------------------------ train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # ------------------------------ optional scaling
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
