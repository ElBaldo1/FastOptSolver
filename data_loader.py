"""
FastOptSolver – data_loader.py
--------------------------------
Caricamento robusto di ./dataset/Housing.csv

Novità
------
* Parametro `target_col` facoltativo: se lo passi, quella colonna diventa
  il vettore risposta.  
* Se `target_col` è None:
    1. Usa "MEDV" se esiste ed è numerica.
    2. Altrimenti sceglie **l’ultima colonna numerica** del DataFrame.
    3. Se non trova colonne numeriche, lancia ValueError.
* Tutte le feature non numeriche (object/category) vengono convertite:
    - binarie "yes"/"no" → 1/0
    - le restanti → one-hot (get_dummies, drop_first=True)
* Opzione di normalizzazione sul train set soltanto (evita leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------#
# Helper
# ---------------------------------------------------------------------------#
def _dataset_path() -> Path:
    return Path(__file__).resolve().parent / "dataset" / "Housing.csv"


def _clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colonne object/category in float64 (yes/no o one-hot)."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object", "category"]).columns

    # normalizza stringhe
    for col in obj_cols:
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # yes/no → 1/0
    bin_cols = [c for c in obj_cols if set(out[c].unique()) <= {"yes", "no"}]
    for col in bin_cols:
        out[col] = out[col].map({"yes": 1, "no": 0})

    # one-hot resto
    remain = out.select_dtypes(include=["object", "category"]).columns
    if len(remain) > 0:
        out = pd.get_dummies(out, columns=remain, drop_first=True)

    return out.astype("float64")


# ---------------------------------------------------------------------------#
# Public API
# ---------------------------------------------------------------------------#
def load_housing(
    normalize: bool = False,
    test_size: float = 0.2,
    random_state: int | None = None,
    target_col: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    normalize : bool, default=False
        Se True applica StandardScaler (fit su X_train, transform su entrambi).
    test_size : float, default=0.2
        Quota di campioni per il test set.
    random_state : int or None
        Seed per train_test_split.
    target_col : str or None, default=None
        Nome esplicito della colonna target. Se None segue la logica descritta
        sopra (MEDV → ultima colonna numerica).

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    csv_path = _dataset_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Housing dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # -------------------------- determina la colonna target
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' non presente nel CSV")
    else:
        if "MEDV" in df.columns and pd.api.types.is_numeric_dtype(df["MEDV"]):
            target_col = "MEDV"
        else:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                raise ValueError(
                    "Impossibile individuare una colonna target numerica; "
                    "specifica target_col esplicitamente."
                )
            target_col = numeric_cols[-1]  # ultima numerica

    y = df[target_col].to_numpy(dtype=np.float64)

    # Features
    X_df = _clean_categoricals(df.drop(columns=target_col))
    X = X_df.to_numpy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Normalizzazione
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
